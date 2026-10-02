//! Qwen2.5-VL vision tower: 14 px patches, 2-D RoPE, window attention
//! (112 px windows) with a few full-attention blocks, 2x2 patch merger.

use crate::config::VisionConfig;
use crate::image::Patches;
use crate::weights::{Linear, RmsNorm, Weights};
use anyhow::Result;
use candle_core::{D, DType, Device, Module, Tensor};
use std::collections::BTreeMap;

/// Query rows per chunk in full attention, at most; bounds the score matrix.
const FULL_ATTN_CHUNK: usize = 1024;
/// Score-matrix elements per full-attention chunk (heads x rows x keys):
/// 64M = 256 MB in F32. A fixed 1024-row chunk grows with the key count —
/// 1.5 GB per chunk on a 23k-patch crop, plus softmax copies.
const FULL_ATTN_SCORES: usize = 64 << 20;

struct Block {
    norm1: RmsNorm,
    norm2: RmsNorm,
    qkv: Linear,
    proj: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}

pub struct VisionTower {
    patch_embed: Tensor,
    blocks: Vec<Block>,
    ln_q: RmsNorm,
    mlp0: Linear,
    mlp2: Linear,
    cfg: VisionConfig,
    dtype: DType,
    device: Device,
}

/// Token layout of one image after the window reordering.
struct Layout {
    /// For each merged unit in window order, its row-major index.
    window_index: Vec<u32>,
    /// Token count of each window, in order (non-empty windows only).
    window_lens: Vec<usize>,
    /// RoPE angles `[n_tokens, head_dim / 2]` in window order.
    angles: Vec<f32>,
}

impl VisionTower {
    pub fn load(w: &Weights, cfg: &VisionConfig, dtype: DType) -> Result<Self> {
        let pe = w.get("visual.patch_embed.proj.weight", dtype)?;
        let patch_embed = pe.reshape((cfg.hidden_size, ()))?;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let p = format!("visual.blocks.{i}");
            blocks.push(Block {
                norm1: RmsNorm::load(w, &format!("{p}.norm1"), 1e-6, dtype)?,
                norm2: RmsNorm::load(w, &format!("{p}.norm2"), 1e-6, dtype)?,
                qkv: w.linear(&format!("{p}.attn.qkv"), true, dtype)?,
                proj: w.linear(&format!("{p}.attn.proj"), true, dtype)?,
                gate: w.linear(&format!("{p}.mlp.gate_proj"), true, dtype)?,
                up: w.linear(&format!("{p}.mlp.up_proj"), true, dtype)?,
                down: w.linear(&format!("{p}.mlp.down_proj"), true, dtype)?,
            });
        }
        Ok(Self {
            patch_embed,
            blocks,
            ln_q: RmsNorm::load(w, "visual.merger.ln_q", 1e-6, dtype)?,
            mlp0: w.linear("visual.merger.mlp.0", true, dtype)?,
            mlp2: w.linear("visual.merger.mlp.2", true, dtype)?,
            cfg: cfg.clone(),
            dtype,
            device: w.device.clone(),
        })
    }

    fn layout(&self, gh: usize, gw: usize) -> Layout {
        let m = self.cfg.spatial_merge_size;
        let unit = m * m;
        let (lh, lw) = (gh / m, gw / m);
        let win = self.cfg.window_size / m / self.cfg.patch_size;
        let (nwh, nww) = (lh.div_ceil(win), lw.div_ceil(win));

        let mut window_index = Vec::with_capacity(lh * lw);
        let mut window_lens = Vec::new();
        for wy in 0..nwh {
            for wx in 0..nww {
                let mut n = 0;
                for y in wy * win..((wy + 1) * win).min(lh) {
                    for x in wx * win..((wx + 1) * win).min(lw) {
                        window_index.push((y * lw + x) as u32);
                        n += 1;
                    }
                }
                if n > 0 {
                    window_lens.push(n * unit);
                }
            }
        }

        // 2-D RoPE: first half of the rotary dims from the patch row,
        // second half from the column.
        let head_dim = self.cfg.hidden_size / self.cfg.num_heads;
        let rot = head_dim / 2;
        let inv: Vec<f32> = (0..rot / 2)
            .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / rot as f32))
            .collect();
        let mut angles = Vec::with_capacity(lh * lw * unit * rot);
        for &u in &window_index {
            let (by, bx) = (u as usize / lw, u as usize % lw);
            for sy in 0..m {
                for sx in 0..m {
                    let (py, px) = ((by * m + sy) as f32, (bx * m + sx) as f32);
                    angles.extend(inv.iter().map(|f| py * f));
                    angles.extend(inv.iter().map(|f| px * f));
                }
            }
        }
        Layout {
            window_index,
            window_lens,
            angles,
        }
    }

    /// Encodes one image into `[n_tokens, out_hidden]` LLM embeddings.
    pub fn forward(&self, patches: &Patches) -> Result<Tensor> {
        let dev = &self.device;
        let unit = self.cfg.spatial_merge_size.pow(2);
        let n = patches.grid_h * patches.grid_w;
        let layout = self.layout(patches.grid_h, patches.grid_w);

        let x = Tensor::from_slice(&patches.data, (n, patches.dim), dev)?.to_dtype(self.dtype)?;
        let x = x.matmul(&self.patch_embed.t()?)?;

        // Reorder merged units (groups of `unit` tokens) into window order.
        let wi = Tensor::new(layout.window_index.as_slice(), dev)?;
        let hidden = self.cfg.hidden_size;
        let mut x = x
            .reshape((n / unit, unit, hidden))?
            .index_select(&wi, 0)?
            .reshape((n, hidden))?;

        let rot = hidden / self.cfg.num_heads / 2;
        let angles = Tensor::from_slice(&layout.angles, (n, rot), dev)?;
        let cos = angles.cos()?.to_dtype(self.dtype)?;
        let sin = angles.sin()?.to_dtype(self.dtype)?;

        let windows = WindowPlan::new(&layout.window_lens, dev)?;
        for (i, b) in self.blocks.iter().enumerate() {
            let full = self.cfg.fullatt_block_indexes.contains(&i);
            let h = b.norm1.forward(&x)?;
            let h = self.attention(b, &h, &cos, &sin, if full { None } else { Some(&windows) })?;
            x = (x + h)?;
            let h = b.norm2.forward(&x)?;
            let h = b
                .down
                .forward(&(b.gate.forward(&h)?.silu()? * b.up.forward(&h)?)?)?;
            x = (x + h)?;
        }

        let x = self.ln_q.forward(&x)?.reshape((n / unit, hidden * unit))?;
        let x = self.mlp2.forward(&self.mlp0.forward(&x)?.gelu_erf()?)?;

        // Back to row-major unit order.
        let mut inv = vec![0u32; layout.window_index.len()];
        for (pos, &u) in layout.window_index.iter().enumerate() {
            inv[u as usize] = pos as u32;
        }
        Ok(x.index_select(&Tensor::new(inv.as_slice(), dev)?, 0)?)
    }

    fn attention(
        &self,
        b: &Block,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        windows: Option<&WindowPlan>,
    ) -> Result<Tensor> {
        let (n, hidden) = x.dims2()?;
        let heads = self.cfg.num_heads;
        let hd = hidden / heads;
        let qkv = b.qkv.forward(x)?.reshape((n, 3, heads, hd))?;
        // [1, heads, n, hd] for candle's rope.
        let part = |i: usize| -> Result<Tensor> {
            Ok(qkv
                .narrow(1, i, 1)?
                .squeeze(1)?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()?)
        };
        let q = candle_nn::rotary_emb::rope(&part(0)?, cos, sin)?.squeeze(0)?;
        let k = candle_nn::rotary_emb::rope(&part(1)?, cos, sin)?.squeeze(0)?;
        let v = part(2)?.squeeze(0)?;
        let scale = 1.0 / (hd as f64).sqrt();

        // out: [heads, n, hd]
        let out = match windows {
            None => {
                let kt = k.t()?.contiguous()?;
                let mut chunks = Vec::new();
                let mut s = 0;
                while s < n {
                    let rows = (FULL_ATTN_SCORES / (heads * n)).clamp(64, FULL_ATTN_CHUNK);
                    let len = rows.min(n - s);
                    let qc = q.narrow(1, s, len)?;
                    chunks.push(sdpa(&qc, &kt, &v, scale)?);
                    s += len;
                }
                Tensor::cat(&chunks, 1)?
            }
            Some(plan) => plan.attend(&q, &k, &v, scale)?,
        };
        let out = out.transpose(0, 1)?.reshape((n, hidden))?;
        Ok(b.proj.forward(&out)?)
    }
}

/// softmax(q kᵀ · scale) v with the softmax in F32. `kt` is kᵀ.
fn sdpa(q: &Tensor, kt: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let dtype = q.dtype();
    let s = (q.matmul(kt)?.to_dtype(DType::F32)? * scale)?;
    let p = candle_nn::ops::softmax(&s, D::Minus1)?.to_dtype(dtype)?;
    Ok(p.matmul(v)?)
}

/// Windows grouped by token count, so each group runs as one batched
/// attention.
struct WindowPlan {
    /// (window length, token indices of the group's windows, window count)
    groups: Vec<(usize, Tensor, usize)>,
    /// Maps the concatenated group outputs back to sequence order.
    inverse: Tensor,
}

impl WindowPlan {
    fn new(lens: &[usize], dev: &Device) -> Result<Self> {
        let mut by_len: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut starts = Vec::with_capacity(lens.len());
        let mut s = 0;
        for &l in lens {
            starts.push(s);
            s += l;
        }
        for (i, &l) in lens.iter().enumerate() {
            by_len.entry(l).or_default().push(i);
        }
        let mut groups = Vec::new();
        let mut order: Vec<u32> = Vec::with_capacity(s);
        for (len, wins) in by_len {
            let idx: Vec<u32> = wins
                .iter()
                .flat_map(|&w| (starts[w]..starts[w] + len).map(|t| t as u32))
                .collect();
            order.extend_from_slice(&idx);
            groups.push((len, Tensor::new(idx.as_slice(), dev)?, wins.len()));
        }
        let mut inv = vec![0u32; s];
        for (pos, &t) in order.iter().enumerate() {
            inv[t as usize] = pos as u32;
        }
        Ok(Self {
            groups,
            inverse: Tensor::new(inv.as_slice(), dev)?,
        })
    }

    /// q, k, v: `[heads, n, hd]` → `[heads, n, hd]`.
    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
        let (heads, _, hd) = q.dims3()?;
        let mut outs = Vec::with_capacity(self.groups.len());
        for (len, idx, nw) in &self.groups {
            // [heads, nw*len, hd] → [heads*nw, len, hd]
            let g = |t: &Tensor| -> Result<Tensor> {
                Ok(t.index_select(idx, 1)?.reshape((heads * nw, *len, hd))?)
            };
            let (qg, kg, vg) = (g(q)?, g(k)?, g(v)?);
            let o = sdpa(&qg, &kg.t()?.contiguous()?, &vg, scale)?;
            outs.push(o.reshape((heads, nw * len, hd))?);
        }
        Ok(Tensor::cat(&outs, 1)?.index_select(&self.inverse, 1)?)
    }
}
