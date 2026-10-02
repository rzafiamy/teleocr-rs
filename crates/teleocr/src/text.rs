//! Text decoder: Qwen3-style blocks (per-head QK RMSNorm, GQA, SwiGLU)
//! with Qwen2-VL multimodal RoPE (t/h/w sections of the rotary dims).

use crate::config::Config;
use crate::weights::{Linear, RmsNorm, Weights};
use anyhow::Result;
use candle_core::quantized::QMatMul;
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::kv_cache::KvCache;

struct Layer {
    input_norm: RmsNorm,
    post_norm: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
}

enum Embed {
    Dense(Tensor),
    Quant(QMatMul),
}

pub struct TextModel {
    embed: Embed,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    cfg: Config,
    inv_freq: Vec<f32>,
    dtype: DType,
    device: Device,
}

/// Per-sequence decoding state.
pub struct Cache {
    kv: Vec<KvCache>,
    /// Next text position (same on the t/h/w axes once past the image).
    pub next_pos: usize,
}

impl TextModel {
    pub fn load(w: &Weights, cfg: &Config, dtype: DType) -> Result<Self> {
        let eps = cfg.rms_norm_eps;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let lin = |n: &str| w.linear(&format!("{p}.{n}"), false, dtype);
            layers.push(Layer {
                input_norm: RmsNorm::load(w, &format!("{p}.input_layernorm"), eps, dtype)?,
                post_norm: RmsNorm::load(w, &format!("{p}.post_attention_layernorm"), eps, dtype)?,
                q: lin("self_attn.q_proj")?,
                k: lin("self_attn.k_proj")?,
                v: lin("self_attn.v_proj")?,
                o: lin("self_attn.o_proj")?,
                q_norm: RmsNorm::load(w, &format!("{p}.self_attn.q_norm"), eps, dtype)?,
                k_norm: RmsNorm::load(w, &format!("{p}.self_attn.k_norm"), eps, dtype)?,
                gate: lin("mlp.gate_proj")?,
                up: lin("mlp.up_proj")?,
                down: lin("mlp.down_proj")?,
            });
        }
        let embed_name = "model.embed_tokens";
        let embed = match w.qmatmul(&format!("{embed_name}.weight"))? {
            Some(q) => Embed::Quant(q),
            None => Embed::Dense(w.get(&format!("{embed_name}.weight"), dtype)?),
        };
        let head = if !cfg.tie_word_embeddings && w.contains("lm_head.weight") {
            "lm_head"
        } else {
            embed_name
        };
        let hd = cfg.head_dim;
        let inv_freq = (0..hd / 2)
            .map(|i| 1.0 / (cfg.rope_theta as f32).powf((2 * i) as f32 / hd as f32))
            .collect();
        Ok(Self {
            embed,
            layers,
            norm: RmsNorm::load(w, "model.norm", eps, dtype)?,
            lm_head: w.linear(head, false, dtype)?,
            cfg: cfg.clone(),
            inv_freq,
            dtype,
            device: w.device.clone(),
        })
    }

    /// A cache with room for `max_len` positions; it grows by that much
    /// again whenever it fills up.
    pub fn new_cache(&self, max_len: usize) -> Cache {
        Cache {
            kv: (0..self.layers.len())
                .map(|_| KvCache::new(2, max_len))
                .collect(),
            next_pos: 0,
        }
    }

    pub fn embed(&self, ids: &[u32]) -> Result<Tensor> {
        let ids = Tensor::new(ids, &self.device)?;
        Ok(match &self.embed {
            Embed::Dense(w) => w.index_select(&ids, 0)?,
            Embed::Quant(q) => q.embedding(&ids)?.to_dtype(self.dtype)?,
        })
    }

    /// cos/sin `[seq, head_dim/2]` for positions `pos[axis][i]`.
    fn mrope(&self, pos: &[Vec<u32>; 3]) -> Result<(Tensor, Tensor)> {
        let n = pos[0].len();
        let half = self.cfg.head_dim / 2;
        let sec = &self.cfg.rope_scaling.mrope_section;
        let axis_of: Vec<usize> = (0..half)
            .map(|j| {
                let mut acc = 0;
                for (a, s) in sec.iter().enumerate() {
                    acc += s;
                    if j < acc {
                        return a;
                    }
                }
                sec.len() - 1
            })
            .collect();
        let inv = &self.inv_freq;
        let axis_of = &axis_of;
        let ang: Vec<f32> = (0..n)
            .flat_map(|i| (0..half).map(move |j| pos[axis_of[j]][i] as f32 * inv[j]))
            .collect();
        let a = Tensor::from_vec(ang, (n, half), &self.device)?;
        Ok((
            a.cos()?.to_dtype(self.dtype)?,
            a.sin()?.to_dtype(self.dtype)?,
        ))
    }

    /// Runs `xs` `[seq, hidden]` at positions `pos`; returns the logits of
    /// the last position `[vocab]` (F32).
    pub fn forward(&self, xs: &Tensor, pos: &[Vec<u32>; 3], cache: &mut Cache) -> Result<Tensor> {
        let (seq, _) = xs.dims2()?;
        let (cos, sin) = self.mrope(pos)?;
        let past = cache.kv[0].current_seq_len();
        let mask = if seq > 1 {
            Some(self.causal_mask(seq, past)?)
        } else {
            None
        };

        let mut x = xs.unsqueeze(0)?.to_dtype(self.dtype)?;
        for (l, kv) in self.layers.iter().zip(cache.kv.iter_mut()) {
            let h = l.input_norm.forward(&x)?;
            let h = self.attention(l, &h, &cos, &sin, mask.as_ref(), kv)?;
            x = (x + h)?;
            let h = l.post_norm.forward(&x)?;
            let h = l
                .down
                .forward(&(l.gate.forward(&h)?.silu()? * l.up.forward(&h)?)?)?;
            x = (x + h)?;
        }
        let last = x.narrow(1, seq - 1, 1)?;
        let logits = self.lm_head.forward(&self.norm.forward(&last)?)?;
        Ok(logits.flatten_all()?.to_dtype(DType::F32)?)
    }

    /// One decoding step for a batch of sequences sharing `cache`, whose
    /// rows are left-padded (`pad[i]` empty slots in front of row i).
    /// `tokens[i]` sits at text position `pos[i]`. Returns logits `[B, vocab]`.
    pub fn decode_batch(
        &self,
        tokens: &[u32],
        pos: &[u32],
        pad: &[usize],
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let b = tokens.len();
        let half = self.cfg.head_dim / 2;
        let ang: Vec<f32> = pos
            .iter()
            .flat_map(|&p| self.inv_freq.iter().map(move |f| p as f32 * f))
            .collect();
        let a = Tensor::from_vec(ang, (b, 1, half), &self.device)?;
        let (cos, sin) = (
            a.cos()?.to_dtype(self.dtype)?,
            a.sin()?.to_dtype(self.dtype)?,
        );
        let total = cache.kv[0].current_seq_len() + 1;
        let mask = if pad.iter().any(|&p| p > 0) {
            let m: Vec<f32> = pad
                .iter()
                .flat_map(|&p| (0..total).map(move |j| if j < p { f32::NEG_INFINITY } else { 0.0 }))
                .collect();
            Some(Tensor::from_vec(m, (b, 1, 1, total), &self.device)?)
        } else {
            None
        };

        let mut x = self
            .embed(tokens)?
            .reshape((b, 1, ()))?
            .to_dtype(self.dtype)?;
        for (l, kv) in self.layers.iter().zip(cache.kv.iter_mut()) {
            let h = l.input_norm.forward(&x)?;
            let h = self.attention(l, &h, &cos, &sin, mask.as_ref(), kv)?;
            x = (x + h)?;
            let h = l.post_norm.forward(&x)?;
            let h = l
                .down
                .forward(&(l.gate.forward(&h)?.silu()? * l.up.forward(&h)?)?)?;
            x = (x + h)?;
        }
        let logits = self.lm_head.forward(&self.norm.forward(&x)?)?;
        Ok(logits.reshape((b, ()))?.to_dtype(DType::F32)?)
    }

    /// Merges single-sequence caches into one batch cache with room for
    /// `extra` more steps; rows are left-padded to the longest. Returns the
    /// cache and each row's padding.
    pub fn stack_caches(&self, caches: &[Cache], extra: usize) -> Result<(Cache, Vec<usize>)> {
        let lens: Vec<usize> = caches.iter().map(|c| c.kv[0].current_seq_len()).collect();
        let longest = lens.iter().copied().max().unwrap_or(0);
        let pads: Vec<usize> = lens.iter().map(|n| longest - n).collect();
        let mut out = self.new_cache(longest + extra);
        for layer in 0..self.layers.len() {
            let mut ks = Vec::with_capacity(caches.len());
            let mut vs = Vec::with_capacity(caches.len());
            for (c, &pad) in caches.iter().zip(&pads) {
                let kv = &c.kv[layer];
                for (src, dst) in [(kv.k()?, &mut ks), (kv.v()?, &mut vs)] {
                    let t = src.ok_or_else(|| anyhow::anyhow!("empty cache"))?;
                    let t = if pad > 0 {
                        let (b, h, _, d) = t.dims4()?;
                        Tensor::cat(
                            &[&Tensor::zeros((b, h, pad, d), t.dtype(), t.device())?, &t],
                            2,
                        )?
                    } else {
                        t
                    };
                    dst.push(t);
                }
            }
            out.kv[layer].append(&Tensor::cat(&ks, 0)?, &Tensor::cat(&vs, 0)?)?;
        }
        Ok((out, pads))
    }

    /// Keeps only the batch rows `keep` (in that order).
    pub fn retain_rows(&self, cache: &mut Cache, keep: &[usize]) -> Result<()> {
        let idx = Tensor::new(
            keep.iter()
                .map(|&i| i as u32)
                .collect::<Vec<_>>()
                .as_slice(),
            &self.device,
        )?;
        for kv in cache.kv.iter_mut() {
            let cap = kv.k_cache().max_seq_len();
            let k = kv
                .k()?
                .ok_or_else(|| anyhow::anyhow!("empty cache"))?
                .contiguous()?
                .index_select(&idx, 0)?;
            let v = kv
                .v()?
                .ok_or_else(|| anyhow::anyhow!("empty cache"))?
                .contiguous()?
                .index_select(&idx, 0)?;
            let mut fresh = KvCache::new(2, cap);
            fresh.append(&k.contiguous()?, &v.contiguous()?)?;
            *kv = fresh;
        }
        Ok(())
    }

    fn causal_mask(&self, seq: usize, past: usize) -> Result<Tensor> {
        let total = past + seq;
        let m: Vec<f32> = (0..seq)
            .flat_map(|i| {
                (0..total).map(move |j| if j > past + i { f32::NEG_INFINITY } else { 0.0 })
            })
            .collect();
        Ok(Tensor::from_vec(m, (1, 1, seq, total), &self.device)?)
    }

    fn attention(
        &self,
        l: &Layer,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        kv: &mut KvCache,
    ) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let (nh, nkv, hd) = (
            self.cfg.num_attention_heads,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim,
        );
        let q = l.q.forward(x)?.reshape((b, seq, nh, hd))?;
        let k = l.k.forward(x)?.reshape((b, seq, nkv, hd))?;
        let v = l.v.forward(x)?.reshape((b, seq, nkv, hd))?;
        let q = l.q_norm.forward(&q)?.transpose(1, 2)?.contiguous()?;
        let k = l.k_norm.forward(&k)?.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let q = candle_nn::rotary_emb::rope(&q, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k, cos, sin)?;
        let (k, v) = kv.append(&k, &v)?;

        // GQA: view q as [b, nkv, rep*seq, hd] so each KV head serves its
        // group without materializing repeated K/V.
        let rep = nh / nkv;
        let total = k.dim(2)?;
        let qg = q.reshape((b, nkv, rep * seq, hd))?;
        // K and V are views of the preallocated cache: no per-step copy.
        let scores = (qg.matmul(&k.t()?)?.to_dtype(DType::F32)? / (hd as f64).sqrt())?;
        let scores = match mask {
            Some(m) => scores
                .reshape((b, nkv * rep, seq, total))?
                .broadcast_add(m)?
                .reshape((b, nkv, rep * seq, total))?,
            None => scores,
        };
        let p = candle_nn::ops::softmax(&scores, D::Minus1)?.to_dtype(v.dtype())?;
        let o = p.matmul(&v)?; // [b, nkv, rep*seq, hd]
        let o = o
            .reshape((b, nh, seq, hd))?
            .transpose(1, 2)?
            .reshape((b, seq, nh * hd))?;
        Ok(l.o.forward(&o)?)
    }
}
