//! End-to-end inference: chat prompt with one image, vision encoding,
//! greedy decoding with the model's repetition penalty.

use crate::config::Config;
use crate::gguf;
use crate::image::{PreprocessConfig, preprocess};
use crate::text::{Cache, TextModel};
use crate::vision::VisionTower;
use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, Tensor};
use image::RgbImage;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

pub const SYSTEM_PROMPT: &str = "You are a helpful assistant.";
/// KV positions reserved past the prompt up front; the cache grows when a
/// longer output needs it (reserving `max_new_tokens` would cost ~230 KB per
/// position per sequence in F32).
const CACHE_HEADROOM: usize = 512;
/// End of turn and end of text.
const EOS: [u32; 2] = [151_645, 151_643];

/// Task prompts from the model card, verbatim.
pub fn task_prompt(task: &str) -> Option<&'static str> {
    Some(match task {
        "text" => "Please output the text content from the image.",
        "table" => "This is the image of a table. Please output the table in OTSL format.",
        "formula" => {
            "Please write out the expression of the formula in the image using LaTeX format."
        }
        "code" => "The image contains a code snippet, please output the parsing result.",
        "layout" => "Analyze the image layout.",
        "layout_seg" => "\nMulti-point Layout Segmentation Analysis.",
        "figure" => "This is a scientific figure. Please extract the table implied by this figure.",
        _ => return None,
    })
}

#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Compute dtype of the vision tower (BF16 on GPU, F32 on CPU by default).
    pub vision_dtype: Option<DType>,
    /// Compute dtype of the text decoder.
    pub text_dtype: Option<DType>,
}

/// Greedy decoding with logit penalties.
///
/// `repetition_penalty` follows transformers (prompt + output, divide
/// positive / multiply negative logits); `presence_penalty` and
/// `frequency_penalty` follow vLLM (output tokens only, subtracted);
/// `no_repeat_ngram_size` bans a token that would repeat an output n-gram.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct GenerateOptions {
    pub max_new_tokens: usize,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub no_repeat_ngram_size: usize,
}

impl Default for GenerateOptions {
    /// The model card's settings (`generation_config.json`).
    fn default() -> Self {
        Self {
            max_new_tokens: 4096,
            repetition_penalty: 1.05,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            no_repeat_ngram_size: 0,
        }
    }
}

impl GenerateOptions {
    /// Settings of the official document pipeline for a block type
    /// (`DEFAULT_SAMPLING_PARAMS` in TeleOCR_client.py).
    pub fn pipeline(block_type: &str) -> Self {
        let frequency_penalty = match block_type {
            "layout" | "layout_seg" => 0.0,
            "table" | "char" | "seal" | "figure" => 0.005,
            _ => 0.05,
        };
        let presence_penalty = if frequency_penalty == 0.0 { 0.0 } else { 1.0 };
        Self {
            max_new_tokens: 4096,
            repetition_penalty: 1.0,
            presence_penalty,
            frequency_penalty,
            no_repeat_ngram_size: 100,
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Timings {
    pub preprocess_ms: f64,
    pub vision_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub prompt_tokens: usize,
    pub image_tokens: usize,
    pub generated_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct Output {
    pub text: String,
    pub tokens: Vec<u32>,
    pub timings: Timings,
    /// True when generation stopped on `max_new_tokens`.
    pub truncated: bool,
}

pub struct Engine {
    pub cfg: Config,
    vision: VisionTower,
    text: TextModel,
    pub tokenizer: Tokenizer,
    pre: PreprocessConfig,
    device: Device,
}

impl Engine {
    /// Loads a `.gguf` file or a Hugging Face directory.
    pub fn load(path: &Path, device: &Device, opts: &LoadOptions) -> Result<Self> {
        let (weights, assets) = gguf::load(path, device)?;
        let cfg = Config::from_json(&assets.config)?;
        let tokenizer = Tokenizer::from_bytes(assets.tokenizer.as_bytes())
            .map_err(|e| anyhow!("tokenizer: {e}"))?;
        let mut pre = PreprocessConfig {
            patch_size: cfg.vision_config.patch_size,
            merge_size: cfg.vision_config.spatial_merge_size,
            temporal_patch_size: cfg.vision_config.temporal_patch_size,
            ..Default::default()
        };
        if let Some(p) = assets.preprocessor.as_deref() {
            let v: serde_json::Value = serde_json::from_str(p)?;
            if let Some(x) = v.get("min_pixels").and_then(|x| x.as_u64()) {
                pre.min_pixels = x as usize;
            }
            if let Some(x) = v.get("max_pixels").and_then(|x| x.as_u64()) {
                pre.max_pixels = x as usize;
            }
        }
        let gpu_default = if device.is_cpu() {
            DType::F32
        } else {
            DType::BF16
        };
        let vision = VisionTower::load(
            &weights,
            &cfg.vision_config,
            opts.vision_dtype.unwrap_or(gpu_default),
        )?;
        let text = TextModel::load(&weights, &cfg, opts.text_dtype.unwrap_or(DType::F32))?;
        Ok(Self {
            cfg,
            vision,
            text,
            tokenizer,
            pre,
            device: device.clone(),
        })
    }

    /// Preprocessed patches and vision embeddings `[n_tokens, hidden]`.
    pub fn encode_image(&self, img: &RgbImage) -> Result<(crate::image::Patches, Tensor)> {
        let patches = preprocess(img, &self.pre)?;
        let emb = self.vision.forward(&patches)?;
        Ok((patches, emb))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Caps the pixels an image is resized to (bounds vision cost and
    /// image tokens).
    pub fn set_max_pixels(&mut self, max_pixels: usize) {
        self.pre.max_pixels = max_pixels;
    }

    fn encode_text(&self, s: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(s, false)
            .map_err(|e| anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec())
    }

    /// Chat-template token ids for one image + prompt; returns the ids and
    /// the index of the first image token.
    pub fn prompt_ids(&self, prompt: &str, image_tokens: usize) -> Result<(Vec<u32>, usize)> {
        let mut ids = self.encode_text(&format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
        ))?;
        ids.push(self.cfg.vision_start_token_id);
        let start = ids.len();
        ids.extend(std::iter::repeat_n(self.cfg.image_token_id, image_tokens));
        ids.push(self.cfg.vision_end_token_id);
        ids.extend(self.encode_text(&format!("{prompt}<|im_end|>\n<|im_start|>assistant\n"))?);
        Ok((ids, start))
    }

    /// Vision encoding + prefill of one image and prompt, with room in the
    /// cache for `extra` decoding steps.
    fn prefill(&self, img: &RgbImage, prompt: &str, extra: usize) -> Result<Prefilled> {
        let mut tm = Timings::default();
        let t0 = Instant::now();
        let patches = preprocess(img, &self.pre)?;
        tm.preprocess_ms = ms(t0);

        let t0 = Instant::now();
        let m = self.cfg.vision_config.spatial_merge_size;
        let image_embeds = self.vision.forward(&patches)?;
        self.device.synchronize()?;
        tm.vision_ms = ms(t0);

        let n_img = patches.n_tokens(m);
        let (lh, lw) = (patches.grid_h / m, patches.grid_w / m);
        let (ids, start) = self.prompt_ids(prompt, n_img)?;
        tm.prompt_tokens = ids.len();
        tm.image_tokens = n_img;

        // Multimodal positions: text before the image counts up; image
        // tokens take (t, row, col) offset by that count; text after the
        // image resumes from the largest image position + 1.
        let mut pos: [Vec<u32>; 3] = Default::default();
        for i in 0..start {
            for p in pos.iter_mut() {
                p.push(i as u32);
            }
        }
        let base = start as u32;
        for r in 0..lh {
            for c in 0..lw {
                pos[0].push(base);
                pos[1].push(base + r as u32);
                pos[2].push(base + c as u32);
            }
        }
        let mut next = base + lh.max(lw) as u32;
        for _ in start + n_img..ids.len() {
            for p in pos.iter_mut() {
                p.push(next);
            }
            next += 1;
        }

        let t0 = Instant::now();
        let embeds = self.text.embed(&ids)?;
        let embeds = Tensor::cat(
            &[
                embeds.narrow(0, 0, start)?,
                image_embeds.to_dtype(embeds.dtype())?,
                embeds.narrow(0, start + n_img, ids.len() - start - n_img)?,
            ],
            0,
        )?;
        let mut cache = self.text.new_cache(ids.len() + extra.min(CACHE_HEADROOM));
        let logits = self.text.forward(&embeds, &pos, &mut cache)?;
        self.device.synchronize()?;
        tm.prefill_ms = ms(t0);
        Ok(Prefilled {
            ids,
            cache,
            logits: logits.to_vec1()?,
            next_pos: next,
            timings: tm,
        })
    }

    /// Runs one image + prompt; `on_token` receives each decoded text piece.
    pub fn generate(
        &self,
        img: &RgbImage,
        prompt: &str,
        opts: &GenerateOptions,
        mut on_token: Option<&mut dyn FnMut(&str)>,
    ) -> Result<Output> {
        let Prefilled {
            ids,
            mut cache,
            mut logits,
            next_pos: mut next,
            timings: mut tm,
        } = self.prefill(img, prompt, opts.max_new_tokens + 1)?;

        let t0 = Instant::now();
        let mut sampler = Sampler::new(opts, &ids);
        let mut out = Vec::new();
        let mut printed = 0usize;
        let mut truncated = true;
        for _ in 0..opts.max_new_tokens {
            let tok = sampler.pick(logits, &out);
            if EOS.contains(&tok) {
                truncated = false;
                break;
            }
            out.push(tok);
            sampler.push(&out);
            if let Some(cb) = on_token.as_deref_mut() {
                let s = self.decode(&out)?;
                // Hold back an incomplete UTF-8 sequence.
                if !s.ends_with('\u{fffd}') && s.len() > printed {
                    cb(&s[printed..]);
                    printed = s.len();
                }
            }
            let p = next;
            next += 1;
            let x = self.text.embed(&[tok])?;
            logits = self
                .text
                .forward(&x, &[vec![p], vec![p], vec![p]], &mut cache)?
                .to_vec1()?;
        }
        tm.decode_ms = ms(t0);
        tm.generated_tokens = out.len();
        let text = self.decode(&out)?;
        Ok(Output {
            text,
            tokens: out,
            timings: tm,
            truncated,
        })
    }

    /// Runs several (image, prompt, options) jobs, decoding up to
    /// `max_batch` of them together: each is prefilled alone, then their
    /// caches are stacked (left-padded) and decoded in lockstep. Batch-1
    /// decoding of this small model is bound by kernel launches, so a batch
    /// costs about as much per step as one sequence. Outputs are identical
    /// to `generate` up to floating-point noise.
    pub fn generate_batch(
        &self,
        jobs: &[(&RgbImage, &str, &GenerateOptions)],
        max_batch: usize,
    ) -> Result<Vec<Output>> {
        let mut outputs = Vec::with_capacity(jobs.len());
        for chunk in jobs.chunks(max_batch.max(1)) {
            outputs.extend(self.decode_chunk(chunk)?);
        }
        Ok(outputs)
    }

    fn decode_chunk(&self, jobs: &[(&RgbImage, &str, &GenerateOptions)]) -> Result<Vec<Output>> {
        if jobs.len() == 1 {
            let (img, prompt, opts) = jobs[0];
            return Ok(vec![self.generate(img, prompt, opts, None)?]);
        }
        let max_new = jobs.iter().map(|j| j.2.max_new_tokens).max().unwrap_or(0);
        let mut pre = Vec::with_capacity(jobs.len());
        for (img, prompt, _) in jobs {
            pre.push(self.prefill(img, prompt, 0)?);
        }
        let caches: Vec<Cache> = pre
            .iter_mut()
            .map(|p| std::mem::replace(&mut p.cache, self.text.new_cache(0)))
            .collect();
        let (mut cache, mut pads) = self
            .text
            .stack_caches(&caches, (max_new + 1).min(CACHE_HEADROOM))?;
        drop(caches);

        let t0 = Instant::now();
        let samplers: Vec<Sampler> = jobs
            .iter()
            .zip(&pre)
            .map(|(j, p)| Sampler::new(j.2, &p.ids))
            .collect();
        let mut outs: Vec<Vec<u32>> = vec![Vec::new(); jobs.len()];
        let mut truncated = vec![true; jobs.len()];
        let mut samplers = samplers;
        let mut next: Vec<u32> = pre.iter().map(|p| p.next_pos).collect();
        // Batch row → job index.
        let mut rows: Vec<usize> = (0..jobs.len()).collect();
        let mut logits: Vec<Vec<f32>> = pre
            .iter_mut()
            .map(|p| std::mem::take(&mut p.logits))
            .collect();

        loop {
            let mut keep = Vec::with_capacity(rows.len());
            let mut toks = Vec::with_capacity(rows.len());
            for (r, &j) in rows.iter().enumerate() {
                let opts = jobs[j].2;
                let done_len = outs[j].len() >= opts.max_new_tokens;
                let tok = samplers[j].pick(std::mem::take(&mut logits[r]), &outs[j]);
                if EOS.contains(&tok) {
                    truncated[j] = false;
                    continue;
                }
                if done_len {
                    continue;
                }
                outs[j].push(tok);
                samplers[j].push(&outs[j]);
                if outs[j].len() >= opts.max_new_tokens {
                    continue; // truncated
                }
                keep.push(r);
                toks.push(tok);
            }
            if keep.is_empty() {
                break;
            }
            if keep.len() < rows.len() {
                self.text.retain_rows(&mut cache, &keep)?;
                rows = keep.iter().map(|&r| rows[r]).collect();
                pads = keep.iter().map(|&r| pads[r]).collect();
            }
            let pos: Vec<u32> = rows.iter().map(|&j| next[j]).collect();
            for &j in &rows {
                next[j] += 1;
            }
            let l = self.text.decode_batch(&toks, &pos, &pads, &mut cache)?;
            logits = l.to_vec2()?;
        }
        let decode_ms = ms(t0);

        let mut res = Vec::with_capacity(jobs.len());
        for (j, p) in pre.into_iter().enumerate() {
            let mut tm = p.timings;
            tm.decode_ms = decode_ms;
            tm.generated_tokens = outs[j].len();
            res.push(Output {
                text: self.decode(&outs[j])?,
                tokens: std::mem::take(&mut outs[j]),
                timings: tm,
                truncated: truncated[j],
            });
        }
        Ok(res)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow!("detokenize: {e}"))
            .context("decode")
    }
}

struct Prefilled {
    ids: Vec<u32>,
    cache: Cache,
    /// Logits of the last prompt position.
    logits: Vec<f32>,
    /// Text position of the first generated token.
    next_pos: u32,
    timings: Timings,
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

struct Sampler<'a> {
    opts: &'a GenerateOptions,
    /// Prompt tokens (for the transformers-style repetition penalty).
    prompt: HashSet<u32>,
    counts: HashMap<u32, u32>,
    /// (n-1)-gram prefix → tokens that followed it in the output.
    ngrams: HashMap<Vec<u32>, Vec<u32>>,
}

impl<'a> Sampler<'a> {
    fn new(opts: &'a GenerateOptions, prompt: &[u32]) -> Self {
        Self {
            opts,
            prompt: prompt.iter().copied().collect(),
            counts: HashMap::new(),
            ngrams: HashMap::new(),
        }
    }

    /// Records the last output token.
    fn push(&mut self, out: &[u32]) {
        let tok = out[out.len() - 1];
        *self.counts.entry(tok).or_default() += 1;
        let n = self.opts.no_repeat_ngram_size;
        if n > 1 && out.len() >= n {
            let prefix = out[out.len() - n..out.len() - 1].to_vec();
            self.ngrams.entry(prefix).or_default().push(tok);
        }
    }

    fn pick(&self, mut v: Vec<f32>, out: &[u32]) -> u32 {
        let o = self.opts;
        if o.repetition_penalty != 1.0 {
            for &t in self.prompt.iter().chain(self.counts.keys()) {
                if let Some(x) = v.get_mut(t as usize) {
                    *x = if *x < 0.0 {
                        *x * o.repetition_penalty
                    } else {
                        *x / o.repetition_penalty
                    };
                }
            }
        }
        if o.presence_penalty != 0.0 || o.frequency_penalty != 0.0 {
            for (&t, &c) in &self.counts {
                if let Some(x) = v.get_mut(t as usize) {
                    *x -= o.presence_penalty + o.frequency_penalty * c as f32;
                }
            }
        }
        let n = o.no_repeat_ngram_size;
        if n > 1
            && out.len() >= n - 1
            && let Some(banned) = self.ngrams.get(&out[out.len() - (n - 1)..])
        {
            {
                for &t in banned {
                    v[t as usize] = f32::NEG_INFINITY;
                }
            }
        }
        let (best, _) = v
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |acc, (i, &x)| {
                if x > acc.1 { (i, x) } else { acc }
            });
        best as u32
    }
}
