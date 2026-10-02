//! Single-file GGUF models and loading from a Hugging Face directory.
//!
//! GGUF layout:
//! - metadata `general.architecture = "teleocr"`, `teleocr.config`
//!   (config.json), `teleocr.tokenizer` (tokenizer.json),
//!   `teleocr.preprocessor` (preprocessor_config.json);
//! - every safetensors tensor under its original name. 2-D linear weights of
//!   the text decoder use `text_dtype`, those of the vision tower
//!   `vision_dtype`; norms and biases stay F32. With tied embeddings
//!   `lm_head.weight` is dropped and the (quantized) embedding table serves
//!   as the output projection too.

use crate::weights::Weights;
use anyhow::{Context, Result, bail};
use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub const ARCH: &str = "teleocr";

/// Files a model needs besides its weights.
pub struct Assets {
    pub config: String,
    pub tokenizer: String,
    pub preprocessor: Option<String>,
}

/// Parses a dtype name: f32, f16, bf16, q8_0, q6k, q5k, q4k, q4_0.
pub fn parse_dtype(name: &str) -> Result<GgmlDType> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "f32" => GgmlDType::F32,
        "f16" => GgmlDType::F16,
        "bf16" => GgmlDType::BF16,
        "q8_0" | "q8" => GgmlDType::Q8_0,
        "q6k" | "q6_k" => GgmlDType::Q6K,
        "q5k" | "q5_k" => GgmlDType::Q5K,
        "q4k" | "q4_k" => GgmlDType::Q4K,
        "q4_0" => GgmlDType::Q4_0,
        other => bail!("unknown dtype '{other}' (f32, f16, bf16, q8_0, q6k, q5k, q4k, q4_0)"),
    })
}

/// `wanted` if the row length fits its block, else the closest that does.
fn fit_dtype(wanted: GgmlDType, row: usize) -> GgmlDType {
    for dt in [wanted, GgmlDType::Q8_0, GgmlDType::F16] {
        if row.is_multiple_of(dt.block_size()) {
            return dt;
        }
    }
    GgmlDType::F32
}

fn read_assets(dir: &Path) -> Result<Assets> {
    let read = |f: &str| {
        std::fs::read_to_string(dir.join(f)).with_context(|| format!("reading {:?}", dir.join(f)))
    };
    Ok(Assets {
        config: read("config.json")?,
        tokenizer: read("tokenizer.json")?,
        preprocessor: read("preprocessor_config.json").ok(),
    })
}

fn load_safetensors(dir: &Path) -> Result<HashMap<String, Tensor>> {
    let mut files: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("no .safetensors in {dir:?}");
    }
    let mut out = HashMap::new();
    for f in files {
        out.extend(candle_core::safetensors::load(&f, &Device::Cpu)?);
    }
    Ok(out)
}

fn tied(config: &str) -> bool {
    crate::config::Config::from_json(config)
        .map(|c| c.tie_word_embeddings)
        .unwrap_or(false)
}

pub struct ConvertOptions {
    pub text_dtype: GgmlDType,
    pub vision_dtype: GgmlDType,
}

/// Converts a Hugging Face directory to one GGUF file; returns per-dtype
/// tensor counts.
pub fn convert(dir: &Path, out: &Path, opts: &ConvertOptions) -> Result<Vec<(String, usize)>> {
    let assets = read_assets(dir)?;
    let tensors = load_safetensors(dir)?;
    let tie = tied(&assets.config);
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    let mut qtensors: Vec<(String, QTensor)> = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for name in names {
        if tie && name == "lm_head.weight" {
            continue;
        }
        let mut t = tensors[name].to_dtype(DType::F32)?;
        if t.rank() > 4 {
            // GGUF stores at most 4 dims; the patch-embedding conv kernel
            // [out, C, T, P, P] is used flattened to [out, C*T*P*P] anyway.
            t = t.flatten_from(1)?;
        }
        let dt = if t.rank() == 2 && name.starts_with("visual.patch_embed") {
            GgmlDType::F16
        } else if t.rank() == 2 && name.ends_with(".weight") {
            let wanted = if name.starts_with("visual.") {
                opts.vision_dtype
            } else {
                opts.text_dtype
            };
            fit_dtype(wanted, t.dim(1)?)
        } else {
            GgmlDType::F32
        };
        *counts.entry(format!("{dt:?}")).or_default() += 1;
        qtensors.push((name.clone(), QTensor::quantize(&t, dt)?));
    }

    let mut metadata = vec![
        (
            "general.architecture",
            gguf_file::Value::String(ARCH.into()),
        ),
        ("teleocr.config", gguf_file::Value::String(assets.config)),
        (
            "teleocr.tokenizer",
            gguf_file::Value::String(assets.tokenizer),
        ),
    ];
    if let Some(p) = assets.preprocessor {
        metadata.push(("teleocr.preprocessor", gguf_file::Value::String(p)));
    }
    let metadata: Vec<(&str, &gguf_file::Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    let refs: Vec<(&str, &QTensor)> = qtensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let mut file = std::io::BufWriter::new(std::fs::File::create(out)?);
    gguf_file::write(&mut file, &metadata, &refs)?;

    let mut counts: Vec<_> = counts.into_iter().collect();
    counts.sort();
    Ok(counts)
}

/// Loads a GGUF file or a Hugging Face directory.
pub fn load(path: &Path, device: &Device) -> Result<(Weights, Assets)> {
    if path.is_dir() {
        let assets = read_assets(path)?;
        let tie = tied(&assets.config);
        let mut map = HashMap::new();
        for (name, t) in load_safetensors(path)? {
            if tie && name == "lm_head.weight" {
                continue;
            }
            let t = t.to_dtype(DType::F32)?;
            map.insert(name, Arc::new(QTensor::quantize(&t, GgmlDType::F32)?));
        }
        // Dense tensors are moved to the device when the layers are built.
        return Ok((Weights::new(map, device.clone()), assets));
    }

    let mut file = std::fs::File::open(path).with_context(|| format!("opening {path:?}"))?;
    let content =
        gguf_file::Content::read(&mut file).with_context(|| format!("reading {path:?}"))?;
    let meta = |k: &str| -> Result<String> {
        match content.metadata.get(k) {
            Some(gguf_file::Value::String(s)) => Ok(s.clone()),
            _ => bail!("{path:?}: missing metadata '{k}'"),
        }
    };
    if meta("general.architecture")? != ARCH {
        bail!("{path:?} is not a {ARCH} GGUF");
    }
    let assets = Assets {
        config: meta("teleocr.config")?,
        tokenizer: meta("teleocr.tokenizer")?,
        preprocessor: meta("teleocr.preprocessor").ok(),
    };
    let mut map = HashMap::new();
    let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    for name in names {
        // Block-quantized tensors go straight to the device (they are used
        // as is); dense ones stay on the CPU until a layer converts them to
        // its compute dtype, so the device never holds two copies.
        let quantized = !matches!(
            content.tensor_infos[&name].ggml_dtype,
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
        );
        let dev = if quantized { device } else { &Device::Cpu };
        let t = content.tensor(&mut file, &name, dev)?;
        map.insert(name, Arc::new(t));
    }
    Ok((Weights::new(map, device.clone()), assets))
}
