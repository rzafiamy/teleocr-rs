//! Compares the port against a Python reference dump (scripts/ref.py).
//!
//! cargo run --release --example parity -- <model> <image> <ref.safetensors> [resize] [--cpu]

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use teleocr::{Engine, GenerateOptions, LoadOptions};

fn stats(name: &str, a: &Tensor, b: &Tensor) -> Result<()> {
    let a = a
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?;
    let b = b
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?;
    let d = (&a - &b)?.abs()?;
    let max = d.max(0)?.to_scalar::<f32>()?;
    let mean = d.mean_all()?.to_scalar::<f32>()?;
    let (av, bv) = (a.to_vec1::<f32>()?, b.to_vec1::<f32>()?);
    let dot: f64 = av
        .iter()
        .zip(&bv)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let na: f64 = av.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = bv.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    println!(
        "{name:<14} max|d|={max:.5} mean|d|={mean:.6} cos={:.6}",
        dot / (na * nb)
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    let args: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let dev = if cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let engine = Engine::load(args[1].as_ref(), &dev, &LoadOptions::default())?;
    let r = candle_core::safetensors::load(args[3], &Device::Cpu)?;
    let mut img = image::open(args[2])?.to_rgb8();
    if let Some(n) = args.get(4) {
        let n: usize = n.parse()?;
        img = teleocr::image::resize_bicubic(&img, n, n);
    }

    let (patches, emb) = engine.encode_image(&img)?;
    let pv = Tensor::from_slice(
        &patches.data,
        (patches.grid_h * patches.grid_w, patches.dim),
        &Device::Cpu,
    )?;
    let rpv = &r["pixel_values"];
    println!(
        "grid {}x{} ref {:?}",
        patches.grid_h,
        patches.grid_w,
        r["image_grid_thw"].to_vec2::<i64>()?
    );
    stats("pixel_values", &pv, rpv)?;
    stats("vision_out", &emb, &r["vision_out"])?;

    let n_img = emb.dim(0)?;
    let rids: Vec<i64> = r["input_ids"].to_vec1()?;
    let prompt_text = {
        let mut st = safetensors_meta(args[3])?;
        st.remove("prompt").unwrap_or_default()
    };
    let (ids, _) = engine.prompt_ids(&prompt_text, n_img)?;
    let same = ids.len() == rids.len() && ids.iter().zip(&rids).all(|(a, b)| *a as i64 == *b);
    println!("input_ids      {} tokens, match={same}", ids.len());

    let out = engine.generate(
        &img,
        &prompt_text,
        &GenerateOptions {
            max_new_tokens: 4096,
            ..Default::default()
        },
        None,
    )?;
    let rgen: Vec<i64> = r["generated"].to_vec1()?;
    let rgen: Vec<u32> = rgen
        .into_iter()
        .map(|x| x as u32)
        .filter(|t| *t != 151645 && *t != 151643)
        .collect();
    let common = out
        .tokens
        .iter()
        .zip(&rgen)
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "generated      rust {} ref {} common-prefix {} exact={}",
        out.tokens.len(),
        rgen.len(),
        common,
        out.tokens == rgen
    );
    println!("timings {:?}", out.timings);
    Ok(())
}

fn safetensors_meta(path: &str) -> Result<std::collections::HashMap<String, String>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut n = [0u8; 8];
    f.read_exact(&mut n)?;
    let mut h = vec![0u8; u64::from_le_bytes(n) as usize];
    f.read_exact(&mut h)?;
    let v: serde_json::Value = serde_json::from_slice(&h)?;
    Ok(v.get("__metadata__")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_default())
}
