use anyhow::{Context, Result};
use candle_core::Device;
use clap::{Parser, Subcommand};
use std::io::Write;
use std::path::PathBuf;
use teleocr::{Engine, GenerateOptions, LayoutMode, LoadOptions, ParseOptions, task_prompt};

#[cfg(feature = "pdf")]
mod pdf;
mod server;

#[derive(Parser)]
#[command(
    name = "teleocr",
    version,
    about = "TeleOCR document parsing (Rust/Candle, GGUF)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct ModelArgs {
    /// Model: a .gguf file or a Hugging Face directory.
    #[arg(short, long, env = "TELEOCR_MODEL")]
    model: PathBuf,
    /// Run on the CPU even when a GPU is available.
    #[arg(long, env = "TELEOCR_CPU")]
    cpu: bool,
    /// CPU threads (default: all cores).
    #[arg(long, env = "TELEOCR_THREADS")]
    threads: Option<usize>,
    /// Upper bound on the pixels an image is resized to before the vision
    /// tower (default 12845056 = the model's); lower = faster, fewer tokens.
    #[arg(long, env = "TELEOCR_MAX_PIXELS")]
    max_pixels: Option<usize>,
    /// Document parsing: pages above this many pixels are downscaled before
    /// their blocks are cropped (default 4500000, ~A4 at 215 DPI).
    #[arg(long, env = "TELEOCR_MAX_PAGE_PIXELS")]
    max_page_pixels: Option<u64>,
    /// KV cache positions one decoding batch may use (rows x prompt
    /// length; default 16384 = 3.7 GB). Bounds GPU memory on large crops.
    #[arg(long, env = "TELEOCR_KV_BUDGET")]
    kv_budget: Option<usize>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Convert a Hugging Face checkpoint directory to GGUF.
    Convert {
        dir: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
        /// Dtype of the text decoder's linear weights.
        #[arg(long, default_value = "q8_0")]
        text_dtype: String,
        /// Dtype of the vision tower's linear weights.
        #[arg(long, default_value = "f16")]
        vision_dtype: String,
    },
    /// Parse a document page: layout, then every block, to Markdown.
    Parse {
        #[command(flatten)]
        model: ModelArgs,
        image: PathBuf,
        /// Layout mode: detection (clean pages) or segmentation (photos).
        #[arg(long, default_value = "detection")]
        mode: String,
        /// Keep headers, footers and page numbers in the Markdown.
        #[arg(long)]
        paratext: bool,
        /// Print the blocks as JSON instead of Markdown.
        #[arg(long)]
        json: bool,
        /// PDF pages, 1-based ("1-3,5"); default all.
        #[arg(long)]
        pages: Option<String>,
        /// PDF render resolution.
        #[arg(long, default_value_t = 200.0)]
        dpi: f32,
        /// Sequences decoded together (1 = one at a time); memory grows
        /// with it (~0.6 GB per extra sequence on a full page).
        #[arg(long, default_value_t = 8)]
        batch: usize,
    },
    /// Serve the HTTP API (/v1/ocr, /v1/chat/completions).
    Serve {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long, env = "TELEOCR_HOST", default_value = "127.0.0.1")]
        host: String,
        #[arg(long, env = "TELEOCR_PORT", default_value_t = 8090)]
        port: u16,
        /// Model id reported by /v1/models.
        #[arg(long, env = "TELEOCR_MODEL_ID", default_value = "teleocr")]
        model_id: String,
        /// Sequences decoded together when parsing (1 = one at a time).
        #[arg(long, env = "TELEOCR_BATCH", default_value_t = 8)]
        batch: usize,
    },
    /// Run one task on one image.
    Run {
        #[command(flatten)]
        model: ModelArgs,
        image: PathBuf,
        /// text, table, formula, code, layout, layout_seg, figure — or a free prompt.
        #[arg(short, long, default_value = "text")]
        task: String,
        /// Resize the image to NxN first (the layout tasks expect 1036).
        #[arg(long)]
        resize: Option<u32>,
        #[arg(long, default_value_t = 4096)]
        max_tokens: usize,
        /// Print timings as JSON on stderr.
        #[arg(long)]
        timings: bool,
    },
}

fn device(cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    Ok(Device::cuda_if_available(0)?)
}

/// Pages of an image or (with the `pdf` feature) a PDF.
pub fn load_document(bytes: &[u8], pages: Option<&str>, dpi: f32) -> Result<Vec<image::RgbImage>> {
    #[cfg(feature = "pdf")]
    if pdf::is_pdf(bytes) {
        return pdf::render(bytes, dpi, pages);
    }
    #[cfg(not(feature = "pdf"))]
    if bytes.starts_with(b"%PDF") {
        anyhow::bail!("built without PDF support (feature `pdf`)");
    }
    let _ = (pages, dpi);
    Ok(vec![image::load_from_memory(bytes)?.to_rgb8()])
}

fn load(m: &ModelArgs) -> Result<Engine> {
    if let Some(t) = m.threads {
        // Read by candle's CPU kernels on first use.
        unsafe { std::env::set_var("RAYON_NUM_THREADS", t.to_string()) };
    }
    let dev = device(m.cpu)?;
    let t0 = std::time::Instant::now();
    let mut e = Engine::load(&m.model, &dev, &LoadOptions::default())?;
    if let Some(p) = m.max_page_pixels {
        e.set_max_page_pixels(p);
    }
    if let Some(t) = m.kv_budget {
        e.set_kv_budget(t);
    }
    if let Some(p) = m.max_pixels {
        e.set_max_pixels(p);
    }
    tracing::info!(
        "model loaded in {:.1}s on {:?}",
        t0.elapsed().as_secs_f64(),
        dev
    );
    Ok(e)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    match Cli::parse().cmd {
        Cmd::Convert {
            dir,
            out,
            text_dtype,
            vision_dtype,
        } => {
            let opts = teleocr::gguf::ConvertOptions {
                text_dtype: teleocr::gguf::parse_dtype(&text_dtype)?,
                vision_dtype: teleocr::gguf::parse_dtype(&vision_dtype)?,
            };
            let counts = teleocr::gguf::convert(&dir, &out, &opts)?;
            let size = std::fs::metadata(&out)?.len() as f64 / 1e6;
            eprintln!("wrote {} ({size:.0} MB): {counts:?}", out.display());
        }
        Cmd::Parse {
            model,
            image,
            mode,
            paratext,
            json,
            pages,
            dpi,
            batch,
        } => {
            let engine = load(&model)?;
            let bytes = std::fs::read(&image).with_context(|| format!("{image:?}"))?;
            let images = load_document(&bytes, pages.as_deref(), dpi)?;
            let mode = match mode.as_str() {
                "detection" => LayoutMode::Detection,
                "segmentation" => LayoutMode::Segmentation,
                m => anyhow::bail!("unknown mode '{m}' (detection, segmentation)"),
            };
            let opts = ParseOptions {
                mode,
                paratext,
                batch,
                ..Default::default()
            };
            let t0 = std::time::Instant::now();
            let parsed = engine.parse_pages(&images, &opts)?;
            for (i, page) in parsed.iter().enumerate() {
                eprintln!("page {}: {}", i + 1, serde_json::to_string(&page.stats)?);
                if !json {
                    if i > 0 {
                        println!();
                    }
                    println!("{}", page.markdown);
                }
            }
            eprintln!(
                "{} page(s) in {:.2}s",
                parsed.len(),
                t0.elapsed().as_secs_f64()
            );
            if json {
                println!("{}", serde_json::to_string_pretty(&parsed)?);
            }
        }
        Cmd::Serve {
            model,
            host,
            port,
            model_id,
            batch,
        } => {
            let engine = load(&model)?;
            tokio::runtime::Runtime::new()?
                .block_on(server::serve(engine, model_id, batch, &host, port))?;
        }
        Cmd::Run {
            model,
            image,
            task,
            resize,
            max_tokens,
            timings,
        } => {
            let engine = load(&model)?;
            let mut img = image::open(&image)
                .with_context(|| format!("{image:?}"))?
                .to_rgb8();
            if let Some(n) = resize {
                img = teleocr::image::resize_bicubic(&img, n as usize, n as usize);
            }
            let prompt = task_prompt(&task).unwrap_or(&task);
            let opts = GenerateOptions {
                max_new_tokens: max_tokens,
                ..Default::default()
            };
            let mut stdout = std::io::stdout();
            let mut cb = |s: &str| {
                let _ = stdout.write_all(s.as_bytes());
                let _ = stdout.flush();
            };
            let out = engine.generate(&img, prompt, &opts, Some(&mut cb))?;
            println!();
            if timings {
                let mut t = serde_json::to_value(&out.timings)?;
                let tps = out.timings.generated_tokens as f64 / (out.timings.decode_ms / 1e3);
                t["decode_tok_s"] = serde_json::json!(tps);
                t["truncated"] = serde_json::json!(out.truncated);
                eprintln!("{t}");
            }
        }
    }
    Ok(())
}
