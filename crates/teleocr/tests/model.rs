//! Tests on a real model (CPU). They need a GGUF written by `teleocr convert`
//! (the q8_0 one: F16 vision is fast on CPU):
//!
//!     TELEOCR_MODEL=models/teleocr-q8_0.gguf cargo test --release -p teleocr --test model
//!
//! Without `TELEOCR_MODEL` each test prints a note and passes (CI has no
//! weights). Images: model-card samples (XingChen-AGI/TeleOCR, Apache-2.0);
//! expected outputs: transformers 4.57 greedy (`scripts/ref.py`).

use candle_core::Device;
use image::RgbImage;
use std::path::PathBuf;
use std::sync::OnceLock;
use teleocr::{Engine, GenerateOptions, LoadOptions, ParseOptions, task_prompt};

fn model_path() -> Option<PathBuf> {
    std::env::var_os("TELEOCR_MODEL").map(PathBuf::from)
}

fn model() -> Option<&'static Engine> {
    static MODEL: OnceLock<Option<Engine>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            let path = model_path()?;
            Some(
                Engine::load(&path, &Device::Cpu, &LoadOptions::default())
                    .expect("loading TELEOCR_MODEL"),
            )
        })
        .as_ref()
}

macro_rules! need_model {
    () => {
        match model() {
            Some(m) => m,
            None => {
                eprintln!("TELEOCR_MODEL not set: skipped");
                return;
            }
        }
    };
}

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(name)
}

fn sample(name: &str) -> RgbImage {
    image::open(data(&format!("{name}.png")))
        .expect("test image")
        .to_rgb8()
}

fn expected(name: &str) -> String {
    std::fs::read_to_string(data(&format!("{name}.expected.txt")))
        .expect("expected output")
        .trim_end()
        .to_string()
}

fn run(engine: &Engine, task: &str) -> String {
    let out = engine
        .generate(
            &sample(task),
            task_prompt(task).unwrap(),
            &GenerateOptions::default(),
            None,
        )
        .unwrap();
    assert!(!out.truncated);
    out.text
}

// covers: REQ-INF-001, REQ-GGF-001
#[test]
fn text_sample_matches_reference() {
    let engine = need_model!();
    assert_eq!(run(engine, "text"), expected("text"));
}

// covers: REQ-INF-001, REQ-TAB-001
#[test]
fn table_sample_matches_reference() {
    let engine = need_model!();
    let otsl = run(engine, "table");
    assert_eq!(otsl, expected("table"));
    let html = teleocr::otsl::to_html(&otsl);
    assert!(html.starts_with("<table>"), "{html}");
    assert!(html.contains("colspan"), "{html}");
}

// covers: REQ-BAT-001
#[test]
fn batched_decoding_matches_single() {
    let engine = need_model!();
    let (text, table) = (sample("text"), sample("table"));
    let opts = GenerateOptions::default();
    let jobs = [
        (&text, task_prompt("text").unwrap(), &opts),
        (&table, task_prompt("table").unwrap(), &opts),
    ];
    let outs = engine.generate_batch(&jobs, 2).unwrap();
    assert_eq!(outs[0].text, expected("text"));
    assert_eq!(outs[1].text, expected("table"));
}

// covers: REQ-MEM-002
#[test]
fn tiny_kv_budget_still_decodes_every_job() {
    let Some(path) = model_path() else {
        eprintln!("TELEOCR_MODEL not set: skipped");
        return;
    };
    let mut engine = Engine::load(&path, &Device::Cpu, &LoadOptions::default()).unwrap();
    // One position: every batch is cut down to a single job.
    engine.set_kv_budget(1);
    let text = sample("text");
    let opts = GenerateOptions {
        max_new_tokens: 12,
        ..Default::default()
    };
    let jobs = [
        (&text, task_prompt("text").unwrap(), &opts),
        (&text, task_prompt("text").unwrap(), &opts),
    ];
    let outs = engine.generate_batch(&jobs, 8).unwrap();
    assert_eq!(outs.len(), 2);
    assert!(
        expected("text").starts_with(&outs[0].text),
        "{}",
        outs[0].text
    );
    assert_eq!(outs[0].text, outs[1].text);
}

// covers: REQ-PIP-001, REQ-PIP-002
#[test]
fn parse_page_gives_markdown() {
    let engine = need_model!();
    let page = engine
        .parse_page(&sample("text"), &ParseOptions::default())
        .unwrap();
    assert!(page.stats.blocks >= 1);
    assert!(
        page.markdown.contains("Hybrid methods may be required"),
        "{}",
        page.markdown
    );
    for b in &page.blocks {
        assert!(
            b.bbox.iter().all(|v| (0.0..=1.0).contains(v)),
            "{:?}",
            b.bbox
        );
    }
}
