# AGENTS.md

Guidance for AI agents working in this repository.

## Project

Rust/Candle port of TeleOCR (XingChen-AGI/TeleOCR, ~1.4B params, Apache-2.0),
a document-parsing VLM. One binary (`teleocr`), models from a Hugging Face
directory or one GGUF file written by `teleocr convert`.

Architecture: Qwen2.5-VL vision tower (`vision.rs`: 14 px patches, 2-D RoPE,
112 px window attention, full attention in blocks 7/15/23/31, 2x2 merger)
→ Qwen3-style decoder (`text.rs`: per-head QK RMSNorm, head_dim 128 ≠
hidden/heads, GQA 16/8, multimodal RoPE sections [16, 24, 24], tied
embeddings). Prompt, positions and decoding in `model.rs`; the official
document pipeline (layout → crops → per-type prompts → post-processing →
Markdown) in `pipeline.rs`; OTSL tables in `otsl.rs`.

## Where things are

- `crates/teleocr/src/image.rs`: `smart_resize` (Python half-to-even
  rounding), PIL-style antialiased bicubic, patch flattening.
- `crates/teleocr/src/gguf.rs`: GGUF layout and conversion, HF loading.
- `crates/teleocr/src/weights.rs`: dense / `QMatMul` linear, RMSNorm.
- `crates/teleocr-cli/src/`: CLI (`main.rs`), HTTP server (`server.rs`),
  PDF rendering with PDFium (`pdf.rs`, feature `pdf`).
- `scripts/ref.py`: Python reference dump; `crates/teleocr/examples/parity.rs`
  compares against it.

## Commands

```bash
cargo build --release -p teleocr-cli
cargo test --release --workspace
scripts/fetch-pdfium.sh target/release          # PDF input
teleocr convert <hf-dir> -o models/teleocr-q8_0.gguf
teleocr parse -m models/teleocr-q8_0.gguf page.png
cargo run --release --example parity -- <model> <image> <ref.safetensors> [1036] [--cpu]
```

CUDA: `--features cuda`, build into `--target-dir target-cuda`, needs
`/usr/local/cuda/bin` in PATH and `CUDA_COMPUTE_CAP`.

## Rules

- The Python reference (transformers 4.57, `modeling_naviocr.py`) is the
  spec for the model; the official client (`TeleOCR_client.py`,
  github.com/caipeng328/TeleOCR) is the spec for the pipeline (prompts with a
  leading newline, per-type penalties, crops, post-processing).
- Check model changes with the parity example: greedy tokens must match the
  reference exactly on the five model-card samples.
- candle 0.11 pitfalls: `matmul` with a stride-0 batch (`broadcast_left`)
  returns wrong values — flatten to 2-D instead (see `weights.rs`).
