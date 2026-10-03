# Changelog

All notable changes to **teleocr-rs** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-10-03

### Fixed
- **GPU memory on large inputs** (12 Mpx phone photos ran a 24 GB GPU out
  of memory): the text prefill runs in 1024-position chunks instead of
  building the full score matrix (14.4 → 5.3 GB peak on a 4.5 Mpx crop,
  greedy output still token-exact); pages above 4.5 Mpx are downscaled
  before their blocks are cropped (`--max-page-pixels`, was 64 Mpx); batches
  are sorted by size and cut at a KV budget (`--kv-budget`, 16k positions)
  with a one-sequence-at-a-time retry on CUDA out-of-memory; the server
  returns the CUDA pool's cached memory to the driver after each request;
  full-attention chunks of the vision tower are sized by key count.

### Added
- `TELEOCR_*` environment variables for every model and server option
  (`teleocr.example.env`).
- Model tests (`crates/teleocr/tests/model.rs`, run with `TELEOCR_MODEL`),
  `tests/e2e.sh` (CLI, PDF, HTTP API), test images from the model card.
- Project files: specification and traceability matrix (`spec/`),
  `setup.sh`, `build.sh`, `prereq.sh`, CI and release workflows,
  `CREDITS.md`, `CONTRIBUTING.md`, portfolio, `docs/architecture.md`.
- `scripts/fetch-pdfium.sh` also fetches the Windows x64 build.

## [0.1.1] - 2026-10-02

### Fixed
- The server bound PDFium on every request, so every PDF after the first
  failed with `PdfiumLibraryBindingsAlreadyInitialized`; one instance is now
  shared by the process.
- PDF page ranges past the last page are clipped (`1-8` on a 1-page PDF is
  page 1) instead of failing; a range that selects no page is still an error.

## [0.1.0] - 2026-10-02

First release: TeleOCR in Rust/Candle with GGUF weights.

### Added
- **Model**: Qwen2.5-VL vision tower (window attention, 2-D RoPE, 2×2
  merger) and the Qwen3-style decoder (per-head QK RMSNorm, head_dim 128,
  multimodal RoPE). Greedy output token-identical to transformers 4.57 on
  the five model-card samples, on CPU and CUDA.
- **GGUF**: `teleocr convert` writes one file; text and vision linears in
  q8_0, f16 or f32 (`teleocr-q8v.gguf` 1.5 GB for GPUs, `teleocr-q8_0.gguf`
  2.0 GB with F16 vision for CPUs), published on Hugging Face
  (`rleo/TeleOCR-GGUF`).
- **Document pipeline** of the official client: layout (detection or
  segmentation), crops, per-type prompts and decoding settings,
  post-processing, OTSL tables to HTML, Markdown; batched decoding of
  blocks and pages (19.2 → 11.5 s for 3 pages).
- **PDF input** through PDFium (`scripts/fetch-pdfium.sh`), page ranges.
- **CLI**: `convert`, `run`, `parse` (Markdown or `--json`), `serve`.
- **HTTP server**: `POST /v1/ocr` (JSON or multipart, images and PDFs),
  `POST /v1/chat/completions`, `GET /health`, `GET /v1/models`.

[Unreleased]: https://github.com/rzafiamy/teleocr-rs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rzafiamy/teleocr-rs/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/rzafiamy/teleocr-rs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/rzafiamy/teleocr-rs/releases/tag/v0.1.0
