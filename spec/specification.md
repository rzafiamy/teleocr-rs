# Specification

## Goal

Run TeleOCR (XingChen-AGI/TeleOCR, ~1.2B document-parsing VLM) as a single
native binary and a single GGUF model file, with exact parity to the Python
reference, bounded GPU memory on any input, and serve it over an HTTP API so
that [zallama](https://github.com/rzafiamy/zallama) can host it.

## Context and users

TeleOCR leads OmniDocBench v1.6 but ships as a transformers model with
custom code (`modeling_naviocr.py`): Python, PyTorch, 2.8 GB of weights.
llama.cpp cannot load it (Qwen3-style decoder, head_dim 128 ≠ hidden /
heads, per-head QK RMSNorm, behind a Qwen2.5-VL config).

Users:
- **zallama integrator**: declares the server as an OCR backend; needs
  `/health`, `/v1/ocr`, a known and bounded VRAM footprint, one model file.
- **Application developer**: sends a page image or a PDF, gets Markdown
  (headings, paragraphs, tables, formulas, code) or the blocks with their
  boxes.
- **Command-line user**: converts a scanned page or a PDF to Markdown.
- **Maintainer**: converts checkpoints, checks parity against the reference.

Main path: `hf download rleo/TeleOCR-GGUF teleocr-q8v.gguf` (or `teleocr
convert`) → `teleocr serve -m teleocr-q8v.gguf` (zallama) or `teleocr parse
-m ... page.pdf`.

## Priorities

- **Must (core, MVP)**: parity with transformers, single GGUF, document
  pipeline to Markdown, PDF input, CUDA, bounded memory, HTTP API.
- **Should**: CPU and Metal builds, OpenAI-compatible chat endpoint, JSON
  blocks with boxes, segmentation layout for photos.
- **Could / later**: streaming, F16 KV cache, the remaining LaTeX
  post-processing of the official client, image crops in the Markdown
  (see [TODO.md](../TODO.md)).

## Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| REQ-IMG-001 | Images are resized like the HF processor: `smart_resize` with Python half-to-even rounding, PIL-style antialiased bicubic, 14 px patches merged 2×2; `max_pixels` caps the image tokens. | Must |
| REQ-INF-001 | Greedy output is token-identical to transformers 4.57 on the five model-card samples (text, formula, table, code, layout), on CPU (F32 HF directory, Q8_0 GGUF) and CUDA. | Must |
| REQ-INF-002 | Two decoding presets: the model card's (repetition penalty 1.05) for `teleocr run`, the official pipeline's per-type settings (presence / frequency penalties, no repeated 100-grams, prompts with a leading newline) for parsing and the server. | Must |
| REQ-GGF-001 | `teleocr convert` writes one self-contained GGUF (config, preprocessor, tokenizer, weights); the engine loads a GGUF or a Hugging Face directory. | Must |
| REQ-GGF-002 | Linear weights of the text decoder and the vision tower are stored in q8_0, f16 or f32 independently: q8v (all Q8_0) ≤ 1.5 GB, q8_0 (F16 vision) ≤ 2.1 GB. | Must |
| REQ-PIP-001 | Document parsing starts with a layout pass on the page resized to 1036²: detection (boxes) or segmentation (polygons, for curved or skewed photos); its output is parsed into typed blocks with boxes as page fractions, in reading order. | Must |
| REQ-PIP-002 | Every text-bearing block is cropped from the page and recognized with its type's prompt; the page becomes Markdown, without headers, footers and page numbers unless `paratext` is set. | Must |
| REQ-PIP-003 | Block outputs are post-processed like the official client: equations wrapped in `$$`, inline math normalized, code fenced with its language. | Must |
| REQ-TAB-001 | Tables come out in OTSL and are converted to HTML with `colspan` / `rowspan`. | Must |
| REQ-BAT-001 | Several sequences (blocks, pages) are decoded together with left-padded stacked caches; the output equals one-at-a-time decoding. | Must |
| REQ-MEM-001 | GPU memory stays bounded on large inputs: prefill in 1024-position chunks, pages above 4.5 Mpx downscaled before cropping (`--max-page-pixels`), the CUDA pool returned to the driver after each server request. | Must |
| REQ-MEM-002 | Batches are cut at a KV budget (16k positions by default, `--kv-budget`), and a batch that runs out of GPU memory is retried one sequence at a time. | Must |
| REQ-PDF-001 | PDF pages are rendered with PDFium (loaded at run time) at a chosen DPI; page ranges such as `1-3,5` are clipped to the document. | Must |
| REQ-CLI-001 | `teleocr run` runs one task (text, table, formula, code, layout, figure, seal, or a free prompt) on one image and streams the output. | Must |
| REQ-CLI-002 | `teleocr parse` turns an image or a PDF into Markdown, or into JSON blocks with `--json`. | Must |
| REQ-CLI-003 | Model, device, limits and server options can be set through `TELEOCR_*` environment variables; command-line options take precedence. | Should |
| REQ-SRV-001 | `POST /v1/ocr` takes JSON (base64 or data URI) or multipart; `task=parse` (default) returns `{markdown, blocks, stats}`, a PDF `{markdown, pages}`, other tasks `{content, raw, timings}`. | Must |
| REQ-SRV-002 | `POST /v1/chat/completions` (OpenAI shape) takes one `image_url` data URI and a task name or prompt. | Should |
| REQ-SRV-003 | `GET /health` and `GET /v1/models`; bad input (no image, unreadable image, unknown task, PDF with another task) → 400 with a JSON error. | Must |
| REQ-GPU-001 | CUDA (RTX 4090, q8v): ~225 tokens/s per sequence, a full page in ~3-4 s at batch 8, ≤ 9 GB over baseline at batch 16. | Must |

## Non-functional requirements

- Platforms: Linux x86_64 and aarch64, Windows x86_64 (CPU), macOS (Metal);
  CUDA on Linux with the CUDA Toolkit ≥ 12.
- Toolchain pinned by `rust-toolchain.toml`; builds with `--locked`.
- No network access at run time; no secret needed. PDFium is a separate
  shared library (`scripts/fetch-pdfium.sh`).
- Code and weights under Apache-2.0.

## Known limitations

- Single-sequence decoding is bound by kernel launches (~4.4 ms per token);
  the layout pass is the critical path of a page.
- CPU works but is slow on whole pages (tens of seconds each); the Q8_0 vision
  tower is very slow on CPU, use the q8_0 GGUF (F16 vision) there.
- Part of the official client's LaTeX post-processing is not ported
  (`unicode_to_latex`, `\left`/`\right` matching, brace balancing); title
  levels and figure crops are not emitted.
- No streaming on `/v1/chat/completions`; one request at a time per server
  (the model is behind a mutex).
