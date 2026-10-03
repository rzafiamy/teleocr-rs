# Architecture

![pipeline](../portfolio/architecture-diagram.svg)

## Model

| Step | Module | Device |
|---|---|---|
| `smart_resize` (multiple of 28 px, Python rounding), antialiased bicubic, normalization, 14 px patches (temporal 2) | `image.rs` | CPU |
| Patch embedding, 32 blocks, 2-D RoPE, window attention (112 px) with full attention in blocks 7/15/23/31, 2×2 merger → hidden size of the decoder | `vision.rs` | device |
| Chat template with `<|image_pad|>` tokens replaced by the vision embeddings; multimodal RoPE positions (sections 16/24/24) | `model.rs` | CPU → device |
| Qwen3-style decoder: 28 layers, GQA 16/8, head_dim 128, per-head QK RMSNorm, tied embeddings; prefill in 1024-position chunks, KV cache | `text.rs`, `weights.rs` (`QMatMul`) | device |
| Greedy decoding with repetition / presence / frequency penalties and no-repeat n-grams | `model.rs` (`GenerateOptions`) | CPU |
| Batched decoding: per-job prefill, left-padded stacked caches, batches cut at the KV budget | `model.rs` (`generate_batch`) | device |

## Document pipeline (official client)

| Step | Module |
|---|---|
| PDF pages rendered with PDFium at a DPI, page ranges | `crates/teleocr-cli/src/pdf.rs` |
| Layout prompt on the page resized to 1036² (detection or segmentation) | `pipeline.rs` (`LayoutMode`) |
| `<box:…><label:…><…>` lines → typed blocks with page-fraction boxes | `pipeline.rs` (`parse_layout`) |
| Pages downscaled to ≤ 4.5 Mpx, blocks cropped (rotated by their angle) | `pipeline.rs` (`crop_block`) |
| Per-type prompt and decoding settings, all blocks of all pages batched | `pipeline.rs` (`parse_pages`) |
| Post-processing: OTSL → HTML, `$$` equations, inline math, code fences | `otsl.rs`, `pipeline.rs` |
| Markdown in reading order (paratext dropped by default) | `pipeline.rs` (`to_markdown`) |

## Server

`crates/teleocr-cli/src/server.rs` (axum): one engine behind a mutex, work
on a blocking thread, the CUDA pool trimmed after each request
(`release_cached_memory`). Endpoints: [README](../README.md#http-api).

## GGUF

`gguf.rs` stores the HF `config.json`, `preprocessor_config.json` and
`tokenizer.json` as metadata and the weights under their HF names; linear
weights of the decoder and of the vision tower are quantized independently
(`--text-dtype`, `--vision-dtype`).
