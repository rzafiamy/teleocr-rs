# Credits

teleocr-rs is distributed under Apache-2.0. It builds on the work below,
each used under its own license. Versions are those of `Cargo.lock`.

## Authors

- XingChen-AGI (China Telecom) — TeleOCR model, training, Python reference
  (`modeling_naviocr.py`) and weights, Apache-2.0
  ([XingChen-AGI/TeleOCR](https://huggingface.co/XingChen-AGI/TeleOCR));
  official document client `TeleOCR_client.py`
  ([caipeng328/TeleOCR](https://github.com/caipeng328/TeleOCR)), the spec
  for the parsing pipeline.
- Qwen team (Alibaba) — Qwen2.5-VL vision tower and Qwen3 decoder
  architectures TeleOCR is built on.
- Rija Z. ([@rzafiamy](https://github.com/rzafiamy)) — this port: Rust/Candle
  implementation, GGUF, batching, memory work, HTTP server.
- Developed with the assistance of Claude (Anthropic) through Claude Code.

## Third-party sources

- Weights, `config.json`, tokenizer and preprocessor configuration come
  from `XingChen-AGI/TeleOCR` (Apache-2.0); the converted GGUF files keep
  that license.
- `tests/data/text.png` and `tests/data/table.png` are model-card samples
  from the same repository (Apache-2.0); `tests/data/text.pdf` is made from
  `text.png`; `tests/data/*.expected.txt` are transformers outputs.
- PDFium (BSD-3-Clause / Apache-2.0, Google) is downloaded at build time as
  a prebuilt shared library from
  [bblanchon/pdfium-binaries](https://github.com/bblanchon/pdfium-binaries).
- No Tauri library is used: teleocr-rs is a CLI and HTTP server without a
  graphical interface.

## Rust dependencies

| Dependency | Version | Role | License | Source |
|---|---|---|---|---|
| `candle-core` | 0.11.0 | Tensors, quantized matmul (CPU/CUDA/Metal), GGUF read/write | MIT OR Apache-2.0 | https://github.com/huggingface/candle |
| `candle-nn` | 0.11.0 | Layers, activations, RoPE | MIT OR Apache-2.0 | https://github.com/huggingface/candle |
| `tokenizers` | 0.21.4 | Qwen BPE tokenizer | Apache-2.0 | https://github.com/huggingface/tokenizers |
| `image` | 0.25.10 | Image decoding (PNG, JPEG, WebP, TIFF, BMP) | MIT OR Apache-2.0 | https://github.com/image-rs/image |
| `pdfium-render` | 0.9.4 | PDF rendering bindings (optional, `pdf`) | MIT OR Apache-2.0 | https://github.com/ajrcarey/pdfium-render |
| `cudarc` | 0.19.10 | Trimming the CUDA memory pool (optional, `cuda`) | MIT OR Apache-2.0 | https://github.com/coreylowman/cudarc |
| `memmap2` | 0.9.11 | Memory-mapped weights | MIT OR Apache-2.0 | https://github.com/RazrFalcon/memmap2-rs |
| `regex` | 1.13.1 | Layout parsing, post-processing | MIT OR Apache-2.0 | https://github.com/rust-lang/regex |
| `serde` / `serde_json` | 1.0.229 / 1.0.151 | Config, JSON API | MIT OR Apache-2.0 | https://serde.rs |
| `anyhow` | 1.0.104 | Errors | MIT OR Apache-2.0 | https://github.com/dtolnay/anyhow |
| `tracing` / `tracing-subscriber` | 0.1.44 / 0.3.23 | Logs | MIT | https://github.com/tokio-rs/tracing |
| `clap` | 4.6.7 | Command line, environment variables | MIT OR Apache-2.0 | https://github.com/clap-rs/clap |
| `axum` | 0.7.9 | HTTP server | MIT | https://github.com/tokio-rs/axum |
| `tokio` | 1.53.1 | Async runtime | MIT | https://tokio.rs |
| `multer` / `futures-util` | 3.1.0 / 0.3.34 | Multipart uploads | MIT / MIT OR Apache-2.0 | https://github.com/rwf2/multer |
| `base64` | 0.22.1 | Image payloads | MIT OR Apache-2.0 | https://github.com/marshallpierce/rust-base64 |

## Tools

- [transformers](https://github.com/huggingface/transformers) 4.57 (PyTorch)
  for the parity reference (`scripts/ref.py`).
