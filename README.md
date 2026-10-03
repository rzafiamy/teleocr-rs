# teleocr-rs

Rust/[Candle](https://github.com/huggingface/candle) port of
[TeleOCR](https://huggingface.co/XingChen-AGI/TeleOCR) (China Telecom, Apache-2.0),
a ~1.2B document-parsing VLM that leads OmniDocBench v1.6. One binary, one
GGUF file, CPU or CUDA; page images and PDFs to Markdown.

llama.cpp cannot load this model: its decoder is Qwen3-style (head_dim 128 ≠
hidden / heads, per-head QK RMSNorm) behind a Qwen2.5-VL config. This port
implements it directly: Qwen2.5-VL vision tower (window attention, 2-D RoPE,
2×2 merger) and the decoder with multimodal RoPE.

## Results

- **Exact.** Greedy output is token-identical to transformers 4.57 on the five
  model-card samples (text, formula, table, code, 767-token layout), on CPU
  (F32 and Q8_0 GGUF) and on CUDA (Q8_0), with the vision tower in F16 or Q8_0.
- **Fast** (RTX 4090, Q8_0): ~225 tokens/s per sequence, vision tower 0.38 s
  for a 1036×1036 page; full page parsing ~3.2 s at batch 8 (19.2 s → 11.5 s
  for 3 PDF pages vs one block at a time, identical Markdown).
- **Small**: `teleocr-q8v.gguf` is 1.5 GB (Q8_0 text and vision) for GPUs; on
  CPU use the 2.0 GB q8_0 (F16 vision), Q8_0 vision is slow there.
- **Bounded memory**: a 12 Mpx phone photo parses on a shared 24 GB GPU
  (chunked prefill, page downscaling, KV budget); idle back to ~2 GB after
  each request.

Details: [docs/performance.md](docs/performance.md). Demo (a model-card page
and its Markdown): [portfolio/](portfolio/README.md).

## Features

Core (Must, see [spec/specification.md](spec/specification.md) for the IDs):

- **Document parsing** (REQ-PIP-001/002/003): layout pass (detection, or
  segmentation for curved / skewed photos), every block recognized with its
  type's prompt, post-processing, Markdown in reading order or JSON blocks
  with boxes (`crates/teleocr/src/pipeline.rs`).
- **Tables** (REQ-TAB-001): OTSL output converted to HTML with spans (`otsl.rs`).
- **PDF input** (REQ-PDF-001): PDFium rendering at a chosen DPI, page ranges
  (`crates/teleocr-cli/src/pdf.rs`).
- **Parity** (REQ-INF-001/002): token-exact with transformers; the model
  card's and the official pipeline's decoding settings (`vision.rs`,
  `text.rs`, `model.rs`, `image.rs`).
- **One GGUF** (REQ-GGF-001/002): `teleocr convert`, q8_0 / f16 / f32 per tower
  (`gguf.rs`, `weights.rs`).
- **Batching and memory** (REQ-BAT-001, REQ-MEM-001/002): blocks and pages
  decoded together, bounded by a KV budget, OOM retry one by one
  (`model.rs` `generate_batch`, `text.rs` chunked prefill).
- **HTTP API** (REQ-SRV-001/003): `/v1/ocr`, `/health`, `/v1/models`
  (`crates/teleocr-cli/src/server.rs`).

Secondary (Should): OpenAI-style `/v1/chat/completions` (REQ-SRV-002,
`server.rs`), `TELEOCR_*` environment variables (REQ-CLI-003,
`crates/teleocr-cli/src/main.rs`), CPU and Metal builds (`build.sh`).

Each feature maps to its code and tests in [spec/matrix.md](spec/matrix.md).

### Known limitations

- Single-sequence decoding is launch-bound (~4.4 ms per token); the layout
  pass is the critical path of a page.
- CPU parsing takes tens of seconds per page; on CPU use the q8_0 GGUF (the
  Q8_0 vision tower is very slow there).
- Part of the official client's LaTeX post-processing is not ported, nor
  title levels and figure crops ([TODO.md](TODO.md)).
- No streaming; one request at a time per server.

## Installation

### Prerequisites

- Rust (version pinned by `rust-toolchain.toml`, installed by `prereq.sh`
  through rustup), a C/C++ compiler, `pkg-config`, `git`, `curl`.
- NVIDIA GPU: CUDA Toolkit ≥ 12 (`nvcc`) and a matching driver. Apple GPU:
  Xcode Command Line Tools.
- PDF input: the PDFium shared library, downloaded by `build.sh`
  (`scripts/fetch-pdfium.sh`).
- `python3` for `tests/e2e.sh`.

`./prereq.sh` checks all of this and installs what is missing
(`CHECK_ONLY=1 ./prereq.sh` only checks). `./setup.sh` runs it, fetches the
crates and compiles once.

### Platforms

| Platform | Backend | Status |
|---|---|---|
| Linux x86_64 | CUDA, CPU | tested (RTX 4090) |
| Linux aarch64 | CPU | built in CI |
| macOS (Apple Silicon) | Metal, CPU | built in CI |
| Windows x86_64 | CPU | built in CI |

### Build

```bash
./build.sh            # CPU  -> build/teleocr-linux-x86_64-cpu-<version>, PDFium next to it
./build.sh --cuda     # NVIDIA GPU (CUDA_COMPUTE_CAP is read from nvidia-smi)
./build.sh --metal    # Apple GPU
```

By hand: `cargo build --release -p teleocr-cli [--features cuda]`, then
`scripts/fetch-pdfium.sh target/release`. Prebuilt binaries:
[releases](https://github.com/rzafiamy/teleocr-rs/releases).

### Model

Converted models: [rleo/TeleOCR-GGUF](https://huggingface.co/rleo/TeleOCR-GGUF)
(`teleocr-q8v.gguf` for GPU, `teleocr-q8_0.gguf` for CPU).

```bash
hf download rleo/TeleOCR-GGUF teleocr-q8v.gguf --local-dir models

# or convert the Hugging Face checkpoint yourself (text and vision linears in Q8_0)
teleocr convert ./TeleOCR -o models/teleocr-q8v.gguf --vision-dtype q8_0
```

## Usage

```bash
# Full document parsing: layout, then every block, to Markdown
teleocr parse -m models/teleocr-q8v.gguf paper.pdf --pages 1-3
teleocr parse -m models/teleocr-q8v.gguf photo.jpg --mode segmentation   # curved / skewed pages
teleocr parse -m models/teleocr-q8v.gguf page.png --json                 # blocks with bbox, type, content

# One task on one image (text, table, formula, code, layout, layout_seg, figure, seal, or a prompt)
teleocr run -m models/teleocr-q8v.gguf table.png -t table

# HTTP server
teleocr serve -m models/teleocr-q8v.gguf --port 8090
```

`teleocr run` uses the model card's decoding (repetition penalty 1.05);
`parse` and the server use the official pipeline's per-type settings
(presence / frequency penalties, no repeated 100-grams).

## HTTP API

- `POST /v1/ocr`: JSON `{image: base64 | data URI, task?, prompt?, mode?,
  paratext?, max_tokens?, resize?, pages?, dpi?}` or multipart (`file` + the
  same fields). `task=parse` (default) → `{markdown, blocks, stats}`; a PDF →
  `{markdown, pages}`. Other tasks → `{content, raw, timings}`.
- `POST /v1/chat/completions`: OpenAI shape, one `image_url` (data URI) plus
  text (a task name or a prompt).
- `GET /health`, `GET /v1/models`.
- Errors: 400 with `{"error": {"message": …}}` for a missing or unreadable
  image, an unknown task, or a PDF with a task other than `parse`.

```bash
curl -s localhost:8090/v1/ocr -F file=@page.pdf -F pages=1-2 | jq -r .markdown
```

Used by [zallama](https://github.com/rzafiamy/zallama) as the `teleocr-server`
backend (`modality: ocr`).

## Configuration

**Location**: there is no configuration file. The only files on disk are
the model you point to and the PDFium library, wherever you keep them —
suggested: `~/.local/share/teleocr/` on Linux,
`~/Library/Application Support/teleocr/` on macOS, `%APPDATA%\teleocr\` on
Windows, or `models/` in a checkout. `.env` files are read by your shell,
not by the binary. Everything is set by command-line options
(`teleocr <command> --help`) or environment variables — template:
[`teleocr.example.env`](teleocr.example.env), copy to `.env` and load it
(`set -a; . ./.env; set +a`) or export the variables. To change a setting,
edit the option or variable and restart the process; options take
precedence over variables. Model settings (layers, tokenizer, image
preprocessing) are inside the GGUF. Under zallama, the registry entry's
`params` set these options.

| Variable | Option | Default | Meaning |
|---|---|---|---|
| `TELEOCR_MODEL` | `-m`, `--model` | — | GGUF file or Hugging Face directory |
| `TELEOCR_CPU` | `--cpu` | false | Run on the CPU even with a GPU |
| `TELEOCR_THREADS` | `--threads` | all cores | CPU threads |
| `TELEOCR_MAX_PIXELS` | `--max-pixels` | 12845056 | Pixel cap of an image before the vision tower |
| `TELEOCR_MAX_PAGE_PIXELS` | `--max-page-pixels` | 4500000 | Pages above it are downscaled before cropping |
| `TELEOCR_KV_BUDGET` | `--kv-budget` | 16384 | KV positions per decoding batch (3.7 GB) |
| `TELEOCR_HOST` / `TELEOCR_PORT` | `--host` / `--port` | 127.0.0.1 / 8090 | Server address |
| `TELEOCR_MODEL_ID` | `--model-id` | teleocr | Id reported by `/v1/models` |
| `TELEOCR_BATCH` | `--batch` | 8 | Sequences decoded together (`serve`; `parse` has its own `--batch`) |
| `PDFIUM_LIB_PATH` | — | binary's directory | Directory holding `libpdfium.so` / `.dylib` / `pdfium.dll` |
| `RUST_LOG` | — | off | Log level (`info`, `debug`) |
| `CUDA_COMPUTE_CAP` | — | from `nvidia-smi` | Build time only: GPU architecture for `--cuda` |

No secret or token is needed: everything runs from local files.

**Verify** a configuration:

```bash
teleocr serve --help                  # each option with its [env: …] variable and current value
RUST_LOG=info teleocr run -m "$TELEOCR_MODEL" tests/data/text.png   # loads the model, prints the text
curl -s localhost:8090/health         # {"status":"ok"} once the model is loaded
curl -s localhost:8090/v1/models      # the model id in use
```

The log line `model loaded in …s on Cuda(…)` (with `RUST_LOG=info`) shows
the device in use.

## Tests

```bash
cargo test --release --workspace                                    # unit tests
TELEOCR_MODEL=$PWD/models/teleocr-q8_0.gguf \
  cargo test --release -p teleocr --test model                       # model tests (CPU, ~2 min)
E2E_ARGS=--cpu TELEOCR_MODEL=$PWD/models/teleocr-q8_0.gguf tests/e2e.sh   # CLI, PDF, HTTP API
```

CI (`.github/workflows/ci.yml`) runs fmt, clippy and the tests on Linux,
macOS and Windows; model tests skip there (no weights). Manual tests (parity,
GPU speed, memory): [spec/manual-tests.md](spec/manual-tests.md).

## Documentation

- [docs/architecture.md](docs/architecture.md): model and pipeline stages.
- [docs/performance.md](docs/performance.md): parity, speed, memory.
- [spec/](spec/specification.md): requirements, [traceability matrix](spec/matrix.md),
  [manual tests](spec/manual-tests.md).
- [portfolio/](portfolio/README.md): overview and demo.
- [CHANGELOG.md](CHANGELOG.md) · [CREDITS.md](CREDITS.md) ·
  [CONTRIBUTING.md](CONTRIBUTING.md) · [AGENTS.md](AGENTS.md) · [TODO.md](TODO.md)
- Issues: https://github.com/rzafiamy/teleocr-rs/issues

## License

Code: Apache-2.0. Model weights: Apache-2.0 (XingChen-AGI/TeleOCR).
