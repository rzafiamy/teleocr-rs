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

Details: [docs/performance.md](docs/performance.md).

## Build

```bash
cargo build --release -p teleocr-cli                      # CPU
PATH=/usr/local/cuda/bin:$PATH CUDA_COMPUTE_CAP=89 \
  cargo build --release -p teleocr-cli --features cuda --target-dir target-cuda
scripts/fetch-pdfium.sh target/release                     # PDF input (PDFium next to the binary)
```

## Use

Converted models: [rleo/TeleOCR-GGUF](https://huggingface.co/rleo/TeleOCR-GGUF)
(`teleocr-q8v.gguf` for GPU, `teleocr-q8_0.gguf` for CPU).

```bash
hf download rleo/TeleOCR-GGUF teleocr-q8v.gguf --local-dir .

# or convert the Hugging Face checkpoint yourself (text and vision linears in Q8_0)
teleocr convert ./TeleOCR -o teleocr-q8v.gguf --vision-dtype q8_0

# Full document parsing: layout, then every block, to Markdown
teleocr parse -m teleocr-q8v.gguf paper.pdf --pages 1-3
teleocr parse -m teleocr-q8v.gguf photo.jpg --mode segmentation   # curved / skewed pages
teleocr parse -m teleocr-q8v.gguf page.png --json                 # blocks with bbox, type, content

# One task on one image (text, table, formula, code, layout, layout_seg, figure, seal, or a prompt)
teleocr run -m teleocr-q8v.gguf table.png -t table

# HTTP server
teleocr serve -m teleocr-q8v.gguf --port 8090
```

`teleocr run` uses the model card's decoding (repetition penalty 1.05);
`parse` and the server use the official pipeline's per-type settings
(presence / frequency penalties, no repeated 100-grams).

## HTTP API

- `POST /v1/ocr`: JSON `{image: base64 | data URI, task?, mode?, paratext?,
  max_tokens?, pages?, dpi?}` or multipart (`file` + the same fields).
  `task=parse` (default) → `{markdown, blocks, stats}`; a PDF → `{markdown, pages}`.
  Other tasks → `{content, raw, timings}`.
- `POST /v1/chat/completions`: OpenAI shape, one `image_url` (data URI) plus
  text (a task name or a prompt).
- `GET /health`, `GET /v1/models`.

Used by [zallama](https://github.com/rzafiamy/zallama) as the `teleocr-server`
backend (`modality: ocr`).

## License

Code: Apache-2.0. Model weights: Apache-2.0 (XingChen-AGI/TeleOCR).
