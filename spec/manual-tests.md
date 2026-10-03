# Manual tests

Tests that need the Python reference, a GPU or large inputs. Results of the
last run: [docs/performance.md](../docs/performance.md).

## MT-01 — Parity with transformers (REQ-INF-001, REQ-INF-002, REQ-IMG-001, REQ-GGF-002)

Setup: transformers 4.57 in a venv (see `scripts/ref.py`), the
`XingChen-AGI/TeleOCR` checkpoint in `<dir>` (its `assets/` hold the
model-card samples).

```bash
python scripts/ref.py <dir> <dir>/assets/table.png table /tmp/table.safetensors
cargo run --release --example parity -- models/teleocr-q8v.gguf <dir>/assets/table.png /tmp/table.safetensors --cpu
# layout: add the resize argument
python scripts/ref.py <dir> <dir>/assets/layout.jpg layout /tmp/layout.safetensors --resize 1036
cargo run --release --example parity -- models/teleocr-q8v.gguf <dir>/assets/layout.jpg /tmp/layout.safetensors 1036
```

Expected: identical greedy tokens on text, formula, table, code and layout,
vision output cosine ≥ 0.99999, with the HF directory (F32), the q8_0 and the
q8v GGUF, on CPU and CUDA. Last run (2026-10-02, and again after the
chunked prefill of 2026-10-02): 5/5 exact everywhere.

## MT-02 — CUDA speed (REQ-GPU-001)

```bash
./build.sh --cuda
time build/teleocr-linux-x86_64-cuda-* parse -m models/teleocr-q8v.gguf paper.pdf --pages 1-3
build/teleocr-linux-x86_64-cuda-* run -m models/teleocr-q8v.gguf tests/data/text.png --timings
```

Expected (RTX 4090): ~225 tokens/s on one sequence, 3 pages of the TeleOCR
paper in ≤ 12 s at batch 8. Last run: 11.5 s.

## MT-03 — Memory on large inputs (REQ-MEM-001, REQ-MEM-002)

Input: a 12 Mpx phone photo of a page (4000×3000 JPEG).

```bash
teleocr serve -m models/teleocr-q8v.gguf --port 18090 &
watch -n 0.5 nvidia-smi --query-compute-apps=pid,used_memory --format=csv
curl -s localhost:18090/v1/ocr -F file=@photo.jpg | jq .stats
```

Expected: no CUDA out-of-memory, peak under ~8 GB, memory back to ~2 GB
once the request ends; with `--kv-budget 4096` the log shows smaller
batches and the same Markdown. Last run (2026-10-02): the photo that ran
the 24 GB GPU out of memory before the fix parses; idle ~2 GB after
requests.
