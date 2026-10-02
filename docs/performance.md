# Performance and accuracy (2026-10-02)

Machine: RTX 4090 24 GB, 32-thread CPU. Reference: transformers 4.57.6,
`modeling_naviocr.py`, F32 on CPU (`scripts/ref.py`).

## Parity

Greedy tokens vs the reference, model-card samples (`examples/parity.rs`):

| sample | tokens | HF dir F32 (CPU) | q8_0 GGUF CPU | q8_0 CUDA | q8v CPU / CUDA |
|---|---|---|---|---|---|
| text | 65 | exact | exact | exact | exact |
| formula | 165 | exact | exact | exact | exact |
| table (OTSL) | 282 | exact | exact | exact | exact |
| code | 365 | exact | exact | exact | exact |
| layout (1036²) | 767 | exact | exact | exact | exact |

Pixel values differ from the fast HF processor by ≤ 1/255 on a few pixels
(bicubic rounding); vision-tower output cosine ≥ 0.99999.
q8_0 = text Q8_0 + vision F16 (1.97 GB); q8v = text and vision Q8_0 (1.48 GB).
Both also give identical Markdown on 6 pages of the TeleOCR paper.

## Speed (CUDA, q8_0 / q8v alike)

| stage | time |
|---|---|
| vision, 238 image tokens | 0.11 s |
| vision, 1036² page (1369 tokens) | 0.38 s |
| prefill, 1396 tokens | 0.16 s |
| decode, one sequence | ~225 tokens/s |

Document parsing, TeleOCR paper pages 1-3 (71 blocks, 3526 tokens):

| mode | wall |
|---|---|
| one sequence at a time | 19.2 s |
| blocks of a page batched (16) | 16.3 s |
| layouts of all pages batched, then all blocks (batch 8) | 11.5 s |

Six pages: 22.0 s (batch 4), 19.4 s (batch 8), 18.6 s (batch 16).

## Memory (CUDA, q8v, 6 pages)

| batch | peak over baseline |
|---|---|
| 4 | 4.9 GB |
| 8 | 6.4 GB |
| 16 | 8.6 GB |

The F32 KV cache dominates: 229 KB per token per sequence (28 layers × K/V ×
8 heads × 128), and a layout sequence is ~1900 tokens.

## CPU (q8_0 GGUF, 16 threads)

Decode 33 tokens/s; vision 3.6 s for 238 image tokens; prefill of 268 tokens
4.6 s (Q8_0 matmuls are slow for long inputs on CPU; dense F32 prefill is
1.8 s but dense decode only 4.6 tokens/s).

On CPU use the q8_0 GGUF (vision F16): with Q8_0 vision weights (q8v) the
vision tower took 35 s for 464 image tokens (formula sample), against 3.6 s
for 238 tokens with F16 weights: candle's CPU Q8_0 matmul is slow for many
rows. On CUDA both are equal.
