# TODO

- KV cache in F16/BF16 (halves the 229 KB/token F32 cache that dominates
  batched memory); check parity stays exact or near.
- Single-sequence decode is launch-bound (~4.4 ms/token, 28 layers): fuse
  q/k/v and gate/up projections; CUDA graphs if candle gets them.
- Layout pass is now the critical path of a page (one long sequence):
  overlap layout of page N+1 with block decoding of page N.
- CPU prefill: Q8_0 matmuls are slow for long inputs; dequantize once per
  prefill or keep F16 copies for prefill.
- Remaining post-processing of the official client: `unicode_to_latex`,
  `\left`/`\right` matching, unbalanced braces, `\begin`/`\end` fixes,
  LaTeX special-char escaping; title levels; image crops in Markdown.
- Streaming (`stream=true`) on `/v1/chat/completions`.
- Batch the vision tower across crops of similar size.
