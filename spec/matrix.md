# Traceability matrix

Requirement ([specification.md](specification.md)) → feature → module → test.
Automated tests carry the ID in a `covers: REQ-…` comment; manual tests are
described in [manual-tests.md](manual-tests.md). Model tests
(`crates/teleocr/tests/model.rs`) run with `TELEOCR_MODEL` set;
`tests/e2e.sh` runs the release binary.

| ID | Requirement | Feature | Module | Test | Status | Notes |
|---|---|---|---|---|---|---|
| REQ-IMG-001 | HF image preprocessing | Vision | `crates/teleocr/src/image.rs` | `smart_resize_matches_python`, manual MT-01 | ✅ | ≤ 1/255 on a few pixels |
| REQ-INF-001 | Parity with transformers | Inference | `vision.rs`, `text.rs`, `model.rs` | `text_sample_matches_reference`, `table_sample_matches_reference`, manual MT-01 | ✅ | 5/5 samples exact |
| REQ-INF-002 | Decoding presets | Inference | `model.rs` (`GenerateOptions::pipeline`) | `tests/e2e.sh` (`run` exact, server), manual MT-01 | ✅ | |
| REQ-GGF-001 | Single GGUF | GGUF | `crates/teleocr/src/gguf.rs`, `crates/teleocr-cli/src/main.rs` | `text_sample_matches_reference`, `tests/e2e.sh` (with `TELEOCR_HF`) | ✅ | |
| REQ-GGF-002 | Weight dtypes | GGUF | `gguf.rs` (`parse_dtype`), `weights.rs` | `tests/e2e.sh` (with `TELEOCR_HF`), manual MT-01 | ✅ | q8v 1.48 GB, q8_0 1.97 GB |
| REQ-PIP-001 | Layout pass | Pipeline | `crates/teleocr/src/pipeline.rs` (`parse_layout`) | `layout_lines`, `parse_page_gives_markdown` | ✅ | |
| REQ-PIP-002 | Blocks to Markdown | Pipeline | `pipeline.rs` (`parse_pages`, `to_markdown`) | `parse_page_gives_markdown`, `tests/e2e.sh` | ✅ | |
| REQ-PIP-003 | Post-processing | Pipeline | `pipeline.rs` (`post_equation`, `post_text`, `split_code`) | `equations`, `inline_math`, `code` | ✅ | partial, see limitations |
| REQ-TAB-001 | OTSL → HTML | Tables | `crates/teleocr/src/otsl.rs` | `spans`, `empty_and_escape`, `table_sample_matches_reference` | ✅ | |
| REQ-BAT-001 | Batched decoding | Batching | `model.rs` (`generate_batch`) | `batched_decoding_matches_single` | ✅ | |
| REQ-MEM-001 | Bounded GPU memory | Memory | `text.rs` (chunked prefill), `pipeline.rs`, `lib.rs` (`release_cached_memory`) | manual MT-03 | ✅ | 14.4 → 5.3 GB peak on a 4.5 Mpx crop |
| REQ-MEM-002 | KV budget, OOM retry | Memory | `model.rs` (`generate_batch`, `set_kv_budget`) | `tiny_kv_budget_still_decodes_every_job`, manual MT-03 | ✅ | |
| REQ-PDF-001 | PDF input | PDF | `crates/teleocr-cli/src/pdf.rs` | `page_ranges`, `tests/e2e.sh` | ✅ | needs libpdfium |
| REQ-CLI-001 | `teleocr run` | CLI | `crates/teleocr-cli/src/main.rs` | `tests/e2e.sh` | ✅ | |
| REQ-CLI-002 | `teleocr parse` | CLI | `main.rs` | `tests/e2e.sh` | ✅ | Markdown, `--json`, PDF |
| REQ-CLI-003 | `TELEOCR_*` variables | Configuration | `main.rs` (clap `env`) | `tests/e2e.sh` (`TELEOCR_MODEL_ID`) | ✅ | [teleocr.example.env](../teleocr.example.env) |
| REQ-SRV-001 | `/v1/ocr` | Server | `crates/teleocr-cli/src/server.rs` (`ocr`, `parse_pdf`) | `tests/e2e.sh` | ✅ | JSON, multipart, PDF |
| REQ-SRV-002 | `/v1/chat/completions` | Server | `server.rs` (`chat`) | `tests/e2e.sh` | ✅ | |
| REQ-SRV-003 | `/health`, `/v1/models`, errors | Server | `server.rs` | `tests/e2e.sh` | ✅ | 400 on bad input |
| REQ-GPU-001 | CUDA speed and memory | GPU | `model.rs`, `vision.rs` | manual MT-02 | ✅ | [docs/performance.md](../docs/performance.md) |

Updated: 2026-10-03 (v0.2.0).
