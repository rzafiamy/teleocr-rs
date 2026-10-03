# Contributing

1. `./setup.sh` (or `./setup.sh --cuda`) checks the toolchain and compiles.
2. Make the change; keep the Python reference (`modeling_naviocr.py`,
   transformers 4.57) as the spec for the model and the official client
   (`TeleOCR_client.py`) as the spec for the pipeline — see [AGENTS.md](AGENTS.md).
3. Before a pull request:
   ```bash
   cargo fmt --all
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --release --workspace
   TELEOCR_MODEL=$PWD/models/teleocr-q8_0.gguf cargo test --release -p teleocr --test model
   scripts/fetch-pdfium.sh target/release
   E2E_ARGS=--cpu TELEOCR_MODEL=$PWD/models/teleocr-q8_0.gguf tests/e2e.sh target/release/teleocr
   ```
4. Model changes: greedy parity must stay exact on the five model-card
   samples (`scripts/ref.py` + `cargo run --release --example parity`, see
   [spec/manual-tests.md](spec/manual-tests.md)).
5. New requirement → a `REQ-…` line in [spec/specification.md](spec/specification.md),
   a row in [spec/matrix.md](spec/matrix.md) and a test carrying `covers: REQ-…`.
6. Note user-visible changes under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md).

Commits follow Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:` …).
