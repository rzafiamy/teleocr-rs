#!/usr/bin/env bash
# setup.sh — prepares a development checkout: prerequisites, Rust
# dependencies, compile check. Idempotent and non-interactive.
#
#   ./setup.sh            CPU
#   ./setup.sh --cuda     with NVIDIA GPU support
set -euo pipefail
cd "$(dirname "$0")"
FEATURES=()
case "${1:-}" in
  --cuda) FEATURES=(--features teleocr-cli/cuda) ;;
  "") ;;
  *) echo "unknown option: $1 (--cuda)" >&2; exit 1 ;;
esac

./prereq.sh
export PATH="$HOME/.cargo/bin:$PATH"
# Tools that the steps below need (prereq.sh installs them when missing).
for tool in cargo rustc git curl; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool (rerun ./prereq.sh)" >&2; exit 1; }
done
[ -d /usr/local/cuda/bin ] && export PATH="/usr/local/cuda/bin:$PATH"
if [ "${1:-}" = "--cuda" ] && ! command -v nvcc >/dev/null; then
  echo "nvcc not found: install the CUDA Toolkit or run ./setup.sh without --cuda" >&2; exit 1
fi
mkdir -p models
cargo fetch --locked || { echo "cargo fetch failed (network?)" >&2; exit 1; }
cargo check --workspace --locked "${FEATURES[@]}"
echo "setup OK — next: ./build.sh ${1:-} then hf download rleo/TeleOCR-GGUF teleocr-q8v.gguf --local-dir models"
