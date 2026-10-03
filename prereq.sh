#!/usr/bin/env bash
# prereq.sh — checks (and installs when missing) the teleocr-rs toolchain.
#
#   ./prereq.sh                 check and install what is missing (Rust via rustup, build packages)
#   CHECK_ONLY=1 ./prereq.sh    check only, install nothing
#
# GPU builds also need the CUDA Toolkit (nvcc) for NVIDIA or Xcode for Metal.
# The CUDA Toolkit is not installed automatically (size, version tied to the
# driver); the script only reports whether it is present. Idempotent.
set -euo pipefail
cd "$(dirname "$0")"

ok()   { printf '  \033[32m✔\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✘\033[0m %s\n' "$*" >&2; exit 1; }
SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO="sudo -n"

OS=$(uname -s)
echo "System: $OS"
case "$OS" in
  Linux)
    if [ "${CHECK_ONLY:-0}" != 1 ] && { ! command -v cc >/dev/null || ! command -v pkg-config >/dev/null; }; then
      if command -v apt-get >/dev/null; then
        $SUDO apt-get update && $SUDO apt-get install -y build-essential pkg-config libssl-dev curl git \
          || die "cannot install build packages (rerun with sudo)"
      elif command -v dnf >/dev/null; then
        $SUDO dnf install -y gcc gcc-c++ make pkgconf-pkg-config openssl-devel curl git || die "dnf failed"
      elif command -v pacman >/dev/null; then
        $SUDO pacman -S --needed --noconfirm base-devel openssl curl git || die "pacman failed"
      else
        die "unknown package manager: install a C/C++ compiler, pkg-config, OpenSSL headers, curl and git"
      fi
    fi ;;
  Darwin)
    xcode-select -p >/dev/null 2>&1 || die "install the Xcode Command Line Tools: xcode-select --install" ;;
  MINGW*|MSYS*|CYGWIN*)
    warn "Windows: install Visual Studio Build Tools (C++); CUDA through the NVIDIA CUDA Toolkit" ;;
  *) warn "untested system: $OS" ;;
esac
{ command -v cc >/dev/null || command -v cl >/dev/null; } && ok "C/C++ compiler" || die "no C/C++ compiler"
command -v git >/dev/null && ok "git" || die "git missing"
command -v curl >/dev/null && ok "curl" || die "curl missing"

if ! command -v cargo >/dev/null && [ -x "$HOME/.cargo/bin/cargo" ]; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi
if ! command -v cargo >/dev/null; then
  [ "${CHECK_ONLY:-0}" = 1 ] && die "Rust missing (https://rustup.rs)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    || die "cannot install rustup"
  export PATH="$HOME/.cargo/bin:$PATH"
fi
# rust-toolchain.toml pins the version; rustup installs it on first use.
ok "Rust $(cargo --version | cut -d' ' -f2) (pinned by rust-toolchain.toml)"

NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc 2>/dev/null || true)
if [ -n "$NVCC" ]; then
  ok "CUDA: $("$NVCC" --version | grep -o 'release [0-9.]*')  -> ./build.sh --cuda"
else
  warn "nvcc not found: CPU build only (install CUDA Toolkit >= 12 for NVIDIA GPUs)"
fi
command -v python3 >/dev/null && ok "python3 (tests/e2e.sh, scripts/ref.py)" || warn "python3 missing (needed by tests/e2e.sh)"

# PDF input loads PDFium at run time (scripts/fetch-pdfium.sh downloads it).
if compgen -G "target*/release/libpdfium.*" >/dev/null || compgen -G "build/libpdfium.*" >/dev/null \
   || [ -n "${PDFIUM_LIB_PATH:-}" ]; then
  ok "PDFium (PDF input)"
else
  warn "PDFium not downloaded yet: ./build.sh fetches it (or scripts/fetch-pdfium.sh <dir>)"
fi

echo "Final check:"
cargo --version >/dev/null && rustc --version >/dev/null && ok "prerequisites OK"
