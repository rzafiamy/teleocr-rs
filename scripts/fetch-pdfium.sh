#!/usr/bin/env bash
# Downloads a PDFium shared library (bblanchon/pdfium-binaries) matching the
# pdfium-render bindings, next to the teleocr binary or into $1.
#   scripts/fetch-pdfium.sh [dest-dir]
set -euo pipefail
VERSION="${PDFIUM_VERSION:-7881}"
DEST="${1:-target/release}"
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)  PKG=pdfium-linux-x64; LIB=lib/libpdfium.so ;;
  Linux-aarch64) PKG=pdfium-linux-arm64; LIB=lib/libpdfium.so ;;
  Darwin-arm64)  PKG=pdfium-mac-arm64; LIB=lib/libpdfium.dylib ;;
  Darwin-x86_64) PKG=pdfium-mac-x64; LIB=lib/libpdfium.dylib ;;
  MINGW*-x86_64|MSYS*-x86_64|CYGWIN*-x86_64) PKG=pdfium-win-x64; LIB=bin/pdfium.dll ;;
  *) echo "unsupported platform $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
URL="https://github.com/bblanchon/pdfium-binaries/releases/download/chromium%2F${VERSION}/${PKG}.tgz"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
echo "==> $URL"
curl -fsSL "$URL" | tar -xz -C "$TMP"
mkdir -p "$DEST"
cp "$TMP/$LIB" "$DEST/"
echo "==> $DEST/$(basename "$LIB")"
