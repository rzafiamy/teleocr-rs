#!/usr/bin/env bash
# tests/e2e.sh — end-to-end check of the release binary: GGUF conversion,
# `run`, `parse` (image and PDF), and the HTTP server (/health, /v1/models,
# /v1/ocr in JSON and multipart, /v1/chat/completions, errors).
#
#   TELEOCR_HF=<TeleOCR HF dir>     tests/e2e.sh [binary]   converts, then tests the new GGUF
#   TELEOCR_MODEL=<model.gguf>      tests/e2e.sh [binary]   tests an existing GGUF
#
# binary: default target/release/teleocr (build with ./build.sh or cargo
# build --release). PDF steps need libpdfium next to the binary
# (scripts/fetch-pdfium.sh) and are skipped otherwise. Add --cpu through
# E2E_ARGS="--cpu" to keep the GPU free.
#
# covers: REQ-GGF-001, REQ-GGF-002, REQ-CLI-001, REQ-CLI-002, REQ-CLI-003, REQ-PDF-001, REQ-SRV-001, REQ-SRV-002, REQ-SRV-003
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="${1:-target/release/teleocr}"
[ -x "$BIN" ] || { echo "binary not found: $BIN (run ./build.sh or cargo build --release)" >&2; exit 1; }
WORK=$(mktemp -d)
PORT=${PORT:-18990}
ARGS=(${E2E_ARGS:-})
SERVER=""
cleanup() { [ -n "$SERVER" ] && kill "$SERVER" 2>/dev/null; rm -rf "$WORK"; }
trap cleanup EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1]))" "$1"; }
DATA=tests/data
EXPECTED=$(cat "$DATA/text.expected.txt")
PDF=0; [ -e "$(dirname "$BIN")/libpdfium.so" ] || [ -e "$(dirname "$BIN")/libpdfium.dylib" ] || [ -n "${PDFIUM_LIB_PATH:-}" ] && PDF=1

if [ -n "${TELEOCR_HF:-}" ]; then
  echo "== convert q8_0 (vision f16)"
  "$BIN" convert "$TELEOCR_HF" -o "$WORK/m.gguf" >/dev/null 2>&1 || fail "convert"
  size=$(stat -c %s "$WORK/m.gguf")
  [ "$size" -lt 2100000000 ] || fail "q8_0 file too large: $size bytes"
  MODEL="$WORK/m.gguf"
else
  MODEL="${TELEOCR_MODEL:?set TELEOCR_HF (checkpoint dir) or TELEOCR_MODEL (GGUF)}"
fi

echo "== run -t text"
out=$("$BIN" run -m "$MODEL" "${ARGS[@]}" "$DATA/text.png" -t text 2>/dev/null) || fail "run"
[ "$out" = "$EXPECTED" ] || fail "run: unexpected text: $out"

echo "== parse --json"
"$BIN" parse -m "$MODEL" "${ARGS[@]}" "$DATA/text.png" --json >"$WORK/page.json" 2>/dev/null || fail "parse"
json "d[0]['markdown']" <"$WORK/page.json" | grep -q "Hybrid methods may be required" || fail "parse: markdown"

if [ "$PDF" = 1 ]; then
  echo "== parse PDF"
  "$BIN" parse -m "$MODEL" "${ARGS[@]}" "$DATA/text.pdf" --pages 1-5 2>/dev/null | grep -q "Hybrid methods" \
    || fail "parse PDF"
else
  echo "-- PDFium not found: PDF steps skipped"
fi

echo "== serve"
TELEOCR_MODEL_ID=teleocr-e2e "$BIN" serve -m "$MODEL" "${ARGS[@]}" --port "$PORT" >"$WORK/server.log" 2>&1 &
SERVER=$!
for _ in $(seq 240); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 0.5; done
curl -sf "localhost:$PORT/health" >/dev/null || { cat "$WORK/server.log"; fail "server did not start"; }
URL="localhost:$PORT"

[ "$(curl -s "$URL/v1/models" | json "d['data'][0]['id']")" = teleocr-e2e ] || fail "/v1/models"

B64=$(base64 -w0 "$DATA/text.png")
printf '{"image":"data:image/png;base64,%s","task":"text"}' "$B64" >"$WORK/text.json"
content=$(curl -sf "$URL/v1/ocr" -H 'content-type: application/json' -d @"$WORK/text.json" | json "d['content']") \
  || fail "/v1/ocr task=text"
# The server uses the official pipeline's prompts and penalties, not the model card's.
grep -q "Hybrid methods may be required" <<<"$content" || fail "/v1/ocr task=text: $content"

curl -sf "$URL/v1/ocr" -F file=@"$DATA/table.png" -F task=table | json "d['content']" | grep -q "<table" \
  || fail "/v1/ocr multipart task=table"

curl -sf "$URL/v1/ocr" -F file=@"$DATA/text.png" | json "d['markdown']" | grep -q "Hybrid methods" \
  || fail "/v1/ocr parse"

if [ "$PDF" = 1 ]; then
  n=$(curl -sf "$URL/v1/ocr" -F file=@"$DATA/text.pdf" | json "len(d['pages'])") || fail "/v1/ocr PDF"
  [ "$n" = 1 ] || fail "/v1/ocr PDF: $n pages"
fi

printf '{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}},{"type":"text","text":"text"}]}]}' \
  "$B64" >"$WORK/chat.json"
curl -sf "$URL/v1/chat/completions" -H 'content-type: application/json' -d @"$WORK/chat.json" \
  | json "d['choices'][0]['message']['content']" | grep -q "Hybrid methods" || fail "/v1/chat/completions"

code=$(curl -s -o /dev/null -w '%{http_code}' "$URL/v1/ocr" -H 'content-type: application/json' -d '{"task":"text"}')
[ "$code" = 400 ] || fail "missing image returned $code, expected 400"
code=$(curl -s -o /dev/null -w '%{http_code}' "$URL/v1/ocr" -H 'content-type: application/json' \
  -d "{\"image\":\"$B64\",\"task\":\"poem\"}")
[ "$code" = 400 ] || fail "unknown task returned $code, expected 400"
code=$(curl -s -o /dev/null -w '%{http_code}' "$URL/v1/ocr" -H 'content-type: application/json' -d '{"image":"bm90IGFuIGltYWdl"}')
[ "$code" = 400 ] || fail "unreadable image returned $code, expected 400"

echo "e2e OK"
