#!/usr/bin/env bash
# Phase 1 acceptance test. Runs on the LAPTOP against a live pod, cold, with nobody SSHing in.
#   POD_URL=https://<POD_ID>-8000.proxy.runpod.net API_TOKEN=... scripts/smoke_test.sh [image.png]
# Needs: curl, python3 (for JSON parsing). Exit 0 = pass.
set -euo pipefail
POD_URL="${POD_URL:?set POD_URL}"; POD_URL="${POD_URL%/}"
API_TOKEN="${API_TOKEN:?set API_TOKEN}"
IMG="${1:-}"
OUT="${OUT:-smoke_out.png}"
AUTH=(-H "Authorization: Bearer ${API_TOKEN}")
py() { python3 -c "import sys,json; d=json.load(sys.stdin); print($1)"; }

if [ -z "$IMG" ]; then
  IMG="$(mktemp -t smoke_XXXX).png"
  python3 - "$IMG" <<'PY'
import sys
from PIL import Image, ImageDraw
im = Image.new("RGB", (768, 512), (70, 130, 180)); d = ImageDraw.Draw(im)
d.rectangle([0, 340, 768, 512], fill=(60, 140, 60)); d.ellipse([300, 100, 460, 260], fill=(250, 220, 80))
im.save(sys.argv[1])
PY
fi

echo "== 1. /health (no auth) - waiting for model_loaded=true (cold boot can take 5-15 min) =="
for i in $(seq 1 180); do
  H="$(curl -sS --max-time 20 "$POD_URL/health" || true)"
  if [ -n "$H" ]; then
    LOADED="$(echo "$H" | py "d.get('model_loaded')")"
    ERR="$(echo "$H" | py "d.get('load_error') or ''" | head -c 300)"
    echo "  [$i] model_loaded=$LOADED gpu=$(echo "$H" | py "d.get('gpu')") cap=$(echo "$H" | py "d.get('capability')") quant=$(echo "$H" | py "d.get('quant')") vram=$(echo "$H" | py "d.get('vram_used_gb')")GB"
    [ -n "$ERR" ] && { echo "MODEL LOAD ERROR: $ERR"; exit 1; }
    [ "$LOADED" = "True" ] && break
  else
    echo "  [$i] no response yet"
  fi
  sleep 10
done
[ "${LOADED:-}" = "True" ] || { echo "FAIL: model never loaded"; exit 1; }

echo "== 2. auth check: /edit without token must be 401 =="
CODE="$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$POD_URL/edit" -F "image=@$IMG" -F "prompt=x")"
[ "$CODE" = "401" ] || { echo "FAIL: expected 401 got $CODE"; exit 1; }
echo "  ok (401)"

echo "== 3. POST /edit =="
T0=$(date +%s.%N)
R="$(curl -sS "${AUTH[@]}" -X POST "$POD_URL/edit" -F "image=@$IMG" -F "prompt=make the sky a dramatic sunset, keep everything else unchanged" -F "seed=42")"
echo "  $R"
JOB="$(echo "$R" | py "d['job_id']")"

echo "== 4. poll /jobs/$JOB =="
for i in $(seq 1 120); do
  S="$(curl -sS "${AUTH[@]}" "$POD_URL/jobs/$JOB")"
  ST="$(echo "$S" | py "d['status']")"
  echo "  status=$ST progress=$(echo "$S" | py "d['progress']") elapsed=$(echo "$S" | py "d['elapsed_s']")s"
  [ "$ST" = "done" ] && break
  [ "$ST" = "error" ] && { echo "FAIL:"; echo "$S" | py "d['error']"; exit 1; }
  sleep 1
done
[ "$ST" = "done" ] || { echo "FAIL: timed out"; exit 1; }

echo "== 5. download PNG =="
curl -sS "${AUTH[@]}" -o "$OUT" "$POD_URL/jobs/$JOB/image"
T1=$(date +%s.%N)
file "$OUT" 2>/dev/null || true
python3 -c "import sys; from PIL import Image; im=Image.open(sys.argv[1]); print('  valid PNG', im.size)" "$OUT"
curl -sS "${AUTH[@]}" -X DELETE "$POD_URL/jobs/$JOB" >/dev/null
echo "PASS: $OUT  seed=$(echo "$S" | py "d['seed']")  server_elapsed=$(echo "$S" | py "d['elapsed_s']")s  wall=$(python3 -c "print(round($T1-$T0,1))")s"
echo "Record server_elapsed and cold-boot time in NOTES.md."
