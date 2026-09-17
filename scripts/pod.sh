#!/usr/bin/env bash
# One-command pod lifecycle for the imgedit serving pod.
#
#   bash scripts/pod.sh up        create a pod from IMAGE (A40 first, A6000 fallback), write POD_URL to web/.env.local
#   bash scripts/pod.sh status    list pods + $/hr + balance
#   bash scripts/pod.sh down      DELETE every pod named imgedit* (this is the only thing that stops billing)
#   bash scripts/pod.sh wait      block until /health reports model_loaded=true
#
# Needs tools/runpodctl.exe configured (tools\runpodctl.exe doctor) and API_TOKEN in web/.env.local.
set -euo pipefail
cd "$(dirname "$0")/.."
RP="${RUNPODCTL:-./tools/runpodctl.exe}"
IMAGE="${IMAGE:-ghcr.io/pranavsingh-ml/imgedit:v1}"
DISK="${DISK:-100}"
ENVF="web/.env.local"

token() { grep -E '^API_TOKEN=' "$ENVF" 2>/dev/null | cut -d= -f2- | tr -d '\r'; }
pods_json() { "$RP" pod list -o json 2>/dev/null; }

case "${1:-}" in
  up)
    TOKEN="$(token)"
    if [ -z "$TOKEN" ]; then
      TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
      printf 'POD_URL=\nAPI_TOKEN=%s\nPOD_RATE_USD_HR=0.49\nANTHROPIC_API_KEY=\nREWRITE_ENABLED=0\n' "$TOKEN" > "$ENVF"
      echo "generated API_TOKEN -> $ENVF"
    fi
    EXISTING="$(pods_json | python -c 'import sys,json; print(" ".join(p["id"] for p in json.load(sys.stdin) if p.get("name","").startswith("imgedit")))')"
    if [ -n "$EXISTING" ]; then echo "a pod already exists: $EXISTING  (use: pod.sh status / down)"; exit 1; fi
    try() {
      echo "== trying $1 ($2)"
      "$RP" pod create --name imgedit --image "$IMAGE" --gpu-id "$1" --cloud-type "$2" \
        --container-disk-in-gb "$DISK" --ports "8000/http,22/tcp" \
        --env "{\"API_TOKEN\":\"$TOKEN\",\"HF_HOME\":\"/workspace/hf\"${EXTRA_ENV:+,$EXTRA_ENV}}" -o json > .pod_create.json 2>&1
      if grep -q '"error"' .pod_create.json; then python -c 'import json; print("  ", json.load(open(".pod_create.json"))["error"][:120])'; return 1; fi
    }
    try "NVIDIA A40" SECURE || try "NVIDIA RTX A6000" SECURE || try "NVIDIA RTX A6000" COMMUNITY || { echo "no 48GB Ampere card available right now; try again in a few minutes"; exit 1; }
    ID="$(python -c 'import json; print(json.load(open(".pod_create.json"))["id"])')"
    RATE="$(python -c 'import json; print(json.load(open(".pod_create.json")).get("costPerHr",""))')"
    rm -f .pod_create.json
    URL="https://${ID}-8000.proxy.runpod.net"
    sed -i "s|^POD_URL=.*|POD_URL=${URL}|; s|^POD_RATE_USD_HR=.*|POD_RATE_USD_HR=${RATE:-0.49}|" "$ENVF"
    echo "created pod $ID at \$${RATE}/hr"
    echo "POD_URL=$URL  (written to $ENVF)"
    echo "cold boot takes ~6-12 min; run:  bash scripts/pod.sh wait   then  cd web && npm run dev"
    echo "WHEN DONE:  bash scripts/pod.sh down"
    ;;
  status)
    PODS="$(pods_json)" USER_JSON="$("$RP" user -o json 2>/dev/null)" python - <<'PY2'
import os, json
ps = json.loads(os.environ["PODS"] or "[]")
if not ps:
    print("no pods running - not being billed")
for p in ps:
    up = int((p.get("uptimeSeconds") or 0) // 60)
    print(f'{p["id"]}  {p.get("name")}  {p.get("desiredStatus")}  ${p.get("costPerHr")}/hr  up {up} min  https://{p["id"]}-8000.proxy.runpod.net')
u = json.loads(os.environ["USER_JSON"] or "{}")
if u:
    print(f'balance ${u.get("clientBalance", 0):.2f}   spend/hr ${u.get("currentSpendPerHr", 0)}')
PY2
    ;;
  down)
    IDS="$(pods_json | python -c 'import sys,json; print(" ".join(p["id"] for p in json.load(sys.stdin) if p.get("name","").startswith("imgedit")))')"
    [ -n "$IDS" ] || { echo "no imgedit pods to delete"; exit 0; }
    for id in $IDS; do "$RP" pod delete "$id" -o json >/dev/null && echo "deleted $id"; done
    LEFT="$(pods_json | python -c 'import sys,json; print(len(json.load(sys.stdin)))')"
    echo "pods still running (any name): $LEFT"
    ;;
  wait)
    URL="$(grep -E '^POD_URL=' "$ENVF" | cut -d= -f2- | tr -d '\r')"
    [ -n "$URL" ] || { echo "POD_URL not set in $ENVF"; exit 1; }
    for i in $(seq 1 120); do
      H="$(curl -s --max-time 15 "$URL/health" || true)"
      if [ -n "$H" ]; then
        L="$(echo "$H" | python -c 'import sys,json; d=json.load(sys.stdin); print(d.get("model_loaded"), d.get("gpu"), d.get("vram_used_gb"), (d.get("load_error") or "")[:200])')"
        echo "[$i] $L"
        echo "$L" | grep -q '^True' && { echo "READY: $URL"; exit 0; }
        echo "$H" | grep -q '"load_error": *"[^n]' && { echo "MODEL LOAD FAILED - run: bash scripts/pod.sh down"; exit 1; }
      else echo "[$i] booting..."; fi
      sleep 10
    done
    echo "gave up after 20 min - check: bash scripts/pod.sh status"; exit 1
    ;;
  *) sed -n 2,9p "$0"; exit 1 ;;
esac
