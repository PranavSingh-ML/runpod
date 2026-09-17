#!/usr/bin/env bash
# One-command pod lifecycle for the imgedit serving pod. Needs only bash (Git Bash) + node.
#
#   bash scripts/pod.sh up        create a pod from IMAGE (A40 first, A6000 fallback), write POD_URL to web/.env.local
#   bash scripts/pod.sh status    list pods + $/hr + balance
#   bash scripts/pod.sh down      DELETE every pod named imgedit* (this is the only thing that stops billing)
#   bash scripts/pod.sh wait      block until /health reports model_loaded=true
#
# Needs tools/runpodctl.exe configured (tools\runpodctl.exe doctor).
set -euo pipefail
cd "$(dirname "$0")/.."
RP="${RUNPODCTL:-./tools/runpodctl.exe}"
[ -x "$RP" ] || RP="./tools/runpodctl"
[ -x "$RP" ] || { echo "runpodctl not found in tools/ - see README (download it, then: tools\\runpodctl.exe doctor)"; exit 1; }
command -v node >/dev/null || { echo "node not found on PATH - install Node.js and open a new terminal"; exit 1; }
IMAGE="${IMAGE:-ghcr.io/pranavsingh-ml/imgedit:v1}"
DISK="${DISK:-100}"
ENVF="web/.env.local"
REG_AUTH="${REGISTRY_AUTH_ID:-}"   # set if the GHCR package is private (runpodctl registry list)

# js <script> : run a node one-liner with stdin JSON available as `input` (string)
js() { node -e "let input='';process.stdin.on('data',d=>input+=d).on('end',()=>{ $1 })"; }
token() { grep -E '^API_TOKEN=' "$ENVF" 2>/dev/null | cut -d= -f2- | tr -d '\r' || true; }
pods_json() { "$RP" pod list -o json 2>/dev/null || echo "[]"; }
imgedit_ids() { pods_json | js 'const ps=JSON.parse(input||"[]"); console.log(ps.filter(p=>(p.name||"").startsWith("imgedit")).map(p=>p.id).join(" "))'; }

case "${1:-}" in
  up)
    TOKEN="$(token)"
    if [ -z "$TOKEN" ]; then
      TOKEN="$(node -e 'console.log(require("crypto").randomBytes(32).toString("base64url"))')"
      printf 'POD_URL=\nAPI_TOKEN=%s\nPOD_RATE_USD_HR=0.49\nANTHROPIC_API_KEY=\nREWRITE_ENABLED=0\n' "$TOKEN" > "$ENVF"
      echo "generated API_TOKEN -> $ENVF"
    fi
    EXISTING="$(imgedit_ids)"
    if [ -n "$EXISTING" ]; then echo "a pod already exists: $EXISTING  (use: pod.sh status / wait / down)"; exit 1; fi
    try() {
      echo "== trying $1 ($2)"
      "$RP" pod create --name imgedit --image "$IMAGE" --gpu-id "$1" --cloud-type "$2" \
        --container-disk-in-gb "$DISK" --ports "8000/http,22/tcp" ${REG_AUTH:+--registry-auth-id "$REG_AUTH"} \
        --env "{\"API_TOKEN\":\"$TOKEN\",\"HF_HOME\":\"/workspace/hf\"${EXTRA_ENV:+,$EXTRA_ENV}}" -o json > .pod_create.json 2>&1 || true
      if grep -q '"error"' .pod_create.json; then js 'console.log("  ", (JSON.parse(input).error||"").slice(0,120))' < .pod_create.json; return 1; fi
      grep -q '"id"' .pod_create.json || { echo "  unexpected response:"; head -c 300 .pod_create.json; echo; return 1; }
    }
    try "NVIDIA A40" SECURE || try "NVIDIA RTX A6000" SECURE || try "NVIDIA RTX A6000" COMMUNITY || { echo "no 48GB Ampere card available right now; try again in a few minutes"; exit 1; }
    ID="$(js 'console.log(JSON.parse(input).id)' < .pod_create.json)"
    RATE="$(js 'console.log(JSON.parse(input).costPerHr ?? "")' < .pod_create.json)"
    rm -f .pod_create.json
    URL="https://${ID}-8000.proxy.runpod.net"
    sed -i "s|^POD_URL=.*|POD_URL=${URL}|; s|^POD_RATE_USD_HR=.*|POD_RATE_USD_HR=${RATE:-0.49}|" "$ENVF"
    echo "created pod $ID at \$${RATE}/hr"
    echo "POD_URL=$URL  (written to $ENVF)"
    echo "cold boot takes ~6-12 min; run:  bash scripts/pod.sh wait   then  cd web && npm run dev"
    echo "WHEN DONE:  bash scripts/pod.sh down"
    ;;
  status)
    pods_json | js '
      const ps=JSON.parse(input||"[]");
      if(!ps.length) console.log("no pods running - not being billed");
      for(const p of ps) console.log(`${p.id}  ${p.name}  ${p.desiredStatus}  $${p.costPerHr}/hr  up ${Math.floor((p.uptimeSeconds||0)/60)} min  https://${p.id}-8000.proxy.runpod.net`);'
    "$RP" user -o json 2>/dev/null | js 'const u=JSON.parse(input||"{}"); if(u.clientBalance!==undefined) console.log(`balance $${u.clientBalance.toFixed(2)}   spend/hr $${u.currentSpendPerHr}`)'
    ;;
  down)
    IDS="$(imgedit_ids)"
    [ -n "$IDS" ] || { echo "no imgedit pods to delete"; exit 0; }
    for id in $IDS; do "$RP" pod delete "$id" -o json >/dev/null && echo "deleted $id"; done
    LEFT="$(pods_json | js 'console.log(JSON.parse(input||"[]").length)')"
    echo "pods still running (any name): $LEFT"
    ;;
  wait)
    URL="$(grep -E '^POD_URL=' "$ENVF" 2>/dev/null | cut -d= -f2- | tr -d '\r' || true)"
    [ -n "$URL" ] || { echo "POD_URL not set in $ENVF - run: bash scripts/pod.sh up"; exit 1; }
    for i in $(seq 1 120); do
      H="$(curl -s --max-time 15 "$URL/health" || true)"
      if [ -n "$H" ]; then
        L="$(echo "$H" | js 'const d=JSON.parse(input); console.log(d.model_loaded, d.gpu, d.vram_used_gb, (d.load_error||"").slice(0,200))')"
        echo "[$i] $L"
        echo "$L" | grep -q '^true' && { echo "READY: $URL"; exit 0; }
        echo "$L" | grep -q 'Traceback\|Error' && { echo "MODEL LOAD FAILED - run: bash scripts/pod.sh down"; exit 1; }
      else echo "[$i] booting..."; fi
      sleep 10
    done
    echo "gave up after 20 min - check: bash scripts/pod.sh status"; exit 1
    ;;
  *) sed -n 2,9p "$0"; exit 1 ;;
esac
