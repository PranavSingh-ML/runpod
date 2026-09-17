#!/usr/bin/env bash
# Phase 0 - runs FROM THE LAPTOP, drives a pod created from imgedit:v0 with PHASE0=1.
# Everything happens inside the exact image that will ship, so the pip freeze is the truth.
#
#   scripts/phase0.sh <POD_ID>
#
# Needs: tools/runpodctl.exe configured (runpodctl doctor - which also registers an SSH key),
# the pod created with --ports "8000/http,22/tcp" --env '{"PHASE0":"1",...}'.
set -euo pipefail
POD_ID="${1:?usage: phase0.sh <POD_ID>}"
cd "$(dirname "$0")/.."
RP="${RUNPODCTL:-./tools/runpodctl.exe}"
KEY="${SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}"   # made by `runpodctl doctor`; or ~/.ssh/imgedit_runpod if you added that one
[ -f "$KEY" ] || KEY="$HOME/.ssh/imgedit_runpod"

echo "== ssh info for $POD_ID =="
read -r HOST PORT < <(python scripts/pod_ssh.py "$POD_ID")
echo "ssh root@$HOST -p $PORT"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$PORT" "root@$HOST")

echo "== environment =="
"${SSH[@]}" 'nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; df -h / | tail -1; python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"'

echo "== probe (downloads 58GB of weights on first run; 5-15 min) =="
"${SSH[@]}" 'cd /app && python phase0_probe.py' "${@:2}"

echo "== copy results back =="
mkdir -p phase0
scp -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "$PORT" \
  "root@$HOST:/app/{requirements.lock.txt,phase0_report.json,phase0_out.png,phase0_out_2.png}" phase0/ || true
ls -la phase0/
echo
echo "NEXT: review phase0/phase0_out.png, then:"
echo "  cp phase0/requirements.lock.txt server/requirements.lock.txt   # freeze it"
echo "  paste phase0/phase0_report.json values into NOTES.md"
echo "  $RP pod delete $POD_ID                                        # TERMINATE THE POD"
