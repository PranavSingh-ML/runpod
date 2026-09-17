#!/usr/bin/env bash
# Pod entrypoint. Keeps RunPod's own /start.sh (sshd, optional jupyter) alive in the
# background for emergencies, then runs the API server as the main process.
set -u
if [ -x /start.sh ]; then
  /start.sh >/tmp/runpod-start.log 2>&1 &
fi
cd /app
echo "[entrypoint] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "[entrypoint] HF_HOME=${HF_HOME:-unset}  API_TOKEN set: $([ -n "${API_TOKEN:-}" ] && echo yes || echo NO)"
# Bind 0.0.0.0 or the RunPod proxy sees nothing.
exec python -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}" --log-level info
