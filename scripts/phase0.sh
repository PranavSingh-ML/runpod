#!/usr/bin/env bash
# Phase 0 - run ON THE POD (stock runpod/pytorch template, A40 48GB, 80-100GB container disk).
#
#   1. From the laptop, copy server/ and scripts/ up (SSH details are on the pod's Connect tab):
#        scp -P <PORT> -r server scripts root@<IP>:/workspace/imgedit/
#   2. On the pod:
#        bash /workspace/imgedit/scripts/phase0.sh
#   3. Copy the results back down:
#        scp -P <PORT> root@<IP>:/workspace/imgedit/server/{requirements.lock.txt,phase0_report.json,phase0_out.png} .
#      then move requirements.lock.txt -> server/requirements.lock.txt (overwrite the provisional one).
#   4. TERMINATE THE POD.
#
# Budget: ~15-30 min of pod time (weight download ~58GB dominates).
set -euo pipefail
cd "$(dirname "$0")/../server"

export HF_HOME="${HF_HOME:-/workspace/hf}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=0   # base image sets 1 but hf_transfer may not be installed; hf_xet is used instead

echo "== base image / torch =="
cat /etc/os-release | head -2
python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability(), torch.cuda.get_device_name())
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
df -h /workspace | tail -1

echo "== pip install (top-level pins) =="
python -m pip install --no-cache-dir -r requirements.in

echo "== probe =="
python phase0_probe.py "$@"
