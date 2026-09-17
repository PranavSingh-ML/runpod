# syntax=docker/dockerfile:1.6
# Base: runpod/pytorch 1.3.1, CUDA 12.9.0, torch 2.9.1, Ubuntu 24.04, Python 3.10.
# Pinned BY DIGEST (verified on Docker Hub 2026-09-17). Never use :latest here.
FROM runpod/pytorch:1.3.1-cu1290-torch291-ubuntu2404@sha256:674cc1090eb6d753af9c99aabfb733c998463dcf720176ac9787972caece4c8f

ENV PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/workspace/hf \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_XET_HIGH_PERFORMANCE=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Zero interactive pip on the pod: everything is baked here, from the frozen lock file.
# torch stays exactly what the base image ships; the lock never lists it (or lists the
# identical version, which pip treats as already satisfied).
COPY requirements.lock.txt /app/requirements.lock.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.lock.txt \
 && python - <<'PY'
import torch, diffusers, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("diffusers", diffusers.__version__, "transformers", transformers.__version__)
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel  # fail the build early if missing
PY

COPY config.py pipeline.py app.py entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000
# RunPod: set "Expose HTTP Ports" = 8000, env API_TOKEN=<random>, HF_HOME=/workspace/hf,
# container disk >= 80GB (weights are ~58GB; 100GB is more comfortable).
CMD ["/app/entrypoint.sh"]
