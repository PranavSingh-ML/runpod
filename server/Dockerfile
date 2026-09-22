# syntax=docker/dockerfile:1.6
# Slim base: CUDA 12.9 driver shim only (109 MB). torch's pip wheels bundle their own CUDA
# libraries, so the image needs no CUDA toolkit. Pinned BY DIGEST (Docker Hub, 2026-09-17).
# Result is ~5 GB compressed vs 16 GB for runpod/pytorch - faster cold-boot pulls on RunPod.
FROM nvidia/cuda:12.9.0-base-ubuntu24.04@sha256:48e21b10467354655f5073c05eebdeaac9818c6b40d70f334f7ad2df000463d8

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/workspace/hf \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_XET_HIGH_PERFORMANCE=1 \
    TOKENIZERS_PARALLELISM=false \
    PATH=/opt/venv/bin:$PATH

# Python 3.12 (Ubuntu 24.04 default) + sshd for Phase 0 / emergencies. Nothing else.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip openssh-server ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /run/sshd /workspace /app \
 && python3 -m venv /opt/venv \
 && pip install --upgrade pip

# torch layer: exact wheel from the cu129 index. Changes rarely, so it caches well.
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
RUN pip install --index-url https://download.pytorch.org/whl/cu129 \
      "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

WORKDIR /app

# Zero interactive pip on the pod: everything is baked here, from the frozen lock file.
# (git is installed above because diffusers is pinned to a git commit until 2.1 ships in a release.)
# The lock never lists torch/torchvision/nvidia-*/triton (phase0_probe strips them) -
# those come from the layer above.
COPY requirements.lock.txt /app/requirements.lock.txt
RUN pip install -r /app/requirements.lock.txt \
 && python - <<'PY'
import torch, diffusers, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("diffusers", diffusers.__version__, "transformers", transformers.__version__)
from diffusers import QwenImage21Pipeline, QwenImageEditPlusPipeline              # v3 + v2 pipelines; fail the build early if missing
from diffusers.hooks import apply_layerwise_casting                               # fp8_layerwise path
from transformers import Qwen3VLForConditionalGeneration, Qwen3_5ForConditionalGeneration  # 2.1 text encoder + PE-I2I rewriter
import peft; print("peft", peft.__version__)
try:
    import bitsandbytes; print("bitsandbytes", bitsandbytes.__version__, "(optional, nf4 path)")
except Exception as e:  # optional path - warn, don't fail the build
    print("WARNING bitsandbytes import failed (nf4 path unavailable):", e)
PY

COPY config.py pipeline.py rewriter.py app.py phase0_probe.py entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000 22
# RunPod: Expose HTTP Ports 8000, TCP 22; env API_TOKEN=<random>; container disk 100 GB.
# PHASE0=1 -> sshd only (no server), for running phase0_probe.py inside this exact image.
CMD ["/app/entrypoint.sh"]
