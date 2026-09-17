"""All tunables for the pod-side server. Everything can be overridden by env var.

Switching models = change MODEL_ID / LORA_* here (or via env). No code changes.
"""
from __future__ import annotations

import os


def _env(name: str, default):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    if isinstance(default, bool):
        return v.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v


# ---- model -----------------------------------------------------------------
# Primary: Qwen/Qwen-Image-Edit-2511 (Apache 2.0, QwenImageEditPlusPipeline).
# Fallback 1: Qwen/Qwen-Image-Edit-2509 (same pipeline class; set LORA_REPO to
#             lightx2v/Qwen-Image-Edit-2509-Lightning and LORA_FILE accordingly).
# Fallback 2: black-forest-labs/FLUX.1-Kontext-dev (PIPELINE=flux_kontext, gated,
#             needs HF_TOKEN, non-commercial license). See pipeline.py.
MODEL_ID: str = _env("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
MODEL_REVISION: str | None = _env("MODEL_REVISION", "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9")  # verified 2026-09-17
PIPELINE: str = _env("PIPELINE", "qwen_edit_plus")  # qwen_edit_plus | flux_kontext

# 8-step Lightning LoRA (verified to exist 2026-09-17 in the lightx2v repo).
LORA_ENABLED: bool = _env("LORA_ENABLED", True)
LORA_REPO: str = _env("LORA_REPO", "lightx2v/Qwen-Image-Edit-2511-Lightning")
LORA_FILE: str = _env("LORA_FILE", "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")
LORA_FUSE: bool = _env("LORA_FUSE", True)  # fuse into weights before quantising (faster; required for fp8 paths)

# ---- quantisation ----------------------------------------------------------
# auto          -> fp8_layerwise on every GPU (capability is still logged; see pipeline.resolve_quant)
# fp8_layerwise -> diffusers layerwise casting. No extra library. Works on Ampere and Ada.
# fp8_torchao   -> torchao Float8DynamicActivationFloat8Weight on the transformer. Ada+ only,
#                  OPT-IN: torchao is not in the image (0.18.0 needs torch>=2.10 - see requirements.in).
# nf4           -> bitsandbytes 4-bit on transformer + text encoder. Smallest VRAM, slowest.
#                  Use on 24GB cards. LoRA cannot be fused on this path (it is loaded unfused).
# bf16          -> no quantisation. Needs ~60GB VRAM. Not for 48GB cards.
QUANT: str = _env("QUANT", "auto")
QUANT_TEXT_ENCODER: bool = _env("QUANT_TEXT_ENCODER", True)  # also fp8-cast / nf4 the 8B text encoder
TEXT_ENCODER_CPU_OFFLOAD: bool = _env("TEXT_ENCODER_CPU_OFFLOAD", False)  # only for 24GB cards

# ---- generation defaults ---------------------------------------------------
DEFAULT_STEPS: int = _env("DEFAULT_STEPS", 8 if LORA_ENABLED else 40)
# For Qwen this is `true_cfg_scale`. Lightning LoRAs are distilled for CFG=1.0
# (which also skips the negative pass => 2x faster). Base model wants ~4.0.
DEFAULT_GUIDANCE: float = _env("DEFAULT_GUIDANCE", 1.0 if LORA_ENABLED else 4.0)
DEFAULT_NEGATIVE: str = _env("DEFAULT_NEGATIVE", " ")
MAX_SIDE: int = _env("MAX_SIDE", 1024)   # long side of output is clamped to this
MIN_SIDE: int = _env("MIN_SIDE", 512)    # and never below this (model is trained ~1MP)
MAX_STEPS: int = _env("MAX_STEPS", 50)
SIZE_MULTIPLE: int = 16                   # QwenImage VAE: 8 * patch 2

# ---- server ----------------------------------------------------------------
API_TOKEN: str = _env("API_TOKEN", "")
HOST: str = _env("HOST", "0.0.0.0")       # MUST be 0.0.0.0 or the RunPod proxy sees nothing
PORT: int = _env("PORT", 8000)
JOB_TTL_S: int = _env("JOB_TTL_S", 30 * 60)
JOB_CAP: int = _env("JOB_CAP", 50)
WARMUP: bool = _env("WARMUP", True)       # one throwaway edit at startup before model_loaded flips true
MOCK: bool = _env("MOCK", False)          # laptop-only: fake pipeline, no torch needed
LOG_PATH: str = _env("LOG_PATH", "/workspace/jobs.jsonl")
