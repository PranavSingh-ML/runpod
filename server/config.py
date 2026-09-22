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
# v3 primary: Qwen/Qwen-Image-2.1 (QwenImage21Pipeline, Qwen Research License = non-commercial).
#             7B DiT + Qwen3-VL 8B text encoder, native up to 2K, up to 10 reference images,
#             sampled WITHOUT guidance (true_cfg_scale 1.0) at 40 steps. No Lightning LoRA yet.
# v2 fallback: PIPELINE=qwen_edit_plus MODEL_ID=Qwen/Qwen-Image-Edit-2511 (Apache 2.0, 1 MP,
#             8-step Lightning LoRA) - or just run the :v2 image.
# Fallback 2: black-forest-labs/FLUX.1-Kontext-dev (PIPELINE=flux_kontext, gated, needs HF_TOKEN).
PIPELINE: str = _env("PIPELINE", "qwen_image_21")  # qwen_image_21 | qwen_edit_plus | flux_kontext
_DEFAULT_MODEL = {"qwen_image_21": "Qwen/Qwen-Image-2.1", "qwen_edit_plus": "Qwen/Qwen-Image-Edit-2511",
                  "flux_kontext": "black-forest-labs/FLUX.1-Kontext-dev"}
_DEFAULT_REVISION = {"Qwen/Qwen-Image-2.1": "790c92633540aa0cb11d9abf19eb46d861714758",       # verified 2026-09-21
                     "Qwen/Qwen-Image-Edit-2511": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"}  # verified 2026-09-17
MODEL_ID: str = _env("MODEL_ID", _DEFAULT_MODEL.get(PIPELINE, "Qwen/Qwen-Image-2.1"))
MODEL_REVISION: str | None = _env("MODEL_REVISION", _DEFAULT_REVISION.get(MODEL_ID))

# Lightning LoRA - only exists for the 2511 pipeline (verified 2026-09-17 in the lightx2v repo).
LORA_ENABLED: bool = _env("LORA_ENABLED", PIPELINE == "qwen_edit_plus")
LORA_REPO: str = _env("LORA_REPO", "lightx2v/Qwen-Image-Edit-2511-Lightning")
LORA_FILE: str = _env("LORA_FILE", "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")
LORA_FUSE: bool = _env("LORA_FUSE", True)  # fuse into weights before quantising (faster; required for fp8 paths)

# ---- prompt rewriter (runs on the same GPU, Qwen-Image-2.1 only) -------------
# Qwen/Qwen-Image-2.1-PE-I2I: a Qwen3.5-VL 9B fine-tune that turns "make it a party outfit" +
# the image into a precise editing directive (system_prompt.txt ships with the model).
# 18.8 GB bf16 -> ~9.5 GB with fp8 storage. Disable to save VRAM / boot time.
PE_ENABLED: bool = _env("PE_ENABLED", PIPELINE == "qwen_image_21")
PE_MODEL_ID: str = _env("PE_MODEL_ID", "Qwen/Qwen-Image-2.1-PE-I2I")
PE_REVISION: str | None = _env("PE_REVISION", "72927bc08afc99b7888ceb7d7d51a12db3700bbd")  # verified 2026-09-21
PE_QUANT: str = _env("PE_QUANT", "fp8_layerwise")   # fp8_layerwise | bf16
PE_THINKING: bool = _env("PE_THINKING", True)       # the model was trained with a <think> block; off = faster, maybe worse
PE_MAX_NEW_TOKENS: int = _env("PE_MAX_NEW_TOKENS", 4096)
PE_TEMPERATURE: float = _env("PE_TEMPERATURE", 1.0)  # model card: do_sample, temperature 1.0, top_p 0.95, top_k 20

# ---- quantisation ----------------------------------------------------------
# QUANT applies to the diffusion transformer:
# auto          -> bf16 for Qwen-Image-2.1 (7B DiT = 14 GB, fits), fp8_layerwise for 2511 (20B).
# fp8_layerwise -> diffusers layerwise casting. No extra library. Works on Ampere and Ada.
# fp8_torchao   -> torchao Float8DynamicActivationFloat8Weight on the transformer. Ada+ only,
#                  OPT-IN: torchao is not in the image (0.18.0 needs torch>=2.10 - see requirements.in).
# nf4           -> bitsandbytes 4-bit on transformer + text encoder. Smallest VRAM, slowest.
#                  Use on 24GB cards. LoRA cannot be fused on this path (it is loaded unfused).
# bf16          -> no quantisation.
QUANT: str = _env("QUANT", "auto")
QUANT_TEXT_ENCODER: bool = _env("QUANT_TEXT_ENCODER", True)  # fp8-cast / nf4 the 8B text encoder (17.5 GB -> ~9 GB)
TEXT_ENCODER_CPU_OFFLOAD: bool = _env("TEXT_ENCODER_CPU_OFFLOAD", False)  # only for 24GB cards

# ---- generation defaults ---------------------------------------------------
_IS_21 = PIPELINE == "qwen_image_21"
DEFAULT_STEPS: int = _env("DEFAULT_STEPS", 40 if _IS_21 else (8 if LORA_ENABLED else 40))
# `true_cfg_scale`. Qwen-Image-2.1 is designed to be sampled without guidance (1.0). For 2511:
# Lightning LoRAs are distilled for 1.0 (skips the negative pass => 2x faster); base wants ~4.0.
DEFAULT_GUIDANCE: float = _env("DEFAULT_GUIDANCE", 1.0 if (_IS_21 or LORA_ENABLED) else 4.0)
DEFAULT_NEGATIVE: str = _env("DEFAULT_NEGATIVE", " ")
# Output sizing.
#  qwen_image_21: the model generates at `resolution`^2 pixels, aspect of the SOURCE image, dims
#    multiples of 32. 1024 = ~1 MP draft, 2048 = native 2K final (4x the pixels, slower).
#  qwen_edit_plus: always generates at its native ~1 MP (anything else mismatches the reference
#    latents -> zoomed/cropped output); MIN/MAX_SIDE only control a post-resize.
DEFAULT_RESOLUTION: int = _env("DEFAULT_RESOLUTION", 1024)
MAX_RESOLUTION: int = _env("MAX_RESOLUTION", 2048)
MIN_RESOLUTION: int = 512
MAX_SIDE: int = _env("MAX_SIDE", 1024)   # 2511 only: default output = input aspect, long side clamped to [MIN_SIDE, MAX_SIDE]
MIN_SIDE: int = _env("MIN_SIDE", 512)
MAX_STEPS: int = _env("MAX_STEPS", 60)
SIZE_MULTIPLE: int = 32 if _IS_21 else 16  # 2.1: VAE 16 * patch 2; 2511: VAE 8 * patch 2

# ---- server ----------------------------------------------------------------
API_TOKEN: str = _env("API_TOKEN", "")
HOST: str = _env("HOST", "0.0.0.0")       # MUST be 0.0.0.0 or the RunPod proxy sees nothing
PORT: int = _env("PORT", 8000)
JOB_TTL_S: int = _env("JOB_TTL_S", 30 * 60)
JOB_CAP: int = _env("JOB_CAP", 50)
WARMUP: bool = _env("WARMUP", True)       # one throwaway edit at startup before model_loaded flips true
MOCK: bool = _env("MOCK", False)          # laptop-only: fake pipeline, no torch needed
LOG_PATH: str = _env("LOG_PATH", "/workspace/jobs.jsonl")
