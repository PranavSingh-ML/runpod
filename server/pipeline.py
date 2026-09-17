"""Model load + a single edit() function. Nothing about HTTP lives here.

Load order on the fp8 paths matters for a 48GB card (weights are 57.7GB in bf16):
  1. transformer -> GPU in bf16 (40.9GB)
  2. fuse the Lightning LoRA into it while it is still bf16
  3. cast it to fp8 storage in place (-> ~20.5GB)
  4. then bring the 16.6GB text encoder over and fp8-cast it too (-> ~8.3GB)
Peak VRAM is step 1, ~41GB. Steady state ~30GB.
"""
from __future__ import annotations

import io
import logging
import math
import time
from typing import Callable

from PIL import Image, ImageDraw

import config

log = logging.getLogger("pipeline")

ProgressCb = Callable[[float], None]


# ----------------------------------------------------------------------------
# helpers shared by real and mock engines
# ----------------------------------------------------------------------------
def _round_to(v: int, m: int) -> int:
    return max(m, (int(v) // m) * m)


def default_size(w: int, h: int) -> tuple[int, int]:
    """Match input aspect; long side clamped to [MIN_SIDE, MAX_SIDE]; multiples of 16."""
    long_side = max(w, h)
    target_long = min(max(long_side, config.MIN_SIDE), config.MAX_SIDE)
    scale = target_long / long_side
    ow, oh = w * scale, h * scale
    return _round_to(round(ow), config.SIZE_MULTIPLE), _round_to(round(oh), config.SIZE_MULTIPLE)


def parse_size(size: str | None, w: int, h: int) -> tuple[int, int]:
    if not size:
        return default_size(w, h)
    try:
        a, b = size.lower().replace("*", "x").split("x")
        ow, oh = int(a), int(b)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"bad size {size!r}; expected WxH") from e
    ow = min(max(ow, 256), 2048)
    oh = min(max(oh, 256), 2048)
    return _round_to(ow, config.SIZE_MULTIPLE), _round_to(oh, config.SIZE_MULTIPLE)


def resolve_quant(capability: tuple[int, int]) -> str:
    """Pick the quant path from compute capability, never from the card name."""
    if config.QUANT != "auto":
        return config.QUANT
    return "fp8_torchao" if tuple(capability) >= (8, 9) else "fp8_layerwise"


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Mock engine: lets the whole HTTP surface + web app be tested on a laptop
# ----------------------------------------------------------------------------
class MockEngine:
    def __init__(self) -> None:
        self.info = {"gpu": "MOCK", "capability": [0, 0], "quant": "none", "model": "mock"}

    def load(self) -> None:
        time.sleep(1.0)

    def vram_used_gb(self) -> float:
        return 0.0

    def edit(self, images, prompt, negative, steps, guidance, seed, width, height, progress: ProgressCb):
        import random

        src = images[0].convert("RGB").resize((width, height))
        rnd = random.Random(seed)
        tint = Image.new("RGB", src.size, (rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255)))
        out = Image.blend(src, tint, 0.25)
        d = ImageDraw.Draw(out)
        d.rectangle([0, 0, width, 28], fill=(0, 0, 0))
        d.text((6, 8), f"MOCK seed={seed} steps={steps} :: {prompt[:80]}", fill=(255, 255, 255))
        for i in range(steps):
            time.sleep(0.15)
            progress((i + 1) / steps)
        return out


# ----------------------------------------------------------------------------
# Real engine
# ----------------------------------------------------------------------------
class Engine:
    def __init__(self) -> None:
        self.pipe = None
        self.info: dict = {}
        self._torch = None

    # -- loading -------------------------------------------------------------
    def load(self) -> None:
        import torch

        self._torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name()
        quant = resolve_quant(cap)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("GPU=%s capability=%s vram=%.1fGB torch=%s cuda=%s -> quant=%s",
                 name, cap, total_gb, torch.__version__, torch.version.cuda, quant)
        self.info = {"gpu": name, "capability": list(cap), "quant": quant,
                     "model": config.MODEL_ID, "torch": torch.__version__, "cuda": torch.version.cuda,
                     "vram_total_gb": round(total_gb, 1), "lora": config.LORA_FILE if config.LORA_ENABLED else None}

        t0 = time.time()
        if config.PIPELINE == "qwen_edit_plus":
            self._load_qwen(quant)
        elif config.PIPELINE == "flux_kontext":
            self._load_flux(quant)
        else:
            raise ValueError(f"unknown PIPELINE {config.PIPELINE}")
        torch.cuda.empty_cache()
        self.info["load_s"] = round(time.time() - t0, 1)
        self.info["vram_after_load_gb"] = self.vram_used_gb()
        log.info("model loaded in %.0fs, vram used %.1fGB", self.info["load_s"], self.info["vram_after_load_gb"])

    def _load_qwen(self, quant: str) -> None:
        import torch
        from diffusers import (FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline,
                               QwenImageTransformer2DModel)

        dtype = torch.bfloat16
        common = {"torch_dtype": dtype}
        if config.MODEL_REVISION:
            common["revision"] = config.MODEL_REVISION

        if quant == "nf4":
            from diffusers import PipelineQuantizationConfig

            comps = ["transformer", "text_encoder"] if config.QUANT_TEXT_ENCODER else ["transformer"]
            qcfg = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                              "bnb_4bit_compute_dtype": dtype},
                components_to_quantize=comps,
            )
            pipe = QwenImageEditPlusPipeline.from_pretrained(config.MODEL_ID, quantization_config=qcfg, **common)
            if config.LORA_ENABLED:
                pipe.load_lora_weights(config.LORA_REPO, weight_name=config.LORA_FILE)  # unfused on nf4
            if config.TEXT_ENCODER_CPU_OFFLOAD:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
        elif quant in ("fp8_layerwise", "fp8_torchao", "bf16"):
            # 1. transformer straight to GPU (bf16, ~41GB)
            transformer = QwenImageTransformer2DModel.from_pretrained(
                config.MODEL_ID, subfolder="transformer", device_map="cuda", **common)
            pipe = QwenImageEditPlusPipeline.from_pretrained(config.MODEL_ID, transformer=transformer, **common)
            # 2. LoRA while still bf16
            if config.LORA_ENABLED:
                pipe.load_lora_weights(config.LORA_REPO, weight_name=config.LORA_FILE)
                if config.LORA_FUSE:
                    pipe.fuse_lora()
                    pipe.unload_lora_weights()
            # 3. shrink the transformer in place
            if quant == "fp8_layerwise":
                pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype)
            elif quant == "fp8_torchao":
                from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

                quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
            torch.cuda.empty_cache()
            # 4. everything else over, then fp8-cast the 8B text encoder (weight-only, any arch)
            if config.TEXT_ENCODER_CPU_OFFLOAD:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
            if config.QUANT_TEXT_ENCODER and quant != "bf16":
                from diffusers.hooks import apply_layerwise_casting

                apply_layerwise_casting(pipe.text_encoder, storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype,
                                        skip_modules_pattern=("norm", "embed", "lm_head", "patch"))
        else:
            raise ValueError(f"unknown QUANT {quant}")

        if config.LORA_ENABLED:
            # Lightning LoRAs are trained against this exponential-shift schedule.
            # Source: ModelTC/Qwen-Image-Lightning generate_with_diffusers.py (verified 2026-09-17).
            sched_cfg = dict(pipe.scheduler.config)
            sched_cfg.update({"base_shift": math.log(3), "max_shift": math.log(3), "shift": 1.0,
                              "shift_terminal": None, "use_dynamic_shifting": True,
                              "time_shift_type": "exponential"})
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(sched_cfg)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def _load_flux(self, quant: str) -> None:
        """Second fallback. Gated repo: HF_TOKEN must be set. Non-commercial license."""
        import torch
        from diffusers import FluxKontextPipeline

        dtype = torch.bfloat16
        if quant == "nf4":
            from diffusers import PipelineQuantizationConfig

            qcfg = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": dtype},
                components_to_quantize=["transformer", "text_encoder_2"])
            pipe = FluxKontextPipeline.from_pretrained(config.MODEL_ID, torch_dtype=dtype, quantization_config=qcfg)
        else:
            pipe = FluxKontextPipeline.from_pretrained(config.MODEL_ID, torch_dtype=dtype)
            if quant in ("fp8_layerwise", "fp8_torchao"):
                pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype)
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    # -- inference -----------------------------------------------------------
    def vram_used_gb(self) -> float:
        if self._torch is None or not self._torch.cuda.is_available():
            return 0.0
        return round(self._torch.cuda.memory_allocated() / 1e9, 2)

    def edit(self, images: list[Image.Image], prompt: str, negative: str, steps: int, guidance: float,
             seed: int, width: int, height: int, progress: ProgressCb) -> Image.Image:
        torch = self._torch
        gen = torch.Generator(device="cuda").manual_seed(int(seed))

        def _cb(pipe, i, t, kw):
            progress((i + 1) / max(1, steps))
            return kw

        imgs = [im.convert("RGB") for im in images]
        with torch.inference_mode():
            if config.PIPELINE == "qwen_edit_plus":
                out = self.pipe(
                    image=imgs, prompt=prompt, negative_prompt=negative or " ",
                    true_cfg_scale=float(guidance), guidance_scale=1.0,
                    num_inference_steps=int(steps), width=width, height=height,
                    generator=gen, callback_on_step_end=_cb,
                )
            else:  # flux_kontext: single image, guidance is the embedded guidance
                out = self.pipe(
                    image=imgs[0], prompt=prompt, guidance_scale=float(guidance),
                    num_inference_steps=int(steps), width=width, height=height,
                    generator=gen, callback_on_step_end=_cb,
                )
        return out.images[0]


def make_engine():
    return MockEngine() if config.MOCK else Engine()
