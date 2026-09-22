"""Model load + a single edit() function. Nothing about HTTP lives here.

Qwen-Image-2.1 (v3 default) on a 48GB card:
  transformer 7B  -> GPU in bf16 (~14 GB)           [QUANT=auto -> bf16]
  text encoder 8B -> GPU, fp8 storage / bf16 compute (~9 GB)   [QUANT_TEXT_ENCODER]
  VAE (~1.4 GB)
  + rewriter (rewriter.py) 9B, fp8 storage (~9.5 GB)
Steady state ~35 GB. Generation at `resolution`^2 pixels (1024 draft / 2048 final).

Qwen-Image-Edit-2511 (v2, PIPELINE=qwen_edit_plus) - kept as a fallback:
  1. transformer -> GPU in bf16 (40.9GB)  2. fuse Lightning LoRA  3. fp8-cast in place (~20.5GB)
  4. text encoder fp8-cast (~8.3GB). Peak ~41GB, steady ~30GB. Always generates at ~1 MP.
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


def native_dims(resolution: int, w: int, h: int) -> tuple[int, int]:
    """Qwen-Image-2.1 output dims for a `resolution`^2 pixel budget at the source aspect ratio.
    Same maths as diffusers' calculate_dimensions (round to 32) so we never fight the pipeline."""
    ratio = w / h
    ow = math.sqrt(resolution * resolution * ratio)
    oh = ow / ratio
    return max(32, round(ow / 32) * 32), max(32, round(oh / 32) * 32)


def parse_resolution(res: str | int | None) -> int:
    if res is None or res == "":
        return config.DEFAULT_RESOLUTION
    try:
        r = int(res)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"bad resolution {res!r}; expected an int like 1024 or 2048") from e
    return min(max(r, config.MIN_RESOLUTION), config.MAX_RESOLUTION)


def default_size(w: int, h: int) -> tuple[int, int]:
    """2511 path: match input aspect; long side clamped to [MIN_SIDE, MAX_SIDE]; multiples of 16."""
    long_side = max(w, h)
    target_long = min(max(long_side, config.MIN_SIDE), config.MAX_SIDE)
    scale = target_long / long_side
    ow, oh = w * scale, h * scale
    return _round_to(round(ow), config.SIZE_MULTIPLE), _round_to(round(oh), config.SIZE_MULTIPLE)


def parse_size(size: str | None, w: int, h: int, resolution: int | None = None) -> tuple[int, int]:
    """Output size for a job. Explicit `size` (WxH) wins and is applied as a post-resize;
    otherwise 2.1 uses native_dims(resolution) and 2511 uses default_size()."""
    if not size:
        if config.PIPELINE == "qwen_image_21":
            return native_dims(resolution or config.DEFAULT_RESOLUTION, w, h)
        return default_size(w, h)
    try:
        a, b = size.lower().replace("*", "x").split("x")
        ow, oh = int(a), int(b)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"bad size {size!r}; expected WxH") from e
    ow = min(max(ow, 256), 4096)
    oh = min(max(oh, 256), 4096)
    return _round_to(ow, 16), _round_to(oh, 16)


def resolve_quant(capability: tuple[int, int]) -> str:
    """Pick the quant path for the diffusion transformer. fp8 storage / bf16 compute needs no
    extra library and works on Ampere. Real fp8 compute (fp8_torchao) is opt-in: torchao's torch
    coupling broke the build once (2026-09-17)."""
    if config.QUANT != "auto":
        if config.QUANT == "fp8_torchao" and tuple(capability) < (8, 9):
            log.warning("QUANT=fp8_torchao on capability %s (<8.9): fp8 matmuls unsupported, falling back to fp8_layerwise", capability)
            return "fp8_layerwise"
        return config.QUANT
    return "bf16" if config.PIPELINE == "qwen_image_21" else "fp8_layerwise"


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def fp8_cast_module(module, skip=("norm", "embed", "lm_head", "patch")) -> None:
    """Weight-only fp8 storage with bf16 compute for any nn.Module (diffusers layerwise casting)."""
    import torch
    from diffusers.hooks import apply_layerwise_casting

    apply_layerwise_casting(module, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16,
                            skip_modules_pattern=skip)


# ----------------------------------------------------------------------------
# Mock engine: lets the whole HTTP surface + web app be tested on a laptop
# ----------------------------------------------------------------------------
class MockEngine:
    def __init__(self) -> None:
        self.info = {"gpu": "MOCK", "capability": [0, 0], "quant": "none", "model": "mock", "pipeline": config.PIPELINE}

    def load(self) -> None:
        time.sleep(1.0)

    def vram_used_gb(self) -> float:
        return 0.0

    def edit(self, images, prompt, negative, steps, guidance, seed, width, height, progress: ProgressCb,
             resolution: int | None = None):
        import random

        src = images[0].convert("RGB").resize((width, height))
        rnd = random.Random(seed)
        tint = Image.new("RGB", src.size, (rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255)))
        out = Image.blend(src, tint, 0.25)
        d = ImageDraw.Draw(out)
        d.rectangle([0, 0, width, 28], fill=(0, 0, 0))
        d.text((6, 8), f"MOCK seed={seed} steps={steps} res={resolution} :: {prompt[:70]}", fill=(255, 255, 255))
        for i in range(min(steps, 8)):
            time.sleep(0.15)
            progress((i + 1) / min(steps, 8))
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
        log.info("GPU=%s capability=%s vram=%.1fGB torch=%s cuda=%s pipeline=%s -> quant=%s",
                 name, cap, total_gb, torch.__version__, torch.version.cuda, config.PIPELINE, quant)
        self.info = {"gpu": name, "capability": list(cap), "quant": quant, "pipeline": config.PIPELINE,
                     "model": config.MODEL_ID, "torch": torch.__version__, "cuda": torch.version.cuda,
                     "vram_total_gb": round(total_gb, 1), "lora": config.LORA_FILE if config.LORA_ENABLED else None}

        t0 = time.time()
        if config.PIPELINE == "qwen_image_21":
            self._load_qwen21(quant)
        elif config.PIPELINE == "qwen_edit_plus":
            self._load_qwen(quant)
        elif config.PIPELINE == "flux_kontext":
            self._load_flux(quant)
        else:
            raise ValueError(f"unknown PIPELINE {config.PIPELINE}")
        torch.cuda.empty_cache()
        self.info["load_s"] = round(time.time() - t0, 1)
        self.info["vram_after_load_gb"] = self.vram_used_gb()
        log.info("model loaded in %.0fs, vram used %.1fGB", self.info["load_s"], self.info["vram_after_load_gb"])

    def _load_qwen21(self, quant: str) -> None:
        """Qwen-Image-2.1. Load order: everything to GPU in bf16 (33 GB, fits), then fp8-cast the
        text encoder in place. The DiT stays bf16 unless QUANT says otherwise."""
        import torch
        from diffusers import QwenImage21Pipeline

        common = {"torch_dtype": torch.bfloat16}
        if config.MODEL_REVISION:
            common["revision"] = config.MODEL_REVISION
        if quant == "nf4":
            from diffusers import PipelineQuantizationConfig

            comps = ["transformer", "text_encoder"] if config.QUANT_TEXT_ENCODER else ["transformer"]
            qcfg = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
                components_to_quantize=comps)
            pipe = QwenImage21Pipeline.from_pretrained(config.MODEL_ID, quantization_config=qcfg, **common)
        else:
            pipe = QwenImage21Pipeline.from_pretrained(config.MODEL_ID, **common)
        if config.TEXT_ENCODER_CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        if quant == "fp8_layerwise":
            pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        elif quant == "fp8_torchao":
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

            quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
        if config.QUANT_TEXT_ENCODER and quant not in ("nf4",):
            fp8_cast_module(pipe.text_encoder)
        torch.cuda.empty_cache()
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

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
                fp8_cast_module(pipe.text_encoder)
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
             seed: int, width: int, height: int, progress: ProgressCb, resolution: int | None = None) -> Image.Image:
        torch = self._torch
        gen = torch.Generator(device="cuda").manual_seed(int(seed))

        def _cb(pipe, i, t, kw):
            progress((i + 1) / max(1, steps))
            return kw

        imgs = [im.convert("RGB") for im in images]
        with torch.inference_mode():
            if config.PIPELINE == "qwen_image_21":
                # Generate at `resolution`^2 pixels with the SOURCE's aspect (the pipeline would
                # otherwise follow the LAST condition image, i.e. the reference). Dims are the
                # pipeline's own rounding (32), so no reference-latent mismatch.
                res = resolution or config.DEFAULT_RESOLUTION
                gw, gh = native_dims(res, imgs[0].width, imgs[0].height)
                use_cfg = float(guidance) > 1.0
                out = self.pipe(
                    image=imgs, prompt=prompt,
                    negative_prompt=(negative or " ") if use_cfg else None,
                    true_cfg_scale=float(guidance) if use_cfg else 1.0,
                    num_inference_steps=int(steps), width=gw, height=gh, output_resolution=res,
                    generator=gen, callback_on_step_end=_cb,
                )
            elif config.PIPELINE == "qwen_edit_plus":
                # Always let the pipeline pick its native ~1 MP size for the source's aspect
                # ratio (width/height=None). The reference image is conditioned at that same
                # size internally; generating at any other size (e.g. "match a 387x516 input")
                # mismatches the two latent grids and the model reframes -> zoomed/cropped
                # output. The requested size is applied afterwards as a plain resize.
                out = self.pipe(
                    image=imgs, prompt=prompt, negative_prompt=negative or " ",
                    true_cfg_scale=float(guidance), guidance_scale=1.0,
                    num_inference_steps=int(steps), width=None, height=None,
                    generator=gen, callback_on_step_end=_cb,
                )
            else:  # flux_kontext: single image, guidance is the embedded guidance
                out = self.pipe(
                    image=imgs[0], prompt=prompt, guidance_scale=float(guidance),
                    num_inference_steps=int(steps), width=width, height=height,
                    generator=gen, callback_on_step_end=_cb,
                )
        img = out.images[0]
        if img.mode != "RGB":
            img = img.convert("RGB")
        if (img.width, img.height) != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        return img


def make_engine():
    return MockEngine() if config.MOCK else Engine()
