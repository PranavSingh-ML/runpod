"""Prompt rewriter: Qwen/Qwen-Image-2.1-PE-I2I on the same GPU as the editor.

Input: the source image (+ optional reference) and the user's casual instruction.
Output: {"rewritten_prompt": ..., "wh_ratio": "", "ratio_follow": "<image1>", "thinking": ...}
The model card's recipe (verified 2026-09-21): chat template with the shipped system_prompt.txt,
enable_thinking=True, do_sample temperature 1.0 top_p 0.95 top_k 20, JSON after </think>.

Runs on the single GPU worker thread (app.py) so it never overlaps a diffusion step.
"""
from __future__ import annotations

import json
import logging
import re
import time

from PIL import Image

import config

log = logging.getLogger("rewriter")

MAX_SIDE = 1536  # the VL encoder does not need more; keeps the vision token count sane


def _shrink(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    s = MAX_SIDE / max(img.size)
    if s < 1:
        img = img.resize((max(32, round(img.width * s)), max(32, round(img.height * s))), Image.LANCZOS)
    return img


def _parse(gen: str) -> dict:
    thinking, _, answer = gen.partition("</think>")
    if not _:  # no think block at all
        thinking, answer = "", gen
    answer = answer.strip()
    m = re.search(r"\{.*\}", answer, re.S)
    if not m:
        raise ValueError(f"rewriter returned no JSON: {answer[:300]!r}")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict) or not obj.get("rewritten_prompt"):
        raise ValueError(f"rewriter JSON has no rewritten_prompt: {answer[:300]!r}")
    return {"rewritten_prompt": str(obj["rewritten_prompt"]).strip(),
            "wh_ratio": str(obj.get("wh_ratio") or ""), "ratio_follow": str(obj.get("ratio_follow") or ""),
            "thinking": thinking.replace("<think>", "").strip()[:4000]}


class MockRewriter:
    info = {"model": "mock-rewriter"}

    def load(self) -> None:
        pass

    def rewrite(self, images: list[Image.Image], instruction: str) -> dict:
        time.sleep(0.5)
        w, h = images[0].size
        return {"rewritten_prompt": f"[MOCK rewrite of {w}x{h} image] {instruction.strip()} - keep the subject's "
                                    "identity, pose, framing and lighting unchanged.",
                "wh_ratio": "", "ratio_follow": "<image1>", "thinking": "mock"}


class Rewriter:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.system_prompt = ""
        self.info: dict = {"model": config.PE_MODEL_ID}

    def load(self) -> None:
        import huggingface_hub
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        t0 = time.time()
        kw = {"revision": config.PE_REVISION} if config.PE_REVISION else {}
        self.processor = AutoProcessor.from_pretrained(config.PE_MODEL_ID, **kw)
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.PE_MODEL_ID, dtype=torch.bfloat16, device_map="cuda", **kw).eval()
        if config.PE_QUANT == "fp8_layerwise":
            from pipeline import fp8_cast_module

            # "conv" must be skipped: Qwen3.5's linear-attention path calls causal_conv1d_fn, a free
            # function that reads conv1d.weight directly, so the layerwise upcast hook never fires and
            # F.conv1d gets an fp8 weight -> NotImplementedError (A6000, 2026-09-22).
            fp8_cast_module(self.model, skip=("norm", "embed", "lm_head", "patch", "conv"))
        path = huggingface_hub.hf_hub_download(config.PE_MODEL_ID, "system_prompt.txt", **kw)
        with open(path, encoding="utf-8") as f:
            self.system_prompt = f.read().strip()
        torch.cuda.empty_cache()
        self.info.update({"quant": config.PE_QUANT, "thinking": config.PE_THINKING, "load_s": round(time.time() - t0, 1)})
        log.info("rewriter loaded in %.0fs (%s, quant=%s)", self.info["load_s"], config.PE_MODEL_ID, config.PE_QUANT)

    def rewrite(self, images: list[Image.Image], instruction: str) -> dict:
        import torch

        content = [{"type": "image", "image": _shrink(im)} for im in images]
        content.append({"type": "text", "text": instruction.strip()})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            enable_thinking=config.PE_THINKING,
        ).to(self.model.device)
        sampling = ({"do_sample": True, "temperature": config.PE_TEMPERATURE, "top_p": 0.95, "top_k": 20}
                    if config.PE_TEMPERATURE > 0 else {"do_sample": False})
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=int(config.PE_MAX_NEW_TOKENS), **sampling)
        gen = self.processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return _parse(gen)


def make_rewriter():
    if not config.PE_ENABLED:
        return None
    return MockRewriter() if config.MOCK else Rewriter()
