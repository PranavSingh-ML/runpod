"""Phase 0: run ONCE on a stock runpod/pytorch pod, after `pip install -r requirements.in`.

Uses the exact same Engine as production (pipeline.py), so what passes here is what
ships. Produces, in the current directory:
  phase0_report.json     - versions, capability, quant path, VRAM, timings  -> copy into NOTES.md
  phase0_out.png         - the edited image (acceptance criterion)
  requirements.lock.txt  - pip freeze, minus the packages the base image owns -> replace server/requirements.lock.txt

Usage:  python phase0_probe.py [--image path.png] [--prompt "..."] [--steps 8]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_IMAGE_OWNED = ("torch==", "torchvision==", "torchaudio==", "triton==", "torchcodec==", "nvidia-", "pytorch-triton")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", default="make the sky a dramatic sunset, keep everything else unchanged")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--size", default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image, ImageDraw

    import config
    from pipeline import Engine, parse_size, resolve_quant

    report: dict = {"date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "python": platform.python_version(),
                    "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    cap = torch.cuda.get_device_capability()
    report["gpu"] = torch.cuda.get_device_name()
    report["capability"] = list(cap)
    report["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    report["quant"] = resolve_quant(cap)
    report["model"] = config.MODEL_ID
    report["model_revision"] = config.MODEL_REVISION
    report["lora"] = f"{config.LORA_REPO}/{config.LORA_FILE}" if config.LORA_ENABLED else None
    for mod in ("diffusers", "transformers", "accelerate", "peft", "torchao", "bitsandbytes", "huggingface_hub"):
        try:
            report[mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            report[mod] = f"not importable: {e.__class__.__name__}"
    try:
        report["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                                        "--format=csv,noheader"], text=True).strip()
    except Exception:  # noqa: BLE001
        pass
    print(json.dumps(report, indent=2), flush=True)

    # ---- load (this includes the weight download on a cold pod) ----
    eng = Engine()
    t0 = time.time()
    eng.load()
    report["cold_load_s_incl_download"] = round(time.time() - t0, 1)
    report["vram_idle_gb"] = eng.vram_used_gb()
    torch.cuda.reset_peak_memory_stats()
    print(f"loaded in {report['cold_load_s_incl_download']}s, idle VRAM {report['vram_idle_gb']}GB", flush=True)

    # ---- input ----
    if args.image:
        img = Image.open(args.image).convert("RGB")
    else:
        img = Image.new("RGB", (1024, 768), (70, 130, 180))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 500, 1024, 768], fill=(60, 140, 60))
        d.ellipse([400, 150, 620, 370], fill=(250, 220, 80))
        d.text((20, 20), "phase0 test image", fill=(255, 255, 255))
    w, h = parse_size(args.size, img.width, img.height)
    steps = args.steps or config.DEFAULT_STEPS

    # ---- edit 1 (includes kernel warmup) ----
    t0 = time.time()
    out = eng.edit([img], args.prompt, config.DEFAULT_NEGATIVE, steps, config.DEFAULT_GUIDANCE, 42, w, h,
                   lambda p: print(f"  progress {p:.2f}", flush=True))
    report["edit1_s_incl_warmup"] = round(time.time() - t0, 2)
    out.save("phase0_out.png")
    # ---- edit 2 (steady state) ----
    t0 = time.time()
    out2 = eng.edit([img], args.prompt, config.DEFAULT_NEGATIVE, steps, config.DEFAULT_GUIDANCE, 43, w, h, lambda p: None)
    report["edit2_s_steady"] = round(time.time() - t0, 2)
    out2.save("phase0_out_2.png")
    report["edit_steps"] = steps
    report["edit_size"] = f"{w}x{h}"
    report["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(json.dumps(report, indent=2), flush=True)

    with open("phase0_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ---- lock file ----
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    kept = [ln for ln in freeze if ln and not ln.startswith(BASE_IMAGE_OWNED) and " @ file://" not in ln]
    header = [
        f"# Frozen by phase0_probe.py on {report['date']}",
        f"# base image: runpod/pytorch (torch {report['torch']} cuda {report['cuda']}), GPU {report['gpu']} cap {cap}",
        "# torch/torchvision/torchaudio/triton/nvidia-* are owned by the base image and intentionally omitted.",
    ]
    with open("requirements.lock.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(header + kept) + "\n")
    print("\nWROTE phase0_report.json, phase0_out.png, requirements.lock.txt")
    print("Copy requirements.lock.txt -> server/requirements.lock.txt and phase0_report.json -> NOTES.md. Then TERMINATE the pod.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
