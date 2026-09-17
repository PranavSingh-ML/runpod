# NOTES — verified values, measurements, spend

Legend: **VERIFIED** = checked against the live source on the date given, from the laptop.
**MEASURED** = recorded from a real pod run. **TODO(pod)** = cannot be known until Phase 0/1
runs on a GPU; fill in from `phase0_report.json` / `smoke_test.sh` output.

## Environment pins (VERIFIED 2026-09-17)

| Item | Value | Source |
|---|---|---|
| Base image | `nvidia/cuda:12.9.0-base-ubuntu24.04@sha256:48e21b10467354655f5073c05eebdeaac9818c6b40d70f334f7ad2df000463d8` (109 MB) | Docker Hub API |
| Why not runpod/pytorch | 16.3 GB compressed — too big for a free GitHub runner and slow for RunPod to pull each cold boot. Our image is ~5 GB. | Docker Hub API |
| Python / torch | Ubuntu 24.04 python3 = 3.12; `torch==2.9.1+cu129`, `torchvision==0.24.1+cu129` from download.pytorch.org/whl/cu129 (cp312 wheels exist) | PyTorch index |
| Build | GitHub Actions → `ghcr.io/<owner>/imgedit:<tag>` (no local Docker: this laptop has VT-x disabled in firmware) | `.github/workflows/build-image.yml` |
| diffusers | **`0.40.0` from PyPI (released 2026-08-20)** — no git SHA needed | PyPI |
| diffusers tag commit | `d035dcd7cc7c88e0a154609b62887d50bba9fdc2` (v0.40.0) | `git ls-remote` |
| Why no SHA pin | `QwenImageEditPlusPipeline` + `zero_cond_t` (needed by 2511, `_diffusers_version: 0.36.0.dev0` in the model config) are in the 0.40.0 release. A released wheel beats a git SHA for reproducibility. | pipeline file present at tag v0.40.0 |
| transformers | 5.17.0 (2026-09-09) | PyPI |
| accelerate | 1.15.0 | PyPI |
| torchao | 0.18.0 (only used on sm_89+) | PyPI |
| bitsandbytes | 0.50.2 (only used if `QUANT=nf4`) | PyPI |
| fastapi / uvicorn | 0.141.1 / 0.53.0 | PyPI |
| Model revision | `Qwen/Qwen-Image-Edit-2511` @ `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`, Apache 2.0, not gated | HF API |
| Model size | **57.7 GB bf16** (transformer 40.9, text encoder 16.6, vae 0.25) — the spec's "~25 GB" estimate was wrong | HF API |
| Lightning LoRA | `lightx2v/Qwen-Image-Edit-2511-Lightning` / `Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors` (repo sha `d74eba14…`, 4- and 8-step, bf16 + fp32) | HF API |
| Lightning inference settings | steps 8, `true_cfg_scale` 1.0, negative `" "`, scheduler exponential shift with `base_shift = max_shift = ln 3`, `shift_terminal = None` | ModelTC/Qwen-Image-Lightning `generate_with_diffusers.py` |
| Pipeline call | `QwenImageEditPlusPipeline(image=[...], prompt, negative_prompt, true_cfg_scale, guidance_scale=1.0, num_inference_steps, width, height, generator, callback_on_step_end)`; output dims must be multiples of 16; default target area 1024² | diffusers v0.40.0 source |

Lock file status: `server/requirements.lock.txt` is **PROVISIONAL** (top-level pins only).
Phase 0 runs `phase0_probe.py` *inside* the `v0` image on the GPU and writes the real freeze
(`phase0/requirements.lock.txt`) → copy over, tag `v1`. → TODO(pod)

## Hardware / pricing (VERIFIED 2026-09-17, runpod.io/pricing — these move)

| GPU | Secure | Community | Hours on $13 (Secure) |
|---|---|---|---|
| A40 48GB (Ampere, sm_86) | **$0.49/hr** | $0.35/hr | ~26 h (~37 h Community) |
| RTX A6000 48GB (sm_86) | $0.53/hr | $0.33/hr | ~24 h |
| L40S 48GB (Ada, sm_89) | $1.09/hr | $0.79/hr | ~12 h |
| RTX 6000 Ada 48GB (sm_89) | $0.84/hr | $0.74/hr | ~15 h |
| RTX 4090 24GB (sm_89) | $0.74/hr | $0.34/hr | not recommended (needs nf4 + offload) |

Container disk: $0.10/GB/month → 100 GB ≈ $0.014/hr while the pod runs. Network volume
$0.07/GB/month (skipped by design). The spec's A40 estimate ($0.44) is now $0.49.

Cold boot estimate: 57.7 GB download at 200–500 MB/s = 2–5 min ≈ **$0.02–0.04** plus model
load + fp8 cast + warmup (~2–3 min) → budget ~$0.06 per cold boot at A40 rates.

## Quantisation decision

Decided at boot by `torch.cuda.get_device_capability()` (`server/pipeline.py:resolve_quant`):

- **sm < 8.9 (A40/A6000):** `fp8_layerwise` — diffusers layerwise casting. Weights stored
  fp8_e4m3fn, upcast to bf16 per layer for the matmul. Needs no extra library (the point,
  given the version-hell history). Transformer ~20.5 GB + text encoder ~8.3 GB + VAE +
  activations ≈ **~30 GB steady, ~41 GB peak during load** (transformer lands in bf16 first,
  LoRA is fused, then cast). Fits 48 GB with no CPU offload.
- **sm ≥ 8.9 (L40S / 6000 Ada / 4090):** `fp8_torchao` — `Float8DynamicActivationFloat8WeightConfig`
  on the transformer (real fp8 matmuls), text encoder fp8 layerwise. Untested until Phase 0
  runs on Ada; fall back to `QUANT=fp8_layerwise` env if torchao errors.
- **24 GB cards:** `QUANT=nf4 TEXT_ENCODER_CPU_OFFLOAD=1`. Not the plan.
- **bf16 unquantised needs ~60 GB** → not possible on any 48 GB card. (The spec's table implies
  it might fit; it does not: 57.7 GB of weights.)

## Measurements — TODO(pod)

Fill from `phase0_report.json` (Phase 0) and `scripts/smoke_test.sh` (Phase 1).

| Metric | Value |
|---|---|
| GPU / capability actually seen | |
| torch / CUDA / cudnn on pod | |
| quant path chosen | |
| cold load incl. 57.7 GB download (s) | |
| VRAM idle after load (GB) | |
| VRAM peak during an edit (GB) | |
| edit #1 incl. warmup, 8 steps, ~1024px (s) | |
| edit #2 steady state, 8 steps, ~1024px (s) | |
| smoke_test cold-boot wall time (s) | |
| smoke_test `server_elapsed` (s) | |
| `v0` image digest (provisional lock) | |
| `v1` image digest (frozen lock) | |

## Spend log

| Date | What | GPU | Hours | $ |
|---|---|---|---|---|
| | Phase 0 | | | |
| | Phase 1 smoke test | | | |
| | | | | **running total: $0.00 / $13** |

## Laptop-side verification (done 2026-09-17, no GPU)

- `server/app.py` in `MOCK=1` mode: `/health`, 401 without token, 403 with wrong token, async
  `/edit` → poll → PNG, JSON job log, warmup gate — all pass via `scripts/smoke_test.sh`.
- Web app against the mock pod: upload → 3 chained edits → fork from edit #1 with fixed seed
  42 → all 4 PNGs in `data/images/`, 5 rows in `data/history.sqlite` with correct
  `parent_id`s, explicit `size=512x384` honoured. UI checked in Chrome (thread, branches,
  active-source pill, composer Enter-to-send, connection pill, cost meter, settings panel).
- `npm run typecheck` and `npm run build` clean.

## Deviations from spec.md (told the user)

1. diffusers is pinned to a PyPI release, not a git SHA — the needed pipeline is released.
2. Weights are 57.7 GB, not ~25 GB. Recommend a 100 GB container disk over 80 GB.
3. A40 Secure is $0.49/hr, not $0.44. Community A40 is $0.35/hr if latency tolerance allows.
4. `guidance` on the API maps to `true_cfg_scale` (the model has no guidance embedding);
   default 1.0 with the Lightning LoRA, which also halves compute by skipping the negative pass.
5. Output size default is match-input-aspect with long side clamped to `[MIN_SIDE=512, MAX_SIDE=1024]`
   (a floor was added because the model is trained at ~1 MP).
6. Rewriter uses `claude-haiku-4-5` (the spec asked for a cheap text model); `REWRITE_MODEL` env overrides.
7. Phase 0 runs inside the `v0` image (built from top-level pins) instead of on a stock
   `runpod/pytorch` template. Same purpose — one throwaway GPU session produces the frozen
   lock — but the freeze now comes from the exact image that ships, and no local Docker is needed.
