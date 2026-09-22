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
| torchao | **removed from the image.** 0.18.0 imports `torch.nn.functional.ScalingType` (torch ≥ 2.10) and transformers 5.17 imports torchao eagerly when installed → diffusers import failed at build time (Actions run #1, 2026-09-17). `fp8_torchao` is now opt-in. | build log |
| bitsandbytes | 0.50.2 (only used if `QUANT=nf4`) | PyPI |
| fastapi / uvicorn | 0.141.1 / 0.53.0 | PyPI |
| Model revision | `Qwen/Qwen-Image-Edit-2511` @ `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`, Apache 2.0, not gated | HF API |
| Model size | **57.7 GB bf16** (transformer 40.9, text encoder 16.6, vae 0.25) — the spec's "~25 GB" estimate was wrong | HF API |
| Lightning LoRA | `lightx2v/Qwen-Image-Edit-2511-Lightning` / `Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors` (repo sha `d74eba14…`, 4- and 8-step, bf16 + fp32) | HF API |
| Lightning inference settings | steps 8, `true_cfg_scale` 1.0, negative `" "`, scheduler exponential shift with `base_shift = max_shift = ln 3`, `shift_terminal = None` | ModelTC/Qwen-Image-Lightning `generate_with_diffusers.py` |
| Pipeline call | `QwenImageEditPlusPipeline(image=[...], prompt, negative_prompt, true_cfg_scale, guidance_scale=1.0, num_inference_steps, width, height, generator, callback_on_step_end)`; output dims must be multiples of 16; default target area 1024² | diffusers v0.40.0 source |

Lock file status: **FROZEN 2026-09-17 19:38 UTC** — `server/requirements.lock.txt` is the `pip freeze`
taken inside the `v0` image on the Phase 0 pod (59 pinned packages; torch layer excluded). `v1` is
built from it. Notable resolved transitive pins: peft 0.21.0, safetensors 0.8.0, huggingface_hub 1.32.0,
tokenizers 0.23.2, numpy 2.5.2, pydantic 2.13.5.

## v3 pins (VERIFIED 2026-09-21, from the laptop — NOT yet run on a GPU)

| Item | Value | Source |
|---|---|---|
| Model | `Qwen/Qwen-Image-2.1` @ `790c92633540aa0cb11d9abf19eb46d861714758`, Qwen Research License (non-commercial), not gated. 7B DiT + Qwen3-VL 8B text encoder + 16× VAE ≈ 33 GB bf16 | HF API |
| Rewriter | `Qwen/Qwen-Image-2.1-PE-I2I` @ `72927bc08afc99b7888ceb7d7d51a12db3700bbd` — Qwen3.5-VL 9B fine-tune, 18.8 GB bf16, ships `system_prompt.txt`; recipe: chat template, `enable_thinking=True`, do_sample T=1.0 top_p 0.95 top_k 20, JSON `{rewritten_prompt, wh_ratio, ratio_follow}` after `</think>` | model card |
| diffusers | **git pin** `cc8644b447d8f11074d3df06d0ee0e3e7c91bf75` (main, 2026-09-21). `QwenImage21Pipeline` landed in #14804 on 2026-09-18 and is in no release (latest 0.40.0). Dockerfile installs `git` for this. Move to a PyPI pin at the next release. | GitHub API |
| Pipeline call | `QwenImage21Pipeline(image=[...], prompt, negative_prompt?, true_cfg_scale=1.0, num_inference_steps=40, width, height, output_resolution, generator, callback_on_step_end)`; dims = `calculate_dimensions(res², aspect)` rounded to 32; condition images are resized to the same budget | pipeline source at the pinned SHA |
| transformers | 5.17.0 unchanged: has `Qwen3VLForConditionalGeneration` (text encoder) and `Qwen3_5ForConditionalGeneration` (rewriter; pure-torch fallback for the linear-attention path, no compiled kernels needed) | source at v5.17.0 |
| Rewriter fp8 (MEASURED 2026-09-22, A6000) | The pure-torch fallback is **not** fp8-safe: `causal_conv1d_fn` (`modeling_qwen3_5.py:280`) reads `conv1d.weight` directly, so layerwise casting's upcast hook never fires and `F.conv1d` raises `NotImplementedError: "conv_depthwise2d_forward_cuda" not implemented for 'Float8_e4m3fn'` at the rewriter warmup. Fixed by adding `conv` to the skip list in `rewriter.py`; `PE_QUANT=bf16` (18.8 GB) is the no-rebuild workaround. Editor unaffected — only Linear layers are cast on its path. | pod log |
| VRAM plan (48 GB) | DiT bf16 14 + text encoder fp8 ~9 + VAE 1.4 + rewriter fp8 ~9.5 ≈ 35 GB steady; `QUANT=auto` → bf16 DiT for 2.1, fp8_layerwise for 2511 | arithmetic — **TODO(pod)** measure |
| Lock file | `server/requirements.lock.txt` is the 2026-09-17 freeze with only diffusers re-pinned → PROVISIONAL. Phase 0 on the `v3` image re-freezes it. | — |
| Timings | **TODO(pod)**: 40-step 1024² and 2048² edit, rewriter load + rewrite latency (thinking on/off), cold boot with 52 GB of weights | phase0_report.json |

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
- **sm ≥ 8.9 (L40S / 6000 Ada / 4090):** also `fp8_layerwise` by default. Real fp8 matmuls via
  `QUANT=fp8_torchao` are opt-in and need a torch-matched torchao added to `requirements.in`
  first (0.18.0 is incompatible with torch 2.9.1 — see the pins table).
- **24 GB cards:** `QUANT=nf4 TEXT_ENCODER_CPU_OFFLOAD=1`. Not the plan.
- **bf16 unquantised needs ~60 GB** → not possible on any 48 GB card. (The spec's table implies
  it might fit; it does not: 57.7 GB of weights.)

## Measurements (MEASURED — Phase 0, 2026-09-17, pod 1a5stk0e531qn3)

A40 was out of stock in every datacenter (Secure and Community) at the time; ran on the
**RTX A6000** instead — same Ampere sm_86 / 48 GB class, so the quant path and numbers transfer.

| Metric | Value |
|---|---|
| GPU / capability actually seen | NVIDIA RTX A6000, 49140 MiB, driver 595.91.07, **capability (8, 6)** |
| torch / CUDA / cudnn on pod | 2.9.1+cu129 / 12.9 / 91002; Python 3.12.3 |
| quant path chosen | `fp8_layerwise` (transformer + text encoder fp8 storage, bf16 compute); LoRA fused before cast |
| image pull (4.94 GB from GHCR) | ~2.5 min (pod created 19:26:59 → container start 19:29:37 UTC) |
| cold load incl. 57.7 GB download (s) | **544 s** (download ≈ 7.5 min unauthenticated HF; load + LoRA fuse + fp8 cast ≈ 1.5 min) |
| VRAM idle after load (GB) | **30.14** |
| VRAM peak during an edit (GB) | **36.94** (1024×768) |
| edit #1 incl. warmup, 8 steps, 1024×768 (s) | 17.49 |
| edit #2 steady state, 8 steps, 1024×768 (s) | **16.84** → ~2.1 s/step; ≈ $0.0025 per edit at $0.53/hr |
| Phase 0 wall time, create → delete | 21 min |
| smoke_test cold-boot, pod create → `model_loaded: true` (s) | **~365 s** (A40 Secure, pod m7xxar2cqdaai9; image pull + 58 GB download + load + warmup) — this datacenter downloaded much faster than the Phase 0 one |
| smoke_test `server_elapsed` (s) | **13.66 s** for 768×512, 8 steps, via the public proxy; 16.5 s wall incl. upload/poll/download (Phase 1 PASS, A40) |
| `v0` image digest (provisional lock) | `sha256:3c9fd4700c82a7505f782ef93f068a96154613fc2ede18492ef014a24340f490` (4.94 GB, 12 layers) |
| `v1` image digest (frozen lock) | `sha256:faf855ecc0b92e7cd4aed22527934bdcd883713ff5c548fdd502c7a528a96c48` |

Observations from `phase0_out.png` (prompt "make the sky a dramatic sunset, keep everything else
unchanged" on a synthetic blue-sky/green-ground/yellow-disc test image): sky fully replaced with a
photoreal sunset, ground preserved, the disc was interpreted as the sun. Small rendered text in the
corner was garbled — expect to need explicit "keep the text unchanged" instructions for text-bearing images.

Gotchas found: (1) sshd sessions don't inherit Docker `ENV` — fixed in `entrypoint.sh` by writing
`/etc/environment`. (2) bitsandbytes prints "No prebuilt binary for CUDA 12.9, loading CUDA 12.8
instead" — harmless, nf4 path untested. (3) HF download is unauthenticated; setting `HF_TOKEN` on the
pod would raise rate limits / speed. (4) A cold boot is ~2.5 min image pull + ~9 min weights+load
≈ **12 min ≈ $0.10** before the first edit — bigger than the spec's $0.06 estimate.

## Spend log

| Date | What | GPU | Hours | $ |
|---|---|---|---|---|
| 2026-09-17 | Phase 0 (probe, freeze) | RTX A6000 Secure US-TX-1 | 0.35 | $0.18 (balance 13.29 → 13.11) |
| 2026-09-17 | Phase 1 smoke test + first session | A40 Secure | (running) | ~$0.05 to PASS |
| | | | | **running total: ~$0.23 / $13 + current session** |

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
5. Output size default is match-input-aspect with long side clamped to `[MIN_SIDE=512, MAX_SIDE=1024]`.
   **v2 fix:** generation always happens at the pipeline's native ~1 MP size for the aspect ratio;
   the requested size is a post-resize. Passing a non-native `width/height` (e.g. 384×512 for a
   387×516 upload) mismatched the reference-image latents and produced zoomed/cropped results (found
   by the user on the first real session, 2026-09-18). Per-edit time is therefore ~constant (~14–17 s)
   regardless of the requested output size.
6. Rewriter uses `claude-haiku-4-5` (the spec asked for a cheap text model); `REWRITE_MODEL` env overrides.
7. Phase 0 ran on an RTX A6000 (A40 had zero stock everywhere at the time), $0.53/hr Secure.
8. Phase 0 runs inside the `v0` image (built from top-level pins) instead of on a stock
   `runpod/pytorch` template. Same purpose — one throwaway GPU session produces the frozen
   lock — but the freeze now comes from the exact image that ships, and no local Docker is needed.
