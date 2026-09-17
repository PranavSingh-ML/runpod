# Build spec: local chat-style image editor backed by a RunPod GPU

You are Claude Code. Build this repo. Read the whole spec before writing any code.
Where the spec says **VERIFY**, do not guess — run the check and record the real value
in `NOTES.md`.

---

## 0. Goal and hard constraints

Build a **local web app** (runs on the user's laptop) that talks to a **stateless
inference server** on a RunPod GPU pod. The user uploads an image, types an
instruction in a chat box, and gets an edited image back. Each edit can be applied to
the previous output, so edits chain like a conversation.

Constraints that drive every decision below:

1. **Budget is ~$13 total.** Every hour of GPU time is ~3–7% of the budget. Nothing
   may require long trial-and-error *on* the pod.
2. **The user has repeatedly hit dependency/version hell on RunPod templates.** This
   is the primary failure mode to design against. The fix is a **pinned, pre-built
   Docker image**. There must be **zero interactive `pip install` on the pod** in the
   steady state.
3. **No ComfyUI.** Direct `diffusers` pipeline behind a small FastAPI server.
4. **All state lives on the laptop.** The pod is disposable. Killing the pod must lose
   nothing but the warm model.

---

## 1. Decisions (do not relitigate these without telling the user)

### Model: `Qwen/Qwen-Image-Edit-2511`

- 20B MMDiT + Qwen2.5-VL text encoder. Apache 2.0 — permissive for outputs, unlike
  the FLUX non-commercial licenses.
- Instruction-based editing is its native task; it is strong on identity preservation
  across edits and on editing text inside images.
- Use the **8-step Lightning** variant. At 8 steps instead of 20–40, the user gets
  3–5x more images per dollar and the app feels interactive. This is the single
  biggest cost lever in the project.

**Fallback if 2511 fights you:** `Qwen/Qwen-Image-Edit-2509` (same pipeline class,
more settled tooling). Second fallback: `black-forest-labs/FLUX.1-Kontext-dev` (12B,
fits 24GB easily, drifts less over long edit chains, but non-commercial license and
gated on HF). Switching models must be a config change, not a rewrite — see §4.

### Quantization: depends on the GPU generation. This matters.

FP8 *compute* requires Ada (sm_89) or newer. On Ampere (A40, A6000, 3090 — sm_86),
FP8 weights still halve VRAM but get upcast for the math, so you get the memory win
and not the speed win. `torchao` FP8 dynamic-activation configs may fail outright on
sm_86.

| Pod GPU | Arch | Use this |
|---|---|---|
| A40 / A6000 48GB | Ampere | FP8 **weight-only** safetensors, or NF4 (bitsandbytes) |
| L40S / RTX 6000 Ada 48GB | Ada | FP8 (`torchao`), native |
| RTX 4090 24GB | Ada | NF4, plus text-encoder CPU offload |

**VERIFY on first boot:** print `torch.cuda.get_device_capability()` and log it. Pick
the quant path from that, not from the pod name.

### Hardware: start on A40 48GB, upgrade only if latency annoys the user

RunPod Secure Cloud rates (**VERIFY — these move**): A40 48GB ~$0.44/hr, RTX A6000
48GB ~$0.49/hr, L40S 48GB ~$0.86/hr, RTX 4090 24GB ~$0.69/hr Secure / ~$0.34/hr
Community. Billing is per second.

- **A40 48GB @ ~$0.44/hr → ~29 GPU-hours on $13.** Fits transformer + text encoder
  resident with no offload dance. This is the debugging and first-session GPU.
- **L40S 48GB @ ~$0.86/hr → ~15 hours,** but real FP8 and faster per edit. If the
  user complains the chat feels sluggish, this is a dropdown change — the Docker
  image is identical. That portability is the whole point of §2.
- **Do not** start on a 24GB card. Squeezing a 20B model plus an 8.3B text encoder
  into 24GB means CPU offload, which is exactly the kind of fragile setup that
  burned the user before.

### Storage: no network volume

A 60GB RunPod network volume is ~$4.20/month — a third of the budget, charged whether
the pod runs or not. Re-downloading ~25GB of weights on each cold boot costs ~5–8
minutes of GPU time, about **$0.06 per session**. Skip the volume. Download to
container disk, keep the pod alive for the length of a working session, terminate it
when done.

Set container disk to **80GB** (weights + HF cache + headroom).

### Serving: async job API, not synchronous

**Critical:** RunPod's HTTP proxy (`https://POD_ID-8000.proxy.runpod.net`) sits behind
Cloudflare with a **~100-second request timeout**. A synchronous `POST /edit` that
blocks on generation will work in testing and then fail on the first cold model load
or high-resolution edit.

So: `POST /edit` enqueues and returns a `job_id` immediately. The client polls
`GET /jobs/{id}`. This also survives laptop wifi drops, which a long synchronous
request does not.

---

## 2. The version-hell fix (this is the part that must not be skipped)

The user's past attempts failed here. Design for it explicitly.

1. **One `Dockerfile`, one `requirements.lock.txt`, pushed to a registry.** The pod
   runs a fixed image tag. It never installs anything at runtime.
2. **Never `pip install git+https://github.com/huggingface/diffusers` without a
   commit SHA.** New pipeline classes (e.g. `QwenImageEditPlusPipeline`) often land
   on `main` before a release, and `main` is not reproducible. Pin the SHA:
   `pip install "diffusers @ git+https://github.com/huggingface/diffusers@<40-char-sha>"`
   Record the SHA and the date in `NOTES.md`.
3. **Derive the lock file from one throwaway session, then freeze it forever.**
   Phase 0 below does this. `pip freeze > requirements.lock.txt`, commit it, bake it.
4. **Pin the base image tag by digest**, not `:latest`. Use a `runpod/pytorch:*` or
   `nvidia/cuda:*-devel-ubuntu22.04` tag whose CUDA version matches the torch wheel.
   **VERIFY** the current tag list — do not copy a tag from a blog post.
5. **Tag images `imgedit:v1`, `v2`, …** Never `latest`. A pod restart must not be
   able to pull different code.

If the user has no local NVIDIA GPU, the image cannot be smoke-tested locally. In that
case Phase 1's acceptance test *is* the test — budget 15 minutes of pod time for it.

---

## 3. Repo layout

```
imgedit/
├── server/                  # runs on the pod
│   ├── Dockerfile
│   ├── requirements.lock.txt
│   ├── app.py               # FastAPI: routes, job queue
│   ├── pipeline.py          # model load + a single edit() function
│   └── config.py            # model id, quant mode, steps, guidance
├── web/                     # runs on the laptop
│   ├── ... (Vite + React, or Next.js — your call, keep it single-page)
│   └── .env.local           # POD_URL, API_TOKEN, optional ANTHROPIC_API_KEY
├── data/                    # laptop-side state, gitignored
│   ├── history.sqlite
│   └── images/
├── scripts/
│   ├── build_push.sh
│   └── smoke_test.sh        # curl-based, hits a live pod
├── NOTES.md                 # verified versions, measured latencies, actual costs
└── README.md
```

---

## 4. Server contract

Auth: every route except `/health` requires `Authorization: Bearer <API_TOKEN>`,
where the token comes from a pod env var. The proxy URL is public; without this the
user is running an open GPU endpoint on the internet. Non-negotiable.

```
GET  /health
     -> 200 {"status":"ok","model_loaded":bool,"gpu":"NVIDIA A40",
             "capability":[8,6],"vram_used_gb":float,"queue_depth":int}
     No auth. Used for readiness polling while the model loads.

POST /edit                      (multipart/form-data)
     image        : file (required)  — PNG/JPEG, the source
     image2       : file (optional)  — second reference; 2511 supports multi-image
     prompt       : str  (required)  — the edit instruction
     negative     : str  (optional)
     steps        : int  (optional, default from config)
     guidance     : float(optional)
     seed         : int  (optional, -1 = random)
     size         : str  (optional, e.g. "1024x1024"; default = match input aspect,
                          long side clamped to config.MAX_SIDE)
     -> 202 {"job_id": "..."}

GET  /jobs/{job_id}
     -> 200 {"status":"queued|running|done|error",
             "progress": 0.0-1.0,       # step callback
             "seed": int,               # resolved seed, for reproducibility
             "elapsed_s": float,
             "error": str|null}

GET  /jobs/{job_id}/image
     -> 200 image/png (only when status == "done")

DELETE /jobs/{job_id}           # free server-side memory
```

Server implementation rules:

- **One GPU, one worker.** A single background thread drains the queue. Never run two
  generations concurrently — it OOMs and helps nothing.
- Load the model **once at import/startup**, `.to("cuda")`. Do **not** call
  `enable_model_cpu_offload()` on a 48GB card; it silently costs 2–4x latency.
- Wrap generation in `torch.inference_mode()`.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the Dockerfile env.
- Keep finished jobs in an in-memory dict with a 30-minute TTL sweeper. Cap at ~50
  results so a long session cannot exhaust RAM.
- On exception in the worker: catch, store the traceback string in the job, **keep the
  worker alive**. A crashed worker means a dead pod and wasted money.
- Log every job as one JSON line: prompt, seed, steps, resolution, elapsed. The user
  will want this to tune prompts.

---

## 5. Local app

Single page. Priorities: reliability and speed of iteration over visual polish.

**Layout:** left = conversation thread (source image at top, then instruction/result
pairs). Right = a small inspector for steps / guidance / seed / resolution, plus a
connection pill showing pod status and a live session-cost estimate.

**Behaviour that makes it feel like Grok Imagine:**

- The default source for turn *N+1* is the output of turn *N*. The user types
  "now make the background a beach" and it just works.
- **Every result is a branchable node.** Clicking any earlier image sets it as the
  active source, forking the thread. This is what makes it feel conversational rather
  than like a form. Store the tree in SQLite: `id, parent_id, prompt, seed, steps,
  guidance, image_path, created_at`.
- Show the resolved seed on each result with a one-click "reuse seed" and
  "reroll seed". Chained editing is unusable without seed control.
- Optimistic UI: the result card appears immediately in `running` state with a
  progress bar fed by `GET /jobs/{id}` polling at 1s.
- Drag-drop and clipboard paste for the source image.
- Everything written to `data/images/` as it arrives. No result should exist only in
  browser memory.

**Optional prompt rewriter (do this last, behind a toggle):** casual chat messages
make poor edit instructions. If `ANTHROPIC_API_KEY` is set, pass the user's message
plus the last few turns to a cheap text model and ask it to emit one precise,
literal edit instruction (name the subject, name the change, say what must stay
unchanged). Show the rewritten instruction in the UI and let the user edit it before
it is sent. Never send a silently-rewritten prompt. This is a meaningful part of why
polished chat editors feel good, but it is an add-on — ship §6 Phase 3 without it
first.

**Connection setup:** the user pastes the pod proxy URL and token into a settings
panel; persist to `.env.local`. The app polls `/health` every 10s and shows
disconnected / loading-model / ready. Do not ask for RunPod *account* credentials —
the app never needs the RunPod API. (If the user later wants one-click pod
start/stop, that is a separate feature using a RunPod API key and the GraphQL API;
do not build it now.)

---

## 6. Build order — each phase has an acceptance test

Do not start a phase until the previous one's test passes.

**Phase 0 — pin the environment (~30 min pod time, ~$0.25)**
Rent an A40 from a stock `runpod/pytorch` template. In a shell, install torch (already
present), `diffusers` at a chosen SHA, `transformers`, `accelerate`, `safetensors`,
plus the quant lib the capability check selected. Run one edit end to end in a Python
REPL and save the PNG.
*Accept when:* one edited image exists, and `pip freeze > requirements.lock.txt` has
been copied off the pod into the repo. Record the base image tag, the diffusers SHA,
the quant path, and the seconds-per-8-step-edit in `NOTES.md`. **Then terminate the
pod.**

**Phase 1 — bake and serve**
Write `Dockerfile` from the lock file, write `app.py` + `pipeline.py`, build, push.
Launch a pod from the custom image with `Expose HTTP Ports: 8000`, env `API_TOKEN`
and `HF_HOME=/workspace/hf`.
*Accept when:* `scripts/smoke_test.sh` — curl `/health`, `POST /edit`, poll, download
the PNG — passes against the proxy URL from the laptop, with a cold pod, without
anyone SSHing in to fix anything.

**Phase 2 — minimal web app**
Upload, prompt, result, saved to disk. No branching, no inspector.
*Accept when:* three chained edits work from the browser and all three PNGs are on
disk with their rows in SQLite.

**Phase 3 — the chat feel**
Branching tree, seeds, parameter inspector, cost meter, paste/drag, reconnect
handling. Then optionally the rewriter.

---

## 7. Known traps — check each one explicitly

- **Cloudflare 100s proxy timeout.** Handled by the async API. Do not "simplify" it
  back to synchronous.
- **Service must bind `0.0.0.0`, not `127.0.0.1`,** or the RunPod proxy sees nothing.
  This is the single most common "my pod is broken" cause.
- **First request is slow even after startup** (CUDA graph / kernel warmup). Run one
  throwaway 8-step edit on a blank image at startup, before flipping
  `model_loaded: true`. The user should never experience the warmup.
- **`torchao` FP8 on Ampere** may error or silently fall back. Decide by capability,
  not by card name. NF4 via bitsandbytes is the Ampere-safe path.
- **HF gated repos** need `HF_TOKEN`. Qwen is not gated; FLUX Kontext is. If falling
  back to FLUX, the token must be set before the image is built or the pod started.
- **Pod stopped without a network volume is terminated and its disk erased.** Pull
  everything you care about to the laptop before stopping. The design in §1 already
  assumes this, but do not add a "save to pod" feature.
- **RunPod requires ≥1 hour of credit** at the chosen rate before it will deploy an
  on-demand pod. At ~$0.44/hr that is fine with $13; it will bite near the end of the
  balance.
- **Set a spend alert / low-balance notification** in the RunPod console on day one.
- **Terminate the pod at the end of every session.** Put this in the README as the
  last line and print a reminder in the web app when idle >15 minutes.

---

## 8. What to write into NOTES.md as you go

Base image tag + digest · diffusers commit SHA · torch/CUDA versions · GPU and
compute capability · quantization path chosen and why · VRAM at idle and at peak ·
seconds per edit at 8 steps / 1024px · cold-boot seconds including weight download ·
running total spend. This file is what makes the build reproducible next month when
the model has been superseded.