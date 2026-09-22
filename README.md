# imgedit — local chat-style image editor on a RunPod GPU

Upload an image, type an instruction, get the edited image back, keep editing the result
like a conversation. The laptop keeps **all** state (`data/`); the pod is a disposable,
stateless inference server built from a pinned Docker image.

- Model (`:v3`): `Qwen/Qwen-Image-2.1` (native 2K, Qwen Research License = personal/non-commercial)
  + its prompt enhancer `Qwen/Qwen-Image-2.1-PE-I2I` on the same GPU
- Model (`:v2`, fallback): `Qwen/Qwen-Image-Edit-2511` + 8-step Lightning LoRA (Apache 2.0, 1 MP)
- Pod: FastAPI + `diffusers` (no ComfyUI), async job API, bearer-token auth
- Laptop: Vite + React + Express, SQLite (`node:sqlite`, no native build), images on disk
- Budget target: ~$13 total. See `NOTES.md` for measured numbers and `spec.md` for the design.

```
server/    Dockerfile, requirements.lock.txt, app.py, pipeline.py, config.py  (runs on the pod)
web/       the local app                                                    (runs on the laptop)
data/      history.sqlite + images/                                          (gitignored, yours)
scripts/   pod.sh (up|wait|status|down), phase0.sh, build_push.sh, smoke_test.sh
pod.cmd    Windows wrapper: runs scripts/pod.sh under Git Bash (not WSL)
tools/     runpodctl.exe (gitignored - download it; see Prerequisites)
NOTES.md   verified versions, measured latencies, actual spend
```

## Prerequisites

- Laptop: Node ≥ 22.13 (24 recommended — `node:sqlite` is built in), Python 3 + Pillow,
  Git, `ssh`/`scp` (Git for Windows has them). **No Docker needed** — images build on
  GitHub Actions.
- A GitHub repo for this code (public or private; the *package* must be public or RunPod
  needs registry auth — see below).
- RunPod account with ≥ 1 hour of credit at the chosen rate, an API key configured in
  `runpodctl` (`./tools/runpodctl.exe doctor`), and the SSH public key
  `~/.ssh/imgedit_runpod.pub` added to the account (`runpodctl ssh add-key`).
  **Set a low-balance alert in the RunPod console on day one.**

## Image builds (GitHub Actions → GHCR)

```
git tag v0 && git push origin main v0      # builds ghcr.io/<owner>/imgedit:v0
```

`.github/workflows/build-image.yml` builds `server/` and pushes `ghcr.io/<owner>/imgedit:<tag>`;
the run summary shows the digest — copy it into `NOTES.md`. Never `latest`.

After the **first** push: GitHub → your profile → Packages → `imgedit` → Package settings →
**Change visibility → Public**, so RunPod can pull it without credentials. (The image contains
only code; the API token is an env var, never baked in.)

Base image is `nvidia/cuda:12.9.0-base-ubuntu24.04` (109 MB) + Python 3.12 + torch 2.9.1+cu129
from the PyTorch index → ~5 GB compressed. Small on purpose: RunPod pulls it on every cold boot.

## Build order (do not skip ahead — each phase has an acceptance test)

### Phase 0 — pin the environment (~$0.25, once)

Runs *inside* the `v0` image on a real GPU, so the frozen lock is exactly what ships.

```
# 1. build v0 from the provisional lock (top-level pins)         -> Actions, ~10 min, free
git tag v0 && git push origin main v0

# 2. start a Phase-0 pod (sshd only, no server)                   -> meter starts
./tools/runpodctl.exe pod create --name imgedit-phase0 \
  --image ghcr.io/<owner>/imgedit:v0 --gpu-id "NVIDIA A40" \
  --container-disk-in-gb 100 --ports "8000/http,22/tcp" \
  --env '{"PHASE0":"1","API_TOKEN":"phase0","HF_HOME":"/workspace/hf"}' --wait

# 3. drive it from the laptop: env check, model load, two edits, pip freeze, copy back
scripts/phase0.sh <POD_ID>

# 4. freeze + terminate
cp phase0/requirements.lock.txt server/requirements.lock.txt
./tools/runpodctl.exe pod delete <POD_ID>
```

*Accept when:* `phase0/phase0_out.png` is a plausible edit and the lock file is in the repo.
Paste `phase0/phase0_report.json` into `NOTES.md`, commit.

### Phase 1 — bake and serve

```
git tag v1 && git push origin main v1        # image from the frozen lock
./tools/runpodctl.exe pod create --name imgedit \
  --image ghcr.io/<owner>/imgedit:v1 --gpu-id "NVIDIA A40" \
  --container-disk-in-gb 100 --ports "8000/http,22/tcp" \
  --env '{"API_TOKEN":"<long random string>","HF_HOME":"/workspace/hf"}'
```

Then from the laptop, with a cold pod and nobody SSHing in:

```
POD_URL=https://<POD_ID>-8000.proxy.runpod.net API_TOKEN=... scripts/smoke_test.sh
```

*Accept when:* it prints `PASS` and `smoke_out.png` exists. Record cold-boot seconds and
`server_elapsed` in `NOTES.md`. Delete the pod (`runpodctl pod delete`) unless you're going
straight into a session.

### Phase 2 / 3 — the app

```
cd web
npm install
npm run dev          # http://127.0.0.1:5173
```

**Start the pod from the app:** the *Pod* panel (top right) has **Start pod** (A40 → A6000
fallback, same as `pod.sh up`), a boot-phase line (booting → loading model → ready, i.e. `pod.sh
wait`), **Stop pod**, your RunPod balance, and an **auto-stop** countdown. Auto-stop deletes the
pod after 20 idle minutes by default (idle = no edit submitted since the model loaded; set it in
Connection → *auto-stop*, 0 = off). It needs `tools/runpodctl.exe` configured with `doctor`; the
app shells out to it and never sees your RunPod API key.

Without runpodctl you can still run `.\pod.cmd up` (or `bash scripts/pod.sh up`), or paste a pod
URL and token into the Connection panel by hand; Save writes them to `web/.env.local`. The pill goes
disconnected → loading model → ready.

Drop/paste an image, type an instruction, Enter. The result becomes the active source for
the next instruction. Click any earlier image to fork from it. `reuse` copies a seed into
the inspector; `reroll` re-runs the same prompt with a new seed. Every PNG is written to
`data/images/` the moment it arrives; the tree is in `data/history.sqlite`.

Production-ish run: `npm run build && npm start`.

**Prompt enhancer:** tick "enable Enhance / Rewrite" in the Connection panel. On a `:v3` pod
the **Enhance** button sends the image + your casual message to the pod's own rewriter
(`Qwen-Image-2.1-PE-I2I`, a Qwen3.5-VL fine-tune that *looks at the picture*) and puts the
detailed, creative instruction in the box for you to edit — the same design as Grok Imagine's
`revised_prompt`, for $0 extra. On a `:v2` pod (or if the pod rewriter fails) the button is
**Rewrite**: text-only Claude Haiku, needs an Anthropic key. Nothing is ever sent without you
seeing it.

**Resolution (`:v3`):** the inspector's *resolution* picks the generation budget — 1024² (~1 MP,
draft), 1536², or 2048² (native 2K, ≈4 MP, ~4× slower). Output keeps the source's aspect.

## Pod server contract

Every route except `/health` needs `Authorization: Bearer <API_TOKEN>`.

| Route | |
|---|---|
| `GET /health` | `{status, model_loaded, pipeline, rewriter_loaded, gpu, capability, quant, vram_used_gb, queue_depth, defaults, ...}` |
| `POST /edit` (multipart) | `image`, `image2?`, `prompt`, `negative?`, `steps?`, `guidance?`, `seed?` (-1 random), `size?` ("WxH"), `resolution?` (v3: 1024/1536/2048) → `202 {job_id, seed, width, height, resolution}` |
| `POST /rewrite` (multipart) | v3 only: `image`, `image2?`, `instruction` → `202 {job_id}`; the job's `result.rewritten_prompt` holds the enhanced prompt (501 if `PE_ENABLED=0`) |
| `GET /jobs/{id}` | `{kind, status: queued\|running\|done\|error, progress, seed, elapsed_s, error, result?}` |
| `GET /jobs/{id}/image` | `image/png` when done (409 otherwise) |
| `DELETE /jobs/{id}` | free server memory |

Pod-side knobs are env vars (see `server/config.py`): `PIPELINE` (`qwen_image_21` default,
`qwen_edit_plus` for 2511, `flux_kontext`), `MODEL_ID`, `LORA_ENABLED`, `QUANT`
(`auto|fp8_layerwise|fp8_torchao|nf4|bf16`), `QUANT_TEXT_ENCODER`, `DEFAULT_STEPS`,
`DEFAULT_GUIDANCE`, `DEFAULT_RESOLUTION` / `MAX_RESOLUTION` (v3), `MAX_SIDE` (v2), and the
rewriter's `PE_ENABLED`, `PE_QUANT`, `PE_THINKING`, `PE_MAX_NEW_TOKENS`. The `:v3` image still
contains the 2511 code path: `PIPELINE=qwen_edit_plus` on a v3 pod is the same as running `:v2`.

### Test the whole stack on the laptop without a GPU

```
cd server && pip install fastapi uvicorn python-multipart pillow
MOCK=1 API_TOKEN=test python -m uvicorn app:app --port 8000
```

Point the app at `http://127.0.0.1:8000` / token `test`. The mock "edits" are tinted copies
of the input; everything else (queue, polling, saving, branching) is real.

## Known traps (all handled, don't undo them)

- RunPod's proxy has a ~100 s Cloudflare timeout → the API is async; never make `/edit` block.
- The server binds `0.0.0.0`; `127.0.0.1` is invisible to the proxy.
- First request after startup is slow (kernel warmup) → the server runs one throwaway edit
  before reporting `model_loaded: true`.
- FP8 *compute* needs sm_89+. On Ampere (A40) the transformer is stored in fp8 and computed
  in bf16 (`fp8_layerwise`); torchao fp8 is only chosen automatically on sm_89+.
- Stopping a pod without a network volume erases its disk. Nothing you care about lives there.
- RunPod needs ≥ 1 h of credit at the pod's rate before it will deploy.

## Last line

**Terminate the pod at the end of every session** — *Stop pod* in the app, `.\pod.cmd down`, or
`./tools/runpodctl.exe pod delete <POD_ID>`. The app auto-stops after 20 idle minutes (if runpodctl
is set up) and nags after 15; only deleting the pod stops the meter.
