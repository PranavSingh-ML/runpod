# imgedit — local chat-style image editor on a RunPod GPU

Upload an image, type an instruction, get the edited image back, keep editing the result
like a conversation. The laptop keeps **all** state (`data/`); the pod is a disposable,
stateless inference server built from a pinned Docker image.

- Model: `Qwen/Qwen-Image-Edit-2511` + 8-step Lightning LoRA (Apache 2.0)
- Pod: FastAPI + `diffusers` (no ComfyUI), async job API, bearer-token auth
- Laptop: Vite + React + Express, SQLite (`node:sqlite`, no native build), images on disk
- Budget target: ~$13 total. See `NOTES.md` for measured numbers and `spec.md` for the design.

```
server/    Dockerfile, requirements.lock.txt, app.py, pipeline.py, config.py  (runs on the pod)
web/       the local app                                                    (runs on the laptop)
data/      history.sqlite + images/                                          (gitignored, yours)
scripts/   phase0.sh, build_push.sh, smoke_test.sh
NOTES.md   verified versions, measured latencies, actual spend
```

## Prerequisites

- Laptop: Node ≥ 22.13 (24 recommended — `node:sqlite` is built in), Python 3 + Pillow
  (only for `scripts/smoke_test.sh`), Docker with buildx (to build the pod image; no GPU
  needed to build), a container registry you can push to (Docker Hub / GHCR).
- RunPod account with ≥ 1 hour of credit at the chosen rate. **Set a low-balance alert in
  the RunPod console on day one.**

## Build order (do not skip ahead — each phase has an acceptance test)

### Phase 0 — pin the environment (~$0.25, once)

1. RunPod → Pods → Deploy: **A40 48GB** (Secure or Community), template
   `runpod/pytorch:1.3.1-cu1290-torch291-ubuntu2404` (pick this exact tag in the template
   dropdown / "Container Image"), container disk **100 GB** (weights are 58 GB; 80 GB works but
   is tight), no network volume, expose SSH.
2. `scp -P <PORT> -r server scripts root@<IP>:/workspace/imgedit/`
3. On the pod: `bash /workspace/imgedit/scripts/phase0.sh`
   Installs the pinned deps, checks compute capability, loads the model with the same code
   as production, runs two edits, writes `phase0_report.json`, `phase0_out.png`,
   `requirements.lock.txt`.
4. Copy those three files back; put `requirements.lock.txt` over `server/requirements.lock.txt`;
   paste the report values into `NOTES.md`.
5. **Terminate the pod.**

*Accept when:* `phase0_out.png` is a plausible edit and the lock file is in the repo.

### Phase 1 — bake and serve

```
REGISTRY=docker.io/<you> scripts/build_push.sh v1
```

Deploy a pod from `docker.io/<you>/imgedit:v1`:
- Expose HTTP Ports: `8000`
- Env: `API_TOKEN=<long random string>`, `HF_HOME=/workspace/hf`
- Container disk 100 GB, no volume. GPU: A40 48GB (or L40S 48GB later — same image).

Then from the laptop, with a cold pod and nobody SSHing in:

```
POD_URL=https://<POD_ID>-8000.proxy.runpod.net API_TOKEN=... scripts/smoke_test.sh
```

*Accept when:* it prints `PASS` and `smoke_out.png` exists. Record cold-boot seconds and
`server_elapsed` in `NOTES.md`.

### Phase 2 / 3 — the app

```
cd web
npm install
npm run dev          # http://127.0.0.1:5173
```

Open the Connection panel on the right, paste the pod URL and token, Save (written to
`web/.env.local`). The pill goes disconnected → loading model → ready.

Drop/paste an image, type an instruction, Enter. The result becomes the active source for
the next instruction. Click any earlier image to fork from it. `reuse` copies a seed into
the inspector; `reroll` re-runs the same prompt with a new seed. Every PNG is written to
`data/images/` the moment it arrives; the tree is in `data/history.sqlite`.

Production-ish run: `npm run build && npm start`.

**Prompt rewriter (optional):** add an Anthropic API key in the Connection panel and tick
"enable Rewrite". The Rewrite button turns a casual message into a literal edit instruction
and puts it in the box for you to edit — nothing is ever sent without you seeing it.

## Pod server contract

Every route except `/health` needs `Authorization: Bearer <API_TOKEN>`.

| Route | |
|---|---|
| `GET /health` | `{status, model_loaded, gpu, capability, quant, vram_used_gb, queue_depth, ...}` |
| `POST /edit` (multipart) | `image`, `image2?`, `prompt`, `negative?`, `steps?`, `guidance?`, `seed?` (-1 random), `size?` ("WxH") → `202 {job_id, seed, width, height}` |
| `GET /jobs/{id}` | `{status: queued\|running\|done\|error, progress, seed, elapsed_s, error}` |
| `GET /jobs/{id}/image` | `image/png` when done (409 otherwise) |
| `DELETE /jobs/{id}` | free server memory |

Pod-side knobs are env vars (see `server/config.py`): `MODEL_ID`, `LORA_ENABLED`, `QUANT`
(`auto|fp8_layerwise|fp8_torchao|nf4|bf16`), `DEFAULT_STEPS`, `DEFAULT_GUIDANCE`, `MAX_SIDE`.
Switching to `Qwen/Qwen-Image-Edit-2509` or FLUX Kontext is a config change, not a rewrite.

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

**Terminate the pod at the end of every session.** The app nags after 15 idle minutes;
the RunPod console is the only thing that actually stops the meter.
