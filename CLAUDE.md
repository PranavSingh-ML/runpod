# imgedit — instructions for Claude Code (any machine)

Local chat-style image editor. Laptop runs `web/` (Vite+React+Express, SQLite, images on disk);
a disposable RunPod GPU pod runs `server/` from a pinned Docker image. Design: `spec.md`.
Measured numbers, pins, spend log: `NOTES.md`. Human runbook: `README.md`.

## Windows note

Type `bash` on Windows may be **WSL** (an Ubuntu VM with its own ancient Node) rather than Git Bash.
Always use `.\pod.cmd <up|wait|status|down>` on Windows — it runs `scripts/pod.sh` under Git Bash.

## The one rule

**A running pod costs money every minute ($0.49–0.53/hr). Never end a session with a pod up
unless the user explicitly said to keep it.** Deleting the pod (*Stop pod* in the app,
`bash scripts/pod.sh down`, or `POST /api/pod/down`) is the only thing that stops billing ("stop"
in the RunPod console does NOT — a stopped pod still bills its disk). Check with
`bash scripts/pod.sh status` or the Pod panel whenever unsure.

## Daily use (nothing to build)

```
cd web && npm run dev         # http://127.0.0.1:5173  — user edits pictures here
```

The app's **Pod** panel does start / wait / stop (`web/server/runpod.ts` shells out to
`tools/runpodctl.exe`; the app never touches the API key). **Auto-stop** deletes the pod after
`AUTO_STOP_MIN` idle minutes (default 20, in `web/.env.local` / Connection panel; 0 = off). Idle
= no edit submitted since the model loaded; it never fires while an edit is in flight.

Terminal equivalent (still works, same GPU fallback):

```
.\pod.cmd up               # (Windows) creates the pod, writes POD_URL into web/.env.local  (~6-12 min cold boot)
.\pod.cmd wait             # blocks until the model is loaded
.\pod.cmd down             # WHEN DONE   (Linux/mac: bash scripts/pod.sh <cmd>)
```

Results: `data/images/*.png`, history tree: `data/history.sqlite` (gitignored, the user's data —
never delete). The pod is stateless; deleting it loses nothing.

## First-time setup on a new machine

1. Git, Node ≥ 22.13 (24 preferred; `node:sqlite` is built in), Python 3 + Pillow (only for
   `scripts/smoke_test.sh`). **No Docker** — images build on GitHub Actions.
2. `git clone https://github.com/PranavSingh-ML/runpod.git` → `cd web && npm install`.
3. `runpodctl`: download `runpodctl-windows-amd64.exe` (or the Linux/mac build) from
   github.com/runpod/runpodctl/releases into `tools/runpodctl.exe`, then the **user** runs
   `tools\runpodctl.exe doctor` and pastes their RunPod API key. Claude must not handle the key.
4. Optional: copy `data/` from the old machine for history; copy `web/.env.local` or let
   `pod.sh up` generate a fresh `API_TOKEN`.

## Where credentials live (never commit, never print)

| Secret | Location | In git? |
|---|---|---|
| RunPod API key | `~/.runpod/config.toml` (written by `runpodctl doctor`) | no |
| RunPod SSH key | `~/.runpod/ssh/runpodctl-ssh-key` (private; public half is on the account) | no |
| Pod `API_TOKEN` | `web/.env.local` (also passed as pod env at create time) | no (gitignored) |
| Anthropic key (optional rewriter) | `web/.env.local` | no |
| GitHub push | OS credential manager | no |

The Docker image on GHCR is public and contains only code. `apikey.txt` is gitignored — if present it
is a leftover the user should delete.

## Images

- `:v3` (default since 2026-09-21): `Qwen/Qwen-Image-2.1` (`PIPELINE=qwen_image_21`, 7B DiT +
  8B text encoder, native 2K, sampled without guidance at 40 steps) + `Qwen/Qwen-Image-2.1-PE-I2I`
  prompt enhancer on the same GPU (`POST /rewrite`). Research license = personal use only.
  diffusers is git-pinned (2.1 support is not in a PyPI release yet) — the lock is PROVISIONAL
  until Phase 0 re-freezes it on a real pod.
- `:v2`: `Qwen/Qwen-Image-Edit-2511` + Lightning LoRA, 1 MP, 8 steps. Same code path is still
  in v3 (`PIPELINE=qwen_edit_plus`). Use `IMAGE=ghcr.io/pranavsingh-ml/imgedit:v2` if v3 misbehaves.

## Changing the pod image (rare)

Edit `server/*`, commit, then `git tag v3 && git push origin main v3` → GitHub Actions builds
`ghcr.io/pranavsingh-ml/imgedit:v3` (~10 min, free). Never tag `latest`. Then
`IMAGE=ghcr.io/pranavsingh-ml/imgedit:v3 bash scripts/pod.sh up`. If you change Python deps,
re-run Phase 0 (`README.md`) to re-freeze `server/requirements.lock.txt` — do not hand-edit the lock.

## Knobs

- v3 resolution: generation budget `DEFAULT_RESOLUTION` (1024) up to `MAX_RESOLUTION` (2048),
  source aspect, multiples of 32; per-edit via the inspector (`resolution` form field).
- v2 output size: input aspect, long side clamped to 512–1024, multiples of 16 (`MAX_SIDE`).
- Steps/guidance: v3 40 / 1.0 (no Lightning LoRA for 2.1 yet; guidance >1 doubles the time).
  v2: 8 / 1.0 with Lightning; `LORA_ENABLED=0` → 40 / 4.0, 5× slower. The web app adopts the pod's
  defaults whenever the pipeline changes.
- Quant: v3 keeps the 7B DiT in bf16 and fp8-casts the text encoder + rewriter (~35 GB steady).
  v2 `fp8_layerwise` everywhere. `nf4` for 24 GB cards. `fp8_torchao` is opt-in and needs a
  torch-matched torchao added first — torchao 0.18 broke the build once (`NOTES.md`).
- Rewriter: `PE_ENABLED=0` saves ~9.5 GB VRAM and a 19 GB download; the app then falls back to the
  Claude text rewriter if an Anthropic key is set.

## Testing without a GPU

`cd server && MOCK=1 API_TOKEN=test python -m uvicorn app:app --port 8000`, point the app at
`http://127.0.0.1:8000` / token `test`. Everything except the actual model is real.
