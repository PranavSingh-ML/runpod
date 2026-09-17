# imgedit — instructions for Claude Code (any machine)

Local chat-style image editor. Laptop runs `web/` (Vite+React+Express, SQLite, images on disk);
a disposable RunPod GPU pod runs `server/` from a pinned Docker image. Design: `spec.md`.
Measured numbers, pins, spend log: `NOTES.md`. Human runbook: `README.md`.

## Windows note

Type `bash` on Windows may be **WSL** (an Ubuntu VM with its own ancient Node) rather than Git Bash.
Always use `.\pod.cmd <up|wait|status|down>` on Windows — it runs `scripts/pod.sh` under Git Bash.

## The one rule

**A running pod costs money every minute ($0.49–0.53/hr). Never end a session with a pod up
unless the user explicitly said to keep it.** `bash scripts/pod.sh down` is the only thing that
stops billing ("stop" in the RunPod console does NOT — a stopped pod still bills its disk).
Check with `bash scripts/pod.sh status` whenever unsure.

## Daily use (nothing to build)

```
.\pod.cmd up               # (Windows) creates the pod, writes POD_URL into web/.env.local  (~6-12 min cold boot)
.\pod.cmd wait             # blocks until the model is loaded
cd web && npm run dev         # http://127.0.0.1:5173  — user edits pictures here
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

## Changing the pod image (rare)

Edit `server/*`, commit, then `git tag v2 && git push origin main v2` → GitHub Actions builds
`ghcr.io/pranavsingh-ml/imgedit:v2` (~10 min, free). Never tag `latest`. Then
`IMAGE=ghcr.io/pranavsingh-ml/imgedit:v2 bash scripts/pod.sh up`. If you change Python deps,
re-run Phase 0 (`README.md`) to re-freeze `server/requirements.lock.txt` — do not hand-edit the lock.

## Knobs

- Output size: default = input aspect, long side clamped to 512–1024, multiples of 16. Raise
  with `EXTRA_ENV='"MAX_SIDE":"1536"' bash scripts/pod.sh up`, or per-edit via the inspector.
- Steps/guidance: 8 / 1.0 with the Lightning LoRA (default). `LORA_ENABLED=0` → 40 / 4.0, 5× slower.
- Quant: `fp8_layerwise` everywhere (no extra libs). `nf4` for 24 GB cards. `fp8_torchao` is opt-in
  and needs a torch-matched torchao added first — torchao 0.18 broke the build once (`NOTES.md`).

## Testing without a GPU

`cd server && MOCK=1 API_TOKEN=test python -m uvicorn app:app --port 8000`, point the app at
`http://127.0.0.1:8000` / token `test`. Everything except the actual model is real.
