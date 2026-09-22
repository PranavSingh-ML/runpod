"""FastAPI server: routes + a single-worker job queue.

Contract (see README / spec section 4):
  GET    /health                  no auth
  POST   /edit                    multipart -> 202 {"job_id"}
  POST   /rewrite                 multipart -> 202 {"job_id"}   (prompt rewriter, if loaded)
  GET    /jobs/{id}               status/progress/seed/elapsed/error (+ result for rewrite jobs)
  GET    /jobs/{id}/image         image/png when done
  DELETE /jobs/{id}

One GPU, one worker thread shared by edits and rewrites. The worker never dies on a job exception.
"""
from __future__ import annotations

import hmac
import io
import json
import logging
import os
import queue
import random
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

import config
from pipeline import make_engine, parse_resolution, parse_size, to_png_bytes
from rewriter import make_rewriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

if not config.API_TOKEN and not config.MOCK:
    log.error("API_TOKEN env var is empty. Refusing to expose an open GPU endpoint. Set API_TOKEN and restart.")
    sys.exit(2)


# ----------------------------------------------------------------------------
# job store
# ----------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    kind: str = "edit"  # edit | rewrite
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0
    seed: int = 0
    elapsed_s: float = 0.0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    # inputs
    images: list = field(default_factory=list)
    n_images: int = 1
    prompt: str = ""
    negative: str = ""
    steps: int = 8
    guidance: float = 1.0
    width: int = 1024
    height: int = 1024
    resolution: int = 1024
    instruction: str = ""  # rewrite jobs
    # output
    png: Optional[bytes] = None
    result: Optional[dict] = None  # rewrite jobs

    def public(self) -> dict:
        d = {"kind": self.kind, "status": self.status, "progress": round(self.progress, 3), "seed": self.seed,
             "elapsed_s": round(self.elapsed_s, 2), "error": self.error,
             "width": self.width, "height": self.height, "steps": self.steps, "guidance": self.guidance,
             "resolution": self.resolution}
        if self.kind == "rewrite":
            d["result"] = self.result
        return d


class JobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.q: "queue.Queue[str]" = queue.Queue()

    def add(self, job: Job) -> None:
        with self.lock:
            self.jobs[job.id] = job
            self._cap_locked()
        self.q.put(job.id)

    def get(self, job_id: str) -> Optional[Job]:
        with self.lock:
            return self.jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self.lock:
            return self.jobs.pop(job_id, None) is not None

    def depth(self) -> int:
        with self.lock:
            return sum(1 for j in self.jobs.values() if j.status in ("queued", "running"))

    def _cap_locked(self) -> None:
        finished = [j for j in self.jobs.values() if j.status in ("done", "error")]
        if len(finished) > config.JOB_CAP:
            finished.sort(key=lambda j: j.finished_at or 0)
            for j in finished[: len(finished) - config.JOB_CAP]:
                self.jobs.pop(j.id, None)

    def sweep(self) -> None:
        now = time.time()
        with self.lock:
            dead = [k for k, j in self.jobs.items()
                    if j.status in ("done", "error") and (now - (j.finished_at or now)) > config.JOB_TTL_S]
            for k in dead:
                self.jobs.pop(k, None)
            self._cap_locked()


store = JobStore()
engine = make_engine()
rewriter = make_rewriter()
state = {"model_loaded": False, "rewriter_loaded": False, "load_error": None, "started_at": time.time()}


# ----------------------------------------------------------------------------
# worker
# ----------------------------------------------------------------------------
def _log_job(job: Job) -> None:
    rec = {"ts": time.time(), "job_id": job.id, "kind": job.kind, "prompt": job.prompt or job.instruction,
           "negative": job.negative, "seed": job.seed, "steps": job.steps, "guidance": job.guidance,
           "width": job.width, "height": job.height, "resolution": job.resolution,
           "n_images": job.n_images, "elapsed_s": round(job.elapsed_s, 2), "status": job.status,
           "rewritten": (job.result or {}).get("rewritten_prompt") if job.kind == "rewrite" else None,
           "error": (job.error or "").splitlines()[-1] if job.error else None}
    log.info("JOB %s", json.dumps(rec))
    try:
        os.makedirs(os.path.dirname(config.LOG_PATH) or ".", exist_ok=True)
        with open(config.LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def _run_job(job: Job) -> None:
    job.status = "running"
    t0 = time.time()

    def progress(p: float) -> None:
        job.progress = p
        job.elapsed_s = time.time() - t0

    try:
        if job.kind == "rewrite":
            if rewriter is None or not state["rewriter_loaded"]:
                raise RuntimeError("rewriter not loaded on this pod")
            job.result = rewriter.rewrite(job.images, job.instruction)
        else:
            img = engine.edit(job.images, job.prompt, job.negative, job.steps, job.guidance, job.seed,
                              job.width, job.height, progress, resolution=job.resolution)
            job.png = to_png_bytes(img)
        job.status = "done"
        job.progress = 1.0
    except Exception:  # noqa: BLE001 - keep the worker alive no matter what
        job.error = traceback.format_exc()
        job.status = "error"
        log.exception("job %s failed", job.id)
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    finally:
        job.elapsed_s = time.time() - t0
        job.finished_at = time.time()
        job.images = []  # free input memory
        _log_job(job)


def _warmup() -> None:
    if not config.WARMUP:
        return
    log.info("warmup edit starting")
    t0 = time.time()
    blank = Image.new("RGB", (512, 512), (128, 128, 128))
    res = min(config.DEFAULT_RESOLUTION, 1024)
    w, h = parse_size(None, 512, 512, res)
    engine.edit([blank], "add a small red circle in the centre", config.DEFAULT_NEGATIVE, min(config.DEFAULT_STEPS, 8),
                config.DEFAULT_GUIDANCE, 0, w, h, lambda p: None, resolution=res)
    log.info("warmup done in %.1fs", time.time() - t0)


def _load_rewriter() -> None:
    """Best effort: a broken rewriter must not take the editor down."""
    if rewriter is None:
        return
    try:
        rewriter.load()
        if config.WARMUP:
            t0 = time.time()
            rewriter.rewrite([Image.new("RGB", (256, 256), (128, 128, 128))], "make it blue")
            log.info("rewriter warmup done in %.1fs", time.time() - t0)
        state["rewriter_loaded"] = True
    except Exception:  # noqa: BLE001
        state["rewriter_error"] = traceback.format_exc()
        log.exception("REWRITER LOAD FAILED - editing still works; /health reports rewriter_loaded=false")


def worker() -> None:
    try:
        engine.load()
        _warmup()
        state["model_loaded"] = True
        _load_rewriter()
    except Exception:  # noqa: BLE001
        state["load_error"] = traceback.format_exc()
        log.exception("MODEL LOAD FAILED - server stays up so /health can report it")
        # drain queue with errors so clients don't hang forever
        while True:
            jid = store.q.get()
            job = store.get(jid)
            if job:
                job.status, job.error, job.finished_at = "error", "model failed to load:\n" + state["load_error"], time.time()
    while True:
        jid = store.q.get()
        job = store.get(jid)
        if job is None:  # deleted while queued
            continue
        _run_job(job)


def sweeper() -> None:
    while True:
        time.sleep(60)
        store.sweep()


threading.Thread(target=worker, name="gpu-worker", daemon=True).start()
threading.Thread(target=sweeper, name="job-sweeper", daemon=True).start()


# ----------------------------------------------------------------------------
# auth
# ----------------------------------------------------------------------------
def require_token(request: Request) -> None:
    if config.MOCK and not config.API_TOKEN:
        return
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    if not hmac.compare_digest(auth[7:].strip(), config.API_TOKEN):
        raise HTTPException(403, "bad token")


app = FastAPI(title="imgedit", docs_url=None, redoc_url=None)


# ----------------------------------------------------------------------------
# routes
# ----------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    info = engine.info or {}
    return {
        "status": "ok",
        "model_loaded": state["model_loaded"],
        "load_error": state["load_error"],
        "gpu": info.get("gpu"),
        "capability": info.get("capability"),
        "quant": info.get("quant"),
        "model": config.MODEL_ID,
        "pipeline": config.PIPELINE,
        "lora": info.get("lora"),
        "rewriter_loaded": state["rewriter_loaded"],
        "rewriter": (rewriter.info if rewriter is not None else None),
        "rewriter_error": (state.get("rewriter_error") or "")[-400:] or None,
        "vram_used_gb": engine.vram_used_gb(),
        "vram_total_gb": info.get("vram_total_gb"),
        "queue_depth": store.depth(),
        "uptime_s": round(time.time() - state["started_at"], 1),
        "defaults": {"steps": config.DEFAULT_STEPS, "guidance": config.DEFAULT_GUIDANCE, "max_side": config.MAX_SIDE,
                     "resolution": config.DEFAULT_RESOLUTION, "max_resolution": config.MAX_RESOLUTION,
                     "max_steps": config.MAX_STEPS},
    }


async def _read_image(up: UploadFile) -> Image.Image:
    data = await up.read()
    if len(data) > 40 * 1024 * 1024:
        raise HTTPException(413, "image too large (>40MB)")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"cannot decode image: {e}") from e
    return img.convert("RGB")


@app.post("/edit", status_code=202, dependencies=[Depends(require_token)])
async def edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    image2: Optional[UploadFile] = File(None),
    negative: Optional[str] = Form(None),
    steps: Optional[int] = Form(None),
    guidance: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    size: Optional[str] = Form(None),
    resolution: Optional[str] = Form(None),
) -> dict:
    if not prompt.strip():
        raise HTTPException(400, "prompt is empty")
    imgs = [await _read_image(image)]
    if image2 is not None and image2.filename:
        imgs.append(await _read_image(image2))
    try:
        res = parse_resolution(resolution)
        w, h = parse_size(size, imgs[0].width, imgs[0].height, res)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    s = int(steps) if steps is not None else config.DEFAULT_STEPS
    if not (1 <= s <= config.MAX_STEPS):
        raise HTTPException(400, f"steps must be 1..{config.MAX_STEPS}")
    g = float(guidance) if guidance is not None else config.DEFAULT_GUIDANCE
    if seed is None or int(seed) < 0:
        seed = random.randint(0, 2**31 - 1)
    job = Job(id=uuid.uuid4().hex[:12], seed=int(seed), images=imgs, n_images=len(imgs), prompt=prompt.strip(),
              negative=(negative if negative is not None else config.DEFAULT_NEGATIVE),
              steps=s, guidance=g, width=w, height=h, resolution=res)
    store.add(job)
    return {"job_id": job.id, "seed": job.seed, "width": w, "height": h, "resolution": res}


@app.post("/rewrite", status_code=202, dependencies=[Depends(require_token)])
async def rewrite(
    image: UploadFile = File(...),
    instruction: str = Form(...),
    image2: Optional[UploadFile] = File(None),
) -> dict:
    if rewriter is None:
        raise HTTPException(501, "rewriter disabled on this pod (PE_ENABLED=0)")
    if not instruction.strip():
        raise HTTPException(400, "instruction is empty")
    imgs = [await _read_image(image)]
    if image2 is not None and image2.filename:
        imgs.append(await _read_image(image2))
    job = Job(id=uuid.uuid4().hex[:12], kind="rewrite", images=imgs, n_images=len(imgs), instruction=instruction.strip(),
              width=imgs[0].width, height=imgs[0].height)
    store.add(job)
    return {"job_id": job.id}


@app.get("/jobs/{job_id}", dependencies=[Depends(require_token)])
def job_status(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job (expired or deleted)")
    return job.public()


@app.get("/jobs/{job_id}/image", dependencies=[Depends(require_token)])
def job_image(job_id: str) -> Response:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job (expired or deleted)")
    if job.status != "done" or job.png is None:
        return JSONResponse({"status": job.status, "error": job.error}, status_code=409)
    return Response(content=job.png, media_type="image/png",
                    headers={"X-Seed": str(job.seed), "X-Elapsed-S": f"{job.elapsed_s:.2f}"})


@app.delete("/jobs/{job_id}", dependencies=[Depends(require_token)])
def job_delete(job_id: str) -> dict:
    return {"deleted": store.delete(job_id)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
