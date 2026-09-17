// Owns the lifecycle of an edit: submit to the pod, poll at 1s, write the PNG to
// data/images the moment it arrives. Runs server-side so a closed tab loses nothing.
// Also: the 10s health poller and the session cost meter.
import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
import * as db from "./db.js";
import * as pod from "./pod.js";
import { IMAGES_DIR, settings } from "./settings.js";

export function imagePathFor(id: string) {
  return path.join(IMAGES_DIR, `${id}.png`);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const POLL_MS = 1000;
const UNREACHABLE_GRACE_MS = 5 * 60 * 1000; // keep retrying a poll this long over wifi drops

// ---- pod status ---------------------------------------------------------------
export const podState: {
  reachable: boolean;
  health: pod.Health | null;
  lastError: string | null;
  lastOkAt: number | null;
  lastEditAt: number | null;
} = { reachable: false, health: null, lastError: null, lastOkAt: null, lastEditAt: null };

async function pollHealth() {
  if (!settings.POD_URL) {
    podState.reachable = false;
    podState.health = null;
    podState.lastError = "POD_URL not set";
    return;
  }
  try {
    const h = await pod.health();
    podState.reachable = true;
    podState.health = h;
    podState.lastError = null;
    podState.lastOkAt = Date.now();
    // session cost meter: a session is "the pod is reachable"
    const s = db.openSession();
    if (!s) db.startSession(settings.POD_RATE_USD_HR);
    else db.touchSession(s.id, settings.POD_RATE_USD_HR);
  } catch (e: any) {
    podState.reachable = false;
    podState.lastError = e?.message ?? String(e);
    const s = db.openSession();
    // pod gone for > 3 min => session over (terminated or stopped)
    if (s && Date.now() / 1000 - s.last_seen > 180) db.endSession(s.id, s.last_seen);
  }
}

export function startHealthLoop() {
  void pollHealth();
  setInterval(() => void pollHealth(), 10_000);
}

export function pokeHealth() {
  return pollHealth();
}

export function sessionInfo() {
  const s = db.openSession();
  const now = Date.now() / 1000;
  const seconds = s ? Math.max(0, (podState.reachable ? now : s.last_seen) - s.started_at) : 0;
  return {
    active: !!s && podState.reachable,
    startedAt: s?.started_at ?? null,
    seconds,
    rateUsdHr: settings.POD_RATE_USD_HR,
    costUsd: (seconds / 3600) * settings.POD_RATE_USD_HR,
    edits: s?.edits ?? 0,
    totalSpendUsd: db.totalSpend(),
    idleSeconds: podState.lastEditAt ? (Date.now() - podState.lastEditAt) / 1000 : podState.lastOkAt ? seconds : 0,
  };
}

export function resetSession() {
  const s = db.openSession();
  if (s) db.endSession(s.id, Date.now() / 1000);
  if (podState.reachable) db.startSession(settings.POD_RATE_USD_HR);
}

// ---- edit lifecycle -----------------------------------------------------------
export interface EditRequest {
  parentId: string;
  prompt: string;
  negative?: string | null;
  steps?: number | null;
  guidance?: number | null;
  seed?: number | null; // -1 / null = random
  size?: string | null;
  refNodeId?: string | null;
}

const inflight = new Set<string>();

export function createEdit(req: EditRequest): db.NodeRow {
  const parent = db.getNode(req.parentId);
  if (!parent) throw new Error("parent node not found");
  if (parent.status !== "done" || !parent.image_path) throw new Error("parent has no image yet");
  if (req.refNodeId) {
    const ref = db.getNode(req.refNodeId);
    if (!ref || !ref.image_path) throw new Error("reference node has no image");
  }
  const m = /^(\d{3,4})x(\d{3,4})$/.exec((req.size ?? "").trim());
  const id = randomUUID().replace(/-/g, "").slice(0, 16);
  const node = db.insertNode({
    id,
    parent_id: parent.id,
    kind: "edit",
    prompt: req.prompt,
    negative: req.negative ?? null,
    seed: req.seed != null && req.seed >= 0 ? req.seed : null,
    steps: req.steps ?? null,
    guidance: req.guidance ?? null,
    width: m ? Number(m[1]) : null,
    height: m ? Number(m[2]) : null,
    ref_node_id: req.refNodeId ?? null,
    image_path: null,
    status: "queued",
  });
  podState.lastEditAt = Date.now();
  void runEdit(node.id);
  return node;
}

async function runEdit(nodeId: string) {
  if (inflight.has(nodeId)) return;
  inflight.add(nodeId);
  try {
    let node = db.getNode(nodeId);
    if (!node) return;
    const parent = db.getNode(node.parent_id!);
    if (!parent?.image_path) throw new Error("parent image missing");

    // 1. submit (unless we are resuming a job that was already submitted)
    if (!node.job_id) {
      const image = fs.readFileSync(path.join(IMAGES_DIR, parent.image_path));
      let image2: Buffer | null = null;
      if (node.ref_node_id) {
        const ref = db.getNode(node.ref_node_id);
        if (ref?.image_path) image2 = fs.readFileSync(path.join(IMAGES_DIR, ref.image_path));
      }
      const r = await pod.submitEdit(image, image2, {
        prompt: node.prompt!,
        negative: node.negative,
        steps: node.steps,
        guidance: node.guidance,
        seed: node.seed ?? -1,
        size: sizeFor(node),
      });
      db.updateNode(nodeId, { job_id: r.job_id, seed: r.seed, width: r.width, height: r.height, status: "queued" });
      node = db.getNode(nodeId)!;
    }

    // 2. poll at 1s; tolerate the pod being unreachable for a while (wifi drop)
    let unreachableSince: number | null = null;
    for (;;) {
      await sleep(POLL_MS);
      if (!db.getNode(nodeId)) return; // deleted by the user while running
      let st: pod.JobStatus;
      try {
        st = await pod.jobStatus(node.job_id!);
        unreachableSince = null;
      } catch (e: any) {
        if (e instanceof pod.PodError && e.code === 404) throw new Error("job expired on the pod (restarted?)");
        unreachableSince ??= Date.now();
        if (Date.now() - unreachableSince > UNREACHABLE_GRACE_MS) throw new Error(`pod unreachable: ${e.message}`);
        db.updateNode(nodeId, { error: `retrying: ${e.message}` });
        continue;
      }
      if (st.status === "error") throw new Error(st.error ?? "unknown pod error");
      db.updateNode(nodeId, {
        status: st.status === "done" ? "running" : st.status,
        progress: st.progress,
        elapsed_s: st.elapsed_s,
        seed: st.seed,
        width: st.width,
        height: st.height,
        steps: st.steps,
        guidance: st.guidance,
        error: null,
      });
      if (st.status === "done") break;
    }

    // 3. download and write to disk immediately
    const png = await pod.jobImage(node.job_id!);
    const rel = `${nodeId}.png`;
    fs.writeFileSync(path.join(IMAGES_DIR, rel), png);
    db.updateNode(nodeId, { status: "done", progress: 1, image_path: rel, error: null });
    const s = db.openSession();
    if (s) db.bumpSessionEdits(s.id);
    void pod.deleteJob(node.job_id!);
  } catch (e: any) {
    if (db.getNode(nodeId)) db.updateNode(nodeId, { status: "error", error: e?.message ?? String(e) });
  } finally {
    inflight.delete(nodeId);
  }
}

function sizeFor(node: db.NodeRow): string | null {
  // width/height on the node are a user request (from the inspector); otherwise let the pod
  // default (match input aspect, long side clamped to MAX_SIDE).
  return node.width && node.height ? `${node.width}x${node.height}` : null;
}

// On startup, resume anything that was mid-flight when the local app last exited.
export function resumeUnfinished() {
  for (const n of db.unfinishedNodes()) {
    if (n.job_id) void runEdit(n.id);
    else db.updateNode(n.id, { status: "error", error: "local app restarted before submit; re-run" });
  }
}

// Retry a failed node in place (same params, same seed if it had one).
export function retryEdit(nodeId: string) {
  const n = db.getNode(nodeId);
  if (!n || n.kind !== "edit") throw new Error("not an edit node");
  db.updateNode(nodeId, { status: "queued", progress: 0, error: null, job_id: null, image_path: null });
  podState.lastEditAt = Date.now();
  void runEdit(nodeId);
}
