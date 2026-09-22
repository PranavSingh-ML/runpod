// SQLite via Node's built-in node:sqlite (Node >= 22.13 / 24). No native build step.
import fs from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { DATA_DIR, IMAGES_DIR } from "./settings.js";

fs.mkdirSync(IMAGES_DIR, { recursive: true });
export const db = new DatabaseSync(path.join(DATA_DIR, "history.sqlite"));
db.exec("PRAGMA journal_mode = WAL");
db.exec(`
CREATE TABLE IF NOT EXISTS nodes (
  id          TEXT PRIMARY KEY,
  parent_id   TEXT REFERENCES nodes(id),
  kind        TEXT NOT NULL,              -- source | edit | ref
  prompt      TEXT,
  negative    TEXT,
  seed        INTEGER,
  steps       INTEGER,
  guidance    REAL,
  width       INTEGER,
  height      INTEGER,
  ref_node_id TEXT,                       -- optional second reference image (image2)
  image_path  TEXT,                       -- relative to data/images
  status      TEXT NOT NULL DEFAULT 'done', -- queued | running | done | error
  progress    REAL NOT NULL DEFAULT 0,
  error       TEXT,
  job_id      TEXT,
  elapsed_s   REAL,
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS nodes_parent ON nodes(parent_id);
CREATE TABLE IF NOT EXISTS sessions (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at  REAL NOT NULL,
  last_seen   REAL NOT NULL,
  ended_at    REAL,
  rate_usd_hr REAL NOT NULL,
  edits       INTEGER NOT NULL DEFAULT 0
);
`);
// v3 (2026-09-21): generation resolution for the qwen_image_21 pipeline. Older DBs lack the column.
try {
  db.exec("ALTER TABLE nodes ADD COLUMN resolution INTEGER");
} catch {
  /* already there */
}

export interface NodeRow {
  id: string;
  parent_id: string | null;
  kind: "source" | "edit" | "ref";
  prompt: string | null;
  negative: string | null;
  seed: number | null;
  steps: number | null;
  guidance: number | null;
  width: number | null;
  height: number | null;
  resolution: number | null; // qwen_image_21: generation pixel budget (resolution^2); null = pod default
  ref_node_id: string | null;
  image_path: string | null;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  error: string | null;
  job_id: string | null;
  elapsed_s: number | null;
  created_at: string;
}

const COLS =
  "id,parent_id,kind,prompt,negative,seed,steps,guidance,width,height,resolution,ref_node_id,image_path,status,progress,error,job_id,elapsed_s,created_at";

export function allNodes(): NodeRow[] {
  return db.prepare(`SELECT ${COLS} FROM nodes ORDER BY created_at ASC`).all() as unknown as NodeRow[];
}

export function getNode(id: string): NodeRow | undefined {
  return db.prepare(`SELECT ${COLS} FROM nodes WHERE id = ?`).get(id) as unknown as NodeRow | undefined;
}

export function insertNode(
  n: Omit<NodeRow, "created_at" | "progress" | "error" | "job_id" | "elapsed_s" | "resolution"> & Partial<NodeRow>,
) {
  db.prepare(
    `INSERT INTO nodes (id,parent_id,kind,prompt,negative,seed,steps,guidance,width,height,resolution,ref_node_id,image_path,status,progress,error,job_id,elapsed_s)
     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
  ).run(
    n.id, n.parent_id, n.kind, n.prompt, n.negative, n.seed, n.steps, n.guidance, n.width, n.height, n.resolution ?? null,
    n.ref_node_id, n.image_path, n.status, n.progress ?? 0, n.error ?? null, n.job_id ?? null, n.elapsed_s ?? null,
  );
  return getNode(n.id)!;
}

export function updateNode(id: string, patch: Partial<NodeRow>) {
  const keys = Object.keys(patch) as (keyof NodeRow)[];
  if (!keys.length) return;
  const sets = keys.map((k) => `${k} = ?`).join(", ");
  db.prepare(`UPDATE nodes SET ${sets} WHERE id = ?`).run(...keys.map((k) => patch[k] as any), id);
}

export function subtreeIds(id: string): string[] {
  const rows = db
    .prepare(
      `WITH RECURSIVE sub(id) AS (SELECT ? UNION ALL SELECT n.id FROM nodes n JOIN sub ON n.parent_id = sub.id)
       SELECT id FROM sub`,
    )
    .all(id) as unknown as { id: string }[];
  return rows.map((r) => r.id);
}

export function deleteNodes(ids: string[]) {
  const stmt = db.prepare("DELETE FROM nodes WHERE id = ?");
  for (const id of ids) stmt.run(id);
}

export function unfinishedNodes(): NodeRow[] {
  return db
    .prepare(`SELECT ${COLS} FROM nodes WHERE status IN ('queued','running')`)
    .all() as unknown as NodeRow[];
}

// ---- sessions (cost meter) --------------------------------------------------
export interface SessionRow {
  id: number;
  started_at: number;
  last_seen: number;
  ended_at: number | null;
  rate_usd_hr: number;
  edits: number;
}

export function openSession(): SessionRow | undefined {
  return db.prepare("SELECT * FROM sessions WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1").get() as unknown as
    | SessionRow
    | undefined;
}
export function startSession(rate: number): SessionRow {
  const now = Date.now() / 1000;
  db.prepare("INSERT INTO sessions (started_at,last_seen,rate_usd_hr) VALUES (?,?,?)").run(now, now, rate);
  return openSession()!;
}
export function touchSession(id: number, rate: number) {
  db.prepare("UPDATE sessions SET last_seen = ?, rate_usd_hr = ? WHERE id = ?").run(Date.now() / 1000, rate, id);
}
export function endSession(id: number, at: number) {
  db.prepare("UPDATE sessions SET ended_at = ? WHERE id = ?").run(at, id);
}
export function bumpSessionEdits(id: number) {
  db.prepare("UPDATE sessions SET edits = edits + 1 WHERE id = ?").run(id);
}
export function totalSpend(): number {
  const rows = db.prepare("SELECT started_at,last_seen,ended_at,rate_usd_hr FROM sessions").all() as unknown as SessionRow[];
  return rows.reduce((acc, r) => acc + (((r.ended_at ?? r.last_seen) - r.started_at) / 3600) * r.rate_usd_hr, 0);
}
