// Local app server. One process: Express API + Vite (dev middleware or built dist/).
// Owns all laptop-side state: data/history.sqlite and data/images/.
import express, { type Request, type Response, type NextFunction } from "express";
import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
import multer from "multer";
import * as db from "./db.js";
import * as jobs from "./jobs.js";
import { rewritePrompt } from "./rewrite.js";
import { IMAGES_DIR, WEB_DIR, publicSettings, saveSettings, settings } from "./settings.js";

const PORT = Number(process.env.PORT ?? 5173);
const app = express();
app.use(express.json({ limit: "1mb" }));
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 40 * 1024 * 1024 } });

const wrap =
  (fn: (req: Request, res: Response) => Promise<unknown> | unknown) =>
  (req: Request, res: Response, next: NextFunction) =>
    Promise.resolve(fn(req, res)).catch(next);

// ---- settings ----------------------------------------------------------------
app.get("/api/settings", (_req, res) => res.json(publicSettings()));
app.put(
  "/api/settings",
  wrap(async (req, res) => {
    const b = req.body ?? {};
    const patch: Partial<typeof settings> = {};
    if (typeof b.podUrl === "string") patch.POD_URL = b.podUrl;
    if (typeof b.apiToken === "string" && b.apiToken !== "") patch.API_TOKEN = b.apiToken;
    if (typeof b.anthropicKey === "string" && b.anthropicKey !== "") patch.ANTHROPIC_API_KEY = b.anthropicKey;
    if (b.clearAnthropicKey === true) patch.ANTHROPIC_API_KEY = "";
    if (typeof b.rateUsdHr === "number" && b.rateUsdHr >= 0) patch.POD_RATE_USD_HR = b.rateUsdHr;
    if (typeof b.rewriteEnabled === "boolean") patch.REWRITE_ENABLED = b.rewriteEnabled;
    saveSettings(patch);
    await jobs.pokeHealth();
    res.json(publicSettings());
  }),
);

// ---- status (pod health + cost meter) ----------------------------------------
app.get("/api/status", (_req, res) => {
  const h = jobs.podState.health;
  res.json({
    pod: {
      configured: !!settings.POD_URL,
      reachable: jobs.podState.reachable,
      modelLoaded: !!h?.model_loaded,
      loadError: h?.load_error ?? null,
      gpu: h?.gpu ?? null,
      capability: h?.capability ?? null,
      quant: h?.quant ?? null,
      model: h?.model ?? null,
      lora: h?.lora ?? null,
      vramUsedGb: h?.vram_used_gb ?? null,
      vramTotalGb: h?.vram_total_gb ?? null,
      queueDepth: h?.queue_depth ?? 0,
      defaults: h?.defaults ?? null,
      lastError: jobs.podState.lastError,
    },
    session: jobs.sessionInfo(),
  });
});
app.post("/api/session/reset", (_req, res) => {
  jobs.resetSession();
  res.json(jobs.sessionInfo());
});

// ---- nodes -------------------------------------------------------------------
app.get("/api/nodes", (_req, res) => res.json(db.allNodes()));
app.get("/api/nodes/:id", (req, res) => {
  const n = db.getNode(req.params.id as string);
  if (!n) return res.status(404).json({ error: "not found" });
  res.json(n);
});

// upload a source (new thread root) or a reference image (kind=ref)
app.post(
  "/api/upload",
  upload.single("image"),
  wrap(async (req, res) => {
    if (!req.file) throw new Error("no image");
    const kind = req.body?.kind === "ref" ? "ref" : "source";
    const id = randomUUID().replace(/-/g, "").slice(0, 16);
    const rel = `${id}.png`;
    // keep the bytes as uploaded; the pod decodes any PIL-readable format
    fs.writeFileSync(path.join(IMAGES_DIR, rel), req.file.buffer);
    const dims = pngDims(req.file.buffer);
    const node = db.insertNode({
      id,
      parent_id: null,
      kind,
      prompt: req.body?.name ?? req.file.originalname ?? null,
      negative: null,
      seed: null,
      steps: null,
      guidance: null,
      width: dims?.w ?? null,
      height: dims?.h ?? null,
      ref_node_id: null,
      image_path: rel,
      status: "done",
      progress: 1,
    });
    res.json(node);
  }),
);

app.post(
  "/api/edit",
  wrap(async (req, res) => {
    const b = req.body ?? {};
    if (typeof b.parentId !== "string" || typeof b.prompt !== "string" || !b.prompt.trim())
      return res.status(400).json({ error: "parentId and prompt required" });
    if (!settings.POD_URL) return res.status(400).json({ error: "POD_URL not set - open Settings" });
    const node = jobs.createEdit({
      parentId: b.parentId,
      prompt: b.prompt.trim(),
      negative: typeof b.negative === "string" ? b.negative : null,
      steps: Number.isFinite(b.steps) ? Number(b.steps) : null,
      guidance: Number.isFinite(b.guidance) ? Number(b.guidance) : null,
      seed: Number.isFinite(b.seed) ? Number(b.seed) : null,
      size: typeof b.size === "string" ? b.size : null,
      refNodeId: typeof b.refNodeId === "string" ? b.refNodeId : null,
    });
    res.status(202).json(node);
  }),
);

app.post("/api/nodes/:id/retry", (req, res) => {
  try {
    jobs.retryEdit(req.params.id as string);
    res.json(db.getNode(req.params.id as string));
  } catch (e: any) {
    res.status(400).json({ error: e.message });
  }
});

app.delete("/api/nodes/:id", (req, res) => {
  const ids = db.subtreeIds(req.params.id as string);
  for (const id of ids) {
    const n = db.getNode(id);
    if (n?.image_path) fs.rmSync(path.join(IMAGES_DIR, n.image_path), { force: true });
  }
  db.deleteNodes(ids.reverse());
  res.json({ deleted: ids });
});

app.get("/api/images/:id", (req, res) => {
  const n = db.getNode(req.params.id as string);
  if (!n?.image_path) return res.status(404).end();
  res.setHeader("Cache-Control", "private, max-age=31536000, immutable");
  res.sendFile(path.join(IMAGES_DIR, n.image_path));
});

app.post(
  "/api/rewrite",
  wrap(async (req, res) => {
    const { parentId, message } = req.body ?? {};
    if (typeof parentId !== "string" || typeof message !== "string") return res.status(400).json({ error: "bad request" });
    res.json({ instruction: await rewritePrompt(parentId, message) });
  }),
);

// ---- errors ------------------------------------------------------------------
app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
  console.error(err);
  res.status(500).json({ error: err?.message ?? String(err) });
});

// ---- frontend ----------------------------------------------------------------
async function main() {
  if (process.env.NODE_ENV === "production") {
    const dist = path.join(WEB_DIR, "dist");
    if (!fs.existsSync(dist)) throw new Error("dist/ missing - run `npm run build` first");
    app.use(express.static(dist));
    app.get("/{*splat}", (_req, res) => res.sendFile(path.join(dist, "index.html")));
  } else {
    const { createServer } = await import("vite");
    const vite = await createServer({ root: WEB_DIR, server: { middlewareMode: true }, appType: "spa" });
    app.use(vite.middlewares);
  }
  jobs.startHealthLoop();
  jobs.resumeUnfinished();
  app.listen(PORT, "127.0.0.1", () => {
    console.log(`imgedit web  ->  http://127.0.0.1:${PORT}`);
    console.log(`pod: ${settings.POD_URL || "(not set - open Settings in the app)"}`);
  });
}

function pngDims(buf: Buffer): { w: number; h: number } | null {
  // PNG IHDR
  if (buf.length > 24 && buf.toString("ascii", 1, 4) === "PNG") return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
  // JPEG SOFn
  if (buf.length > 4 && buf[0] === 0xff && buf[1] === 0xd8) {
    let i = 2;
    while (i + 9 < buf.length) {
      if (buf[i] !== 0xff) return null;
      const marker = buf[i + 1];
      const len = buf.readUInt16BE(i + 2);
      if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc)
        return { h: buf.readUInt16BE(i + 5), w: buf.readUInt16BE(i + 7) };
      i += 2 + len;
    }
  }
  return null;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
