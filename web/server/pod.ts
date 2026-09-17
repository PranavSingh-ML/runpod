// Thin client for the pod's HTTP contract. All calls are short (<30s) so the RunPod /
// Cloudflare 100s proxy timeout never matters; generation itself is async on the pod.
import { settings } from "./settings.js";

export interface Health {
  status: string;
  model_loaded: boolean;
  load_error: string | null;
  gpu: string | null;
  capability: number[] | null;
  quant: string | null;
  model: string;
  lora: string | null;
  vram_used_gb: number;
  vram_total_gb: number | null;
  queue_depth: number;
  uptime_s: number;
  defaults: { steps: number; guidance: number; max_side: number };
}

export interface JobStatus {
  status: "queued" | "running" | "done" | "error";
  progress: number;
  seed: number;
  elapsed_s: number;
  error: string | null;
  width: number;
  height: number;
  steps: number;
  guidance: number;
}

export class PodError extends Error {
  constructor(message: string, public code: number) {
    super(message);
  }
}

function base(): string {
  if (!settings.POD_URL) throw new PodError("POD_URL not set", 0);
  return settings.POD_URL;
}
function auth(): Record<string, string> {
  return settings.API_TOKEN ? { Authorization: `Bearer ${settings.API_TOKEN}` } : {};
}

async function call(path: string, init: RequestInit = {}, timeoutMs = 30_000): Promise<Response> {
  const res = await fetch(base() + path, {
    ...init,
    headers: { ...auth(), ...(init.headers as Record<string, string> | undefined) },
    signal: AbortSignal.timeout(timeoutMs),
  });
  return res;
}

export async function health(): Promise<Health> {
  const res = await fetch(base() + "/health", { signal: AbortSignal.timeout(10_000) });
  if (!res.ok) throw new PodError(`health ${res.status}`, res.status);
  return (await res.json()) as Health;
}

export interface EditParams {
  prompt: string;
  negative?: string | null;
  steps?: number | null;
  guidance?: number | null;
  seed?: number | null;
  size?: string | null;
}

export async function submitEdit(image: Buffer, image2: Buffer | null, p: EditParams) {
  const fd = new FormData();
  fd.append("image", new Blob([new Uint8Array(image)], { type: "image/png" }), "image.png");
  if (image2) fd.append("image2", new Blob([new Uint8Array(image2)], { type: "image/png" }), "image2.png");
  fd.append("prompt", p.prompt);
  if (p.negative != null) fd.append("negative", p.negative);
  if (p.steps != null) fd.append("steps", String(p.steps));
  if (p.guidance != null) fd.append("guidance", String(p.guidance));
  if (p.seed != null) fd.append("seed", String(p.seed));
  if (p.size) fd.append("size", p.size);
  const res = await call("/edit", { method: "POST", body: fd }, 60_000);
  if (res.status !== 202) throw new PodError(`edit ${res.status}: ${(await res.text()).slice(0, 300)}`, res.status);
  return (await res.json()) as { job_id: string; seed: number; width: number; height: number };
}

export async function jobStatus(jobId: string): Promise<JobStatus> {
  const res = await call(`/jobs/${jobId}`, {}, 15_000);
  if (!res.ok) throw new PodError(`job ${res.status}: ${(await res.text()).slice(0, 200)}`, res.status);
  return (await res.json()) as JobStatus;
}

export async function jobImage(jobId: string): Promise<Buffer> {
  const res = await call(`/jobs/${jobId}/image`, {}, 60_000);
  if (!res.ok) throw new PodError(`image ${res.status}`, res.status);
  return Buffer.from(await res.arrayBuffer());
}

export async function deleteJob(jobId: string): Promise<void> {
  try {
    await call(`/jobs/${jobId}`, { method: "DELETE" }, 10_000);
  } catch {
    /* best effort */
  }
}
