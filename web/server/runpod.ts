// Pod lifecycle from inside the app: a TypeScript port of scripts/pod.sh. Shells out to
// runpodctl (which holds the RunPod API key in ~/.runpod/config.toml) so the app never
// sees the key. Every call here is short; boot progress comes from the health poller.
import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { promisify } from "node:util";
import { REPO_DIR } from "./settings.js";

const execFileP = promisify(execFile);

export interface PodInfo {
  id: string;
  name: string;
  status: string; // desiredStatus as reported by runpodctl (RUNNING | EXITED | ...)
  costPerHr: number;
  uptimeSeconds: number;
  url: string;
}

// GPU fallback order, same as pod.sh: 48 GB Ampere cards only (see NOTES.md for the quant path).
export const GPU_CHOICES: Record<string, { gpu: string; cloud: "SECURE" | "COMMUNITY" }[]> = {
  auto: [
    { gpu: "NVIDIA A40", cloud: "SECURE" },
    { gpu: "NVIDIA RTX A6000", cloud: "SECURE" },
    { gpu: "NVIDIA RTX A6000", cloud: "COMMUNITY" },
  ],
  a40: [{ gpu: "NVIDIA A40", cloud: "SECURE" }],
  a6000: [
    { gpu: "NVIDIA RTX A6000", cloud: "SECURE" },
    { gpu: "NVIDIA RTX A6000", cloud: "COMMUNITY" },
  ],
};

export function runpodctlPath(): string | null {
  const candidates = [
    process.env.RUNPODCTL,
    path.join(REPO_DIR, "tools", "runpodctl.exe"),
    path.join(REPO_DIR, "tools", "runpodctl"),
  ].filter((p): p is string => !!p);
  for (const p of candidates) if (fs.existsSync(p)) return p;
  return null; // not on PATH lookup on purpose: keep it explicit, like pod.sh
}

async function rp(args: string[], timeoutMs = 60_000): Promise<string> {
  const bin = runpodctlPath();
  if (!bin) throw new Error("runpodctl not found in tools/ - download it from github.com/runpod/runpodctl/releases and run `tools\\runpodctl.exe doctor`");
  const { stdout, stderr } = await execFileP(bin, args, { timeout: timeoutMs, maxBuffer: 4 * 1024 * 1024 });
  return (stdout || "").trim() || (stderr || "").trim();
}

function parseJson(text: string): any {
  // runpodctl occasionally prefixes JSON with a log line; take the first { or [.
  const i = Math.min(...["{", "["].map((c) => text.indexOf(c)).filter((n) => n >= 0));
  if (!Number.isFinite(i)) throw new Error(`runpodctl: not JSON: ${text.slice(0, 200)}`);
  return JSON.parse(text.slice(i));
}

export async function listPods(): Promise<PodInfo[]> {
  const out = parseJson((await rp(["pod", "list", "-o", "json"])) || "[]");
  const arr: any[] = Array.isArray(out) ? out : [];
  return arr.map((p) => ({
    id: String(p.id),
    name: String(p.name ?? ""),
    status: String(p.desiredStatus ?? ""),
    costPerHr: Number(p.costPerHr ?? 0),
    uptimeSeconds: Number(p.uptimeSeconds ?? 0),
    url: `https://${p.id}-8000.proxy.runpod.net`,
  }));
}

export async function imgeditPods(): Promise<PodInfo[]> {
  return (await listPods()).filter((p) => p.name.startsWith("imgedit"));
}

export async function balance(): Promise<{ balance: number | null; spendPerHr: number | null }> {
  try {
    const u = parseJson((await rp(["user", "-o", "json"], 20_000)) || "{}");
    return {
      balance: typeof u.clientBalance === "number" ? u.clientBalance : null,
      spendPerHr: typeof u.currentSpendPerHr === "number" ? u.currentSpendPerHr : null,
    };
  } catch {
    return { balance: null, spendPerHr: null };
  }
}

export interface CreateOpts {
  token: string;
  image: string;
  diskGb: number;
  choice: keyof typeof GPU_CHOICES;
  extraEnv?: Record<string, string>;
  registryAuthId?: string;
}

export async function createPod(o: CreateOpts): Promise<PodInfo & { gpu: string; cloud: string }> {
  const attempts = GPU_CHOICES[o.choice] ?? GPU_CHOICES.auto;
  const env = JSON.stringify({ API_TOKEN: o.token, HF_HOME: "/workspace/hf", ...(o.extraEnv ?? {}) });
  const errors: string[] = [];
  for (const a of attempts) {
    const args = [
      "pod", "create", "--name", "imgedit", "--image", o.image, "--gpu-id", a.gpu, "--cloud-type", a.cloud,
      "--container-disk-in-gb", String(o.diskGb), "--ports", "8000/http,22/tcp", "--env", env, "-o", "json",
    ];
    if (o.registryAuthId) args.push("--registry-auth-id", o.registryAuthId);
    let text = "";
    try {
      text = await rp(args, 120_000);
    } catch (e: any) {
      text = String(e?.stdout || e?.stderr || e?.message || e);
    }
    let j: any = null;
    try {
      j = parseJson(text);
    } catch {
      /* not JSON */
    }
    if (j?.id) {
      return {
        id: String(j.id),
        name: "imgedit",
        status: "RUNNING",
        costPerHr: Number(j.costPerHr ?? 0),
        uptimeSeconds: 0,
        url: `https://${j.id}-8000.proxy.runpod.net`,
        gpu: a.gpu,
        cloud: a.cloud,
      };
    }
    errors.push(`${a.gpu} ${a.cloud}: ${String(j?.error ?? text).slice(0, 160)}`);
  }
  throw new Error(`no 48GB Ampere card available right now - try again in a few minutes.\n${errors.join("\n")}`);
}

export async function deletePod(id: string): Promise<void> {
  await rp(["pod", "delete", id, "-o", "json"], 60_000);
}
