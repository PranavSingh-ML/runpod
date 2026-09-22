// Persists connection settings to web/.env.local (gitignored). The browser never sees the raw token.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const WEB_DIR = path.resolve(__dirname, "..");
export const REPO_DIR = path.resolve(WEB_DIR, "..");
export const DATA_DIR = path.join(REPO_DIR, "data");
export const IMAGES_DIR = path.join(DATA_DIR, "images");
const ENV_PATH = path.join(WEB_DIR, ".env.local");

export interface Settings {
  POD_URL: string;
  API_TOKEN: string;
  ANTHROPIC_API_KEY: string;
  POD_RATE_USD_HR: number;
  REWRITE_ENABLED: boolean;
  AUTO_STOP_MIN: number; // 0 = off. Delete the pod after this many idle minutes (no edit submitted).
  POD_IMAGE: string; // image used by the in-app Start button (same default as scripts/pod.sh)
}

const defaults: Settings = {
  POD_URL: "",
  API_TOKEN: "",
  ANTHROPIC_API_KEY: "",
  POD_RATE_USD_HR: 0.49, // A40 Secure Cloud, verified 2026-09-17
  REWRITE_ENABLED: false,
  AUTO_STOP_MIN: 20,
  POD_IMAGE: "ghcr.io/pranavsingh-ml/imgedit:v3", // v3 = Qwen-Image-2.1 + PE-I2I rewriter; :v2 = Qwen-Image-Edit-2511 (Lightning)
};

function parseEnv(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const i = line.indexOf("=");
    if (i < 0) continue;
    let v = line.slice(i + 1).trim();
    if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) v = v.slice(1, -1);
    out[line.slice(0, i).trim()] = v;
  }
  return out;
}

export function loadSettings(): Settings {
  const s: Settings = { ...defaults };
  let env: Record<string, string> = {};
  if (fs.existsSync(ENV_PATH)) env = parseEnv(fs.readFileSync(ENV_PATH, "utf8"));
  // process env wins over file, file wins over defaults
  const get = (k: string) => process.env[k] ?? env[k];
  s.POD_URL = (get("POD_URL") ?? "").replace(/\/+$/, "");
  s.API_TOKEN = get("API_TOKEN") ?? "";
  s.ANTHROPIC_API_KEY = get("ANTHROPIC_API_KEY") ?? "";
  const rate = parseFloat(get("POD_RATE_USD_HR") ?? "");
  if (Number.isFinite(rate) && rate >= 0) s.POD_RATE_USD_HR = rate;
  s.REWRITE_ENABLED = /^(1|true|yes)$/i.test(get("REWRITE_ENABLED") ?? "");
  const autoStop = parseInt(get("AUTO_STOP_MIN") ?? "", 10);
  if (Number.isFinite(autoStop) && autoStop >= 0) s.AUTO_STOP_MIN = autoStop;
  s.POD_IMAGE = (get("IMAGE") ?? get("POD_IMAGE") ?? "").trim() || defaults.POD_IMAGE;
  return s;
}

export let settings: Settings = loadSettings();

export function saveSettings(patch: Partial<Settings>): Settings {
  settings = { ...settings, ...patch };
  settings.POD_URL = settings.POD_URL.trim().replace(/\/+$/, "");
  const lines = [
    "# written by the imgedit settings panel - do not commit",
    `POD_URL=${settings.POD_URL}`,
    `API_TOKEN=${settings.API_TOKEN}`,
    `POD_RATE_USD_HR=${settings.POD_RATE_USD_HR}`,
    `ANTHROPIC_API_KEY=${settings.ANTHROPIC_API_KEY}`,
    `REWRITE_ENABLED=${settings.REWRITE_ENABLED ? "1" : "0"}`,
    `AUTO_STOP_MIN=${settings.AUTO_STOP_MIN}`,
    `POD_IMAGE=${settings.POD_IMAGE}`,
    "",
  ];
  fs.writeFileSync(ENV_PATH, lines.join("\n"), "utf8");
  return settings;
}

export function publicSettings() {
  const mask = (v: string) => (v ? `${v.slice(0, 3)}…${v.slice(-3)}` : "");
  return {
    podUrl: settings.POD_URL,
    hasToken: !!settings.API_TOKEN,
    tokenMasked: mask(settings.API_TOKEN),
    hasAnthropicKey: !!settings.ANTHROPIC_API_KEY,
    anthropicKeyMasked: mask(settings.ANTHROPIC_API_KEY),
    rateUsdHr: settings.POD_RATE_USD_HR,
    rewriteEnabled: settings.REWRITE_ENABLED,
    autoStopMin: settings.AUTO_STOP_MIN,
    podImage: settings.POD_IMAGE,
  };
}
