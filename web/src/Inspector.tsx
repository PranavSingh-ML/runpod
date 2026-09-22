import { useEffect, useState } from "react";
import { api } from "./api";
import type { NodeRow, Params, PublicSettings, Status } from "./types";

interface Props {
  status: Status | null;
  settings: PublicSettings | null;
  onSettings: (s: PublicSettings) => void;
  params: Params;
  onParams: (p: Params) => void;
  active: NodeRow | null;
  onFlash: (m: string) => void;
}

function fmtDur(s: number) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m.toString().padStart(2, "0")}m` : `${m}m`;
}

export function Inspector({ status, settings, onSettings, params, onParams, active, onFlash }: Props) {
  const pod = status?.pod;
  const sess = status?.session;
  const state = !pod?.configured ? "unset" : !pod.reachable ? "disconnected" : pod.loadError ? "load-error" : !pod.modelLoaded ? "loading" : "ready";
  const label = { unset: "no pod URL", disconnected: "disconnected", "load-error": "model failed", loading: "loading model…", ready: "ready" }[state];
  const set = (patch: Partial<Params>) => onParams({ ...params, ...patch });
  const v21 = pod?.pipeline === "qwen_image_21";
  const d = pod?.defaults;
  const maxSteps = d?.max_steps ?? (v21 ? 60 : 40);
  const maxRes = d?.max_resolution ?? 2048;

  return (
    <div className="inspector">
      <PodPanel status={status} onFlash={onFlash} />

      <section>
        <div className={`conn ${state}`}>
          <span className="dot" />
          <span>{label}</span>
          {pod?.gpu && (
            <span className="dim">
              {" "}
              · {pod.gpu} {pod.capability ? `sm_${pod.capability.join("")}` : ""} · {pod.quant}
            </span>
          )}
        </div>
        {pod?.reachable && pod.vramUsedGb != null && (
          <div className="small dim">
            VRAM {pod.vramUsedGb.toFixed(1)}{pod.vramTotalGb ? ` / ${pod.vramTotalGb}` : ""} GB · queue {pod.queueDepth}
            {pod.lora ? " · lightning" : ""}
            {pod.pipeline ? ` · ${pod.pipeline === "qwen_image_21" ? "Qwen-Image-2.1" : pod.pipeline === "qwen_edit_plus" ? "Edit-2511" : pod.pipeline}` : ""}
            {pod.pipeline === "qwen_image_21" ? (pod.rewriterLoaded ? " · rewriter ✓" : " · rewriter ✗") : ""}
          </div>
        )}
        {pod?.reachable && pod.modelLoaded && pod.pipeline === "qwen_image_21" && !pod.rewriterLoaded && pod.rewriterError && (
          <pre className="small err">rewriter: {pod.rewriterError}</pre>
        )}
        {state === "disconnected" && pod?.lastError && <div className="small err">{pod.lastError}</div>}
        {state === "load-error" && <pre className="small err">{pod?.loadError}</pre>}
      </section>

      <section className="cost">
        <div className="big">
          ${(sess?.costUsd ?? 0).toFixed(2)} <span className="dim small">this session</span>
        </div>
        <div className="small dim">
          {sess?.active ? `${fmtDur(sess.seconds)} at $${sess.rateUsdHr}/hr · ${sess.edits} edits` : "pod not running"} · all-time ≈ $
          {(sess?.totalSpendUsd ?? 0).toFixed(2)}
        </div>
        <button
          className="link"
          onClick={() =>
            api
              .resetSession()
              .then(() => onFlash("Session meter reset"))
              .catch((e) => onFlash(e.message))
          }
        >
          reset meter (new pod)
        </button>
      </section>

      <section>
        <h3>Generation</h3>
        <label>
          steps <span className="val">{params.steps}</span>
          <input type="range" min={1} max={maxSteps} value={params.steps} onChange={(e) => set({ steps: Number(e.target.value) })} />
          <span className="hint">
            {v21 ? "Qwen-Image-2.1 has no Lightning LoRA yet: 40 is the model default, 20-30 for quicker drafts." : "8 for the Lightning LoRA. 20–40 only if LORA_ENABLED=0 on the pod."}
          </span>
        </label>
        <label>
          guidance (true_cfg) <span className="val">{params.guidance.toFixed(1)}</span>
          <input type="range" min={1} max={8} step={0.5} value={params.guidance} onChange={(e) => set({ guidance: Number(e.target.value) })} />
          <span className="hint">{v21 ? "Qwen-Image-2.1 is sampled without guidance: keep 1.0 (>1 doubles the time)." : "1.0 with Lightning (also 2× faster). ~4.0 for the base model."}</span>
        </label>
        <label>
          seed
          <div className="row">
            <input
              type="number"
              value={params.seed}
              min={-1}
              onChange={(e) => set({ seed: Math.max(-1, Math.floor(Number(e.target.value) || 0)) })}
            />
            <button className="btn small" onClick={() => set({ seed: -1 })} title="-1 = random each time">
              random
            </button>
            <button className="btn small" onClick={() => set({ seed: Math.floor(Math.random() * 2 ** 31) })}>
              🎲
            </button>
          </div>
          <span className="hint">{params.seed < 0 ? "random per edit" : `fixed: ${params.seed}`}</span>
        </label>
        <label>
          resolution
          {v21 ? (
            <select value={params.resolution} onChange={(e) => set({ resolution: Number(e.target.value) })}>
              <option value={0}>pod default ({d?.resolution ?? 1024}² px, input aspect)</option>
              <option value={1024}>1024² ≈ 1 MP (draft)</option>
              {maxRes >= 1536 && <option value={1536}>1536² ≈ 2.4 MP</option>}
              {maxRes >= 2048 && <option value={2048}>2048² ≈ 4 MP (native 2K, ~4× slower)</option>}
            </select>
          ) : (
            <select value={params.longSide} onChange={(e) => set({ longSide: Number(e.target.value) })}>
              <option value={0}>match input (long side ≤ {pod?.defaults?.max_side ?? 1024})</option>
              <option value={768}>768 long side (fast)</option>
              <option value={1024}>1024 long side</option>
              <option value={1280}>1280 long side</option>
              <option value={1536}>1536 long side (slow)</option>
            </select>
          )}
          {active?.width && active.height && (
            <span className="hint">
              active source {active.width}×{active.height}
            </span>
          )}
        </label>
      </section>

      <SettingsPanel settings={settings} onSettings={onSettings} onFlash={onFlash} podRewriter={!!pod?.rewriterLoaded} />

      <section className="small dim">
        <p>The pod is disposable: every image is already in <code>data/images/</code>.</p>
        <p>
          <b>Terminate the pod when you're done.</b>
        </p>
      </section>
    </div>
  );
}

function SettingsPanel({
  settings,
  onSettings,
  onFlash,
  podRewriter,
}: {
  settings: PublicSettings | null;
  onSettings: (s: PublicSettings) => void;
  onFlash: (m: string) => void;
  podRewriter: boolean;
}) {
  const [open, setOpen] = useState(!settings?.podUrl);
  const [podUrl, setPodUrl] = useState(settings?.podUrl ?? "");
  const [token, setToken] = useState("");
  const [key, setKey] = useState("");
  const [rate, setRate] = useState(String(settings?.rateUsdHr ?? 0.49));
  const [rewrite, setRewrite] = useState(!!settings?.rewriteEnabled);
  const [autoStop, setAutoStop] = useState(String(settings?.autoStopMin ?? 20));
  const [image, setImage] = useState(settings?.podImage ?? "");
  const [touched, setTouched] = useState(false);

  // sync from the server-loaded settings (they arrive after first render, and change on Start/Stop)
  useEffect(() => {
    if (!settings || touched) return;
    setPodUrl(settings.podUrl);
    setRate(String(settings.rateUsdHr));
    setRewrite(settings.rewriteEnabled);
    setAutoStop(String(settings.autoStopMin));
    setImage(settings.podImage);
  }, [settings, touched]);

  const save = async () => {
    try {
      const s = await api.saveSettings({
        podUrl,
        apiToken: token || undefined,
        anthropicKey: key || undefined,
        rateUsdHr: Number(rate),
        rewriteEnabled: rewrite,
        autoStopMin: Math.max(0, Math.floor(Number(autoStop) || 0)),
        podImage: image.trim() || undefined,
      });
      setTouched(false);
      onSettings(s);
      setToken("");
      setKey("");
      onFlash("Settings saved to web/.env.local");
    } catch (e: any) {
      onFlash(`save failed: ${e.message}`);
    }
  };

  return (
    <section>
      <h3 className="clickable" onClick={() => setOpen(!open)}>
        Connection {open ? "▾" : "▸"}
      </h3>
      {open && (
        <div className="settings">
          <label>
            pod URL
            <input
              placeholder="https://<POD_ID>-8000.proxy.runpod.net"
              value={podUrl}
              onChange={(e) => {
                setTouched(true);
                setPodUrl(e.target.value);
              }}
            />
          </label>
          <label>
            API token {settings?.hasToken && <span className="dim">(set: {settings.tokenMasked})</span>}
            <input type="password" placeholder={settings?.hasToken ? "leave blank to keep" : "same as the pod's API_TOKEN env"} value={token} onChange={(e) => setToken(e.target.value)} />
          </label>
          <label>
            $/hr for the cost meter
            <input value={rate} onChange={(e) => setRate(e.target.value)} />
            <span className="hint">A40 Secure $0.49 · A40 Community $0.35 · L40S $1.09 (RunPod, 2026-09-17)</span>
          </label>
          <label>
            Anthropic API key (optional: text-only rewriter for v2 pods / fallback) {settings?.hasAnthropicKey && <span className="dim">(set: {settings.anthropicKeyMasked})</span>}
            <input type="password" placeholder={settings?.hasAnthropicKey ? "leave blank to keep" : "sk-ant-…"} value={key} onChange={(e) => setKey(e.target.value)} />
          </label>
          <label className="row">
            <input type="checkbox" checked={rewrite} onChange={(e) => setRewrite(e.target.checked)} disabled={!settings?.hasAnthropicKey && !key && !podRewriter} />
            enable the “Enhance / Rewrite” button {podRewriter ? "(pod rewriter loaded)" : ""}
          </label>
          <label>
            auto-stop the pod after (idle minutes, 0 = off)
            <input
              value={autoStop}
              onChange={(e) => {
                setTouched(true);
                setAutoStop(e.target.value);
              }}
            />
            <span className="hint">Idle = no edit submitted since the model finished loading. Deleting the pod is the only thing that stops billing.</span>
          </label>
          <label>
            pod image (used by Start)
            <input
              value={image}
              onChange={(e) => {
                setTouched(true);
                setImage(e.target.value);
              }}
            />
          </label>
          <button className="btn primary" onClick={() => void save()}>
            Save
          </button>
        </div>
      )}
    </section>
  );
}

function fmtCountdown(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m ? `${m}m ${sec.toString().padStart(2, "0")}s` : `${sec}s`;
}

// Start / Stop the RunPod pod from the app (the local server shells out to runpodctl).
// "wait" is the boot-phase line, fed by the health poller.
function PodPanel({ status, onFlash }: { status: Status | null; onFlash: (m: string) => void }) {
  const [gpu, setGpu] = useState("auto");
  const [pending, setPending] = useState(false);
  const c = status?.control;
  const pod = status?.pod;
  const sess = status?.session;
  if (!status || !c) return null;

  const busy = pending || !!c.busy;
  const running = c.pods.length > 0;
  const phase = !pod?.reachable
    ? "booting - image pull + weights (v3 ≈ 52 GB, v2 58 GB), ~6-12 min"
    : pod.loadError
      ? "model failed to load - stop the pod"
      : !pod.modelLoaded
        ? "loading model…"
        : "ready";

  const start = async () => {
    setPending(true);
    try {
      const p = await api.podUp(gpu);
      onFlash(`pod ${p.id} created (${p.gpu}, ${p.cloud}) at $${p.costPerHr}/hr - booting`);
    } catch (e: any) {
      onFlash(`start failed: ${e.message}`);
    } finally {
      setPending(false);
    }
  };
  const stop = async () => {
    if (!confirm("Delete the pod? This stops billing. Every image is already saved in data/images/.")) return;
    setPending(true);
    try {
      const r = await api.podDown();
      onFlash(r.deleted.length ? `deleted ${r.deleted.join(" ")} - billing stopped` : "no imgedit pod was running");
    } catch (e: any) {
      onFlash(`stop failed: ${e.message}`);
    } finally {
      setPending(false);
    }
  };

  return (
    <section className="podctl">
      <h3>Pod</h3>
      {!c.available ? (
        <div className="small dim">
          <code>tools/runpodctl.exe</code> not found - Start/Stop unavailable. Download it (see README), run <code>doctor</code>, restart the app.
          Or use <code>pod.cmd</code> and paste the URL below.
        </div>
      ) : running ? (
        <>
          {c.pods.map((p) => (
            <div key={p.id} className="small">
              <b>{p.id}</b> · ${p.costPerHr}/hr · up {Math.floor(p.uptimeSeconds / 60)} min · {p.status}
            </div>
          ))}
          <div className={`small ${pod?.modelLoaded ? "" : "dim"}`}>{phase}</div>
          <div className="small dim">
            {sess?.autoStopMin
              ? sess.autoStopInSeconds != null
                ? `auto-stop after ${sess.autoStopMin} idle min · stops in ${fmtCountdown(sess.autoStopInSeconds)}`
                : `auto-stop after ${sess.autoStopMin} idle min (counting starts once the model is loaded)`
              : "auto-stop is OFF - remember to stop the pod"}
          </div>
          <button className="btn danger" disabled={busy} onClick={() => void stop()}>
            {c.busy === "stopping" ? "stopping…" : "Stop pod (stops billing)"}
          </button>
        </>
      ) : (
        <>
          <div className="row">
            <select value={gpu} onChange={(e) => setGpu(e.target.value)} disabled={busy}>
              <option value="auto">auto (A40 → A6000)</option>
              <option value="a40">A40 Secure $0.49/hr</option>
              <option value="a6000">RTX A6000 $0.53/hr</option>
            </select>
            <button className="btn primary" disabled={busy} onClick={() => void start()}>
              {c.busy === "starting" ? "creating…" : "Start pod"}
            </button>
          </div>
          <div className="small dim">no pod running - not being billed · image {c.image.replace(/^ghcr\.io\//, "")}</div>
        </>
      )}
      {c.balance != null && (
        <div className="small dim">
          RunPod balance ${c.balance.toFixed(2)}
          {c.spendPerHr ? ` · spending $${c.spendPerHr}/hr` : ""}
        </div>
      )}
      {c.lastAction && <div className="small dim">{c.lastAction}</div>}
      {c.lastError && <div className="small err">{c.lastError}</div>}
    </section>
  );
}
