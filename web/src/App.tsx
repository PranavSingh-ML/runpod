import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import type { NodeRow, Params, PublicSettings, Status } from "./types";
import { Thread } from "./Thread";
import { Composer } from "./Composer";
import { Inspector } from "./Inspector";

const LS_ACTIVE = "imgedit.activeId";
const LS_PARAMS = "imgedit.params";

function loadParams(): Params {
  try {
    const p = JSON.parse(localStorage.getItem(LS_PARAMS) ?? "");
    if (p && typeof p.steps === "number") return p;
  } catch {
    /* ignore */
  }
  return { steps: 8, guidance: 1.0, seed: -1, longSide: 0 };
}

export default function App() {
  const [nodes, setNodes] = useState<NodeRow[]>([]);
  const [activeId, setActiveId] = useState<string | null>(() => localStorage.getItem(LS_ACTIVE));
  const [status, setStatus] = useState<Status | null>(null);
  const [settings, setSettings] = useState<PublicSettings | null>(null);
  const [params, setParams] = useState<Params>(loadParams);
  const [refNode, setRefNode] = useState<NodeRow | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const busyRef = useRef(false);

  const byId = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  const active = activeId ? byId.get(activeId) ?? null : null;
  const anyRunning = nodes.some((n) => n.status === "queued" || n.status === "running");

  const flash = useCallback((m: string) => {
    setToast(m);
    setTimeout(() => setToast((t) => (t === m ? null : t)), 4000);
  }, []);

  const refreshNodes = useCallback(async () => {
    try {
      setNodes(await api.nodes());
    } catch (e: any) {
      flash(`local server: ${e.message}`);
    }
  }, [flash]);

  // initial load
  useEffect(() => {
    void refreshNodes();
    api.settings().then(setSettings).catch(() => {});
    api.status().then(setStatus).catch(() => {});
  }, [refreshNodes]);

  // nodes: 1s while something is running, else 5s
  useEffect(() => {
    const t = setInterval(() => void refreshNodes(), anyRunning ? 1000 : 5000);
    return () => clearInterval(t);
  }, [anyRunning, refreshNodes]);

  // status: 3s (the local server itself polls the pod every 10s)
  useEffect(() => {
    const t = setInterval(() => api.status().then(setStatus).catch(() => {}), 3000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (activeId) localStorage.setItem(LS_ACTIVE, activeId);
  }, [activeId]);
  useEffect(() => {
    localStorage.setItem(LS_PARAMS, JSON.stringify(params));
  }, [params]);

  // if the active node vanished (deleted), fall back to the newest done node
  useEffect(() => {
    if (!nodes.length) return;
    if (activeId && byId.has(activeId)) return;
    const last = [...nodes].reverse().find((n) => n.kind !== "ref");
    setActiveId(last?.id ?? null);
  }, [nodes, activeId, byId]);

  // ---- actions ----------------------------------------------------------------
  const uploadSource = useCallback(
    async (file: File | Blob, name?: string) => {
      if (busyRef.current) return;
      busyRef.current = true;
      try {
        const n = await api.upload(file, "source", name);
        await refreshNodes();
        setActiveId(n.id);
        flash("New source uploaded - type an instruction below");
      } catch (e: any) {
        flash(`upload failed: ${e.message}`);
      } finally {
        busyRef.current = false;
      }
    },
    [refreshNodes, flash],
  );

  const uploadRef = useCallback(
    async (file: File | Blob) => {
      try {
        const n = await api.upload(file, "ref", "reference");
        setRefNode(n);
      } catch (e: any) {
        flash(`reference upload failed: ${e.message}`);
      }
    },
    [flash],
  );

  const sizeFor = useCallback(
    (src: NodeRow | null): string | null => {
      if (!params.longSide || !src?.width || !src?.height) return null;
      const long = Math.max(src.width, src.height);
      const s = params.longSide / long;
      const r16 = (v: number) => Math.max(256, Math.round(v / 16) * 16);
      return `${r16(src.width * s)}x${r16(src.height * s)}`;
    },
    [params.longSide],
  );

  const submitEdit = useCallback(
    async (prompt: string, opts?: { parentId?: string; seed?: number }) => {
      const parentId = opts?.parentId ?? activeId;
      if (!parentId) return flash("Upload or paste a source image first");
      const parent = byId.get(parentId);
      if (!parent || parent.status !== "done") return flash("Wait for the source to finish");
      try {
        const n = await api.edit({
          parentId,
          prompt,
          steps: params.steps,
          guidance: params.guidance,
          seed: opts?.seed ?? params.seed,
          size: sizeFor(parent),
          refNodeId: refNode?.id ?? null,
        });
        setRefNode(null);
        await refreshNodes();
        setActiveId(n.id); // optimistic: the running card becomes the active tip
      } catch (e: any) {
        flash(`edit failed: ${e.message}`);
      }
    },
    [activeId, byId, params, sizeFor, refNode, refreshNodes, flash],
  );

  const reroll = useCallback(
    (n: NodeRow) => {
      if (!n.parent_id || !n.prompt) return;
      void submitEdit(n.prompt, { parentId: n.parent_id, seed: -1 });
    },
    [submitEdit],
  );

  const reuseSeed = useCallback(
    (n: NodeRow) => {
      if (n.seed == null) return;
      setParams((p) => ({ ...p, seed: n.seed! }));
      flash(`Seed ${n.seed} set in the inspector`);
    },
    [flash],
  );

  const retry = useCallback(
    async (n: NodeRow) => {
      try {
        await api.retry(n.id);
        await refreshNodes();
      } catch (e: any) {
        flash(e.message);
      }
    },
    [refreshNodes, flash],
  );

  const remove = useCallback(
    async (n: NodeRow) => {
      try {
        await api.remove(n.id);
        if (activeId === n.id || !byId.has(activeId ?? "")) setActiveId(n.parent_id);
        await refreshNodes();
      } catch (e: any) {
        flash(e.message);
      }
    },
    [activeId, byId, refreshNodes, flash],
  );

  // ---- drag/drop + paste anywhere -------------------------------------------------
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const items = Array.from(e.clipboardData?.items ?? []);
      const img = items.find((i) => i.type.startsWith("image/"));
      if (!img) return;
      const f = img.getAsFile();
      if (f) {
        e.preventDefault();
        void uploadSource(f, "pasted.png");
      }
    };
    const onDragOver = (e: DragEvent) => {
      if (e.dataTransfer?.types.includes("Files")) {
        e.preventDefault();
        setDragging(true);
      }
    };
    const onDragLeave = (e: DragEvent) => {
      if (!e.relatedTarget) setDragging(false);
    };
    const onDrop = (e: DragEvent) => {
      setDragging(false);
      const f = Array.from(e.dataTransfer?.files ?? []).find((x) => x.type.startsWith("image/"));
      if (f) {
        e.preventDefault();
        void uploadSource(f);
      }
    };
    window.addEventListener("paste", onPaste);
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("dragleave", onDragLeave);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("paste", onPaste);
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("dragleave", onDragLeave);
      window.removeEventListener("drop", onDrop);
    };
  }, [uploadSource]);

  const roots = nodes.filter((n) => n.kind === "source");
  const idleMin = status?.session.idleSeconds ? status.session.idleSeconds / 60 : 0;

  return (
    <div className={`app ${dragging ? "dragging" : ""}`}>
      <main className="left">
        <header className="roots">
          <span className="label">Sources</span>
          {roots.map((r) => (
            <button
              key={r.id}
              className={`root-thumb ${isAncestorOrSelf(r.id, active, byId) ? "on" : ""}`}
              title={r.prompt ?? r.id}
              onClick={() => setActiveId(r.id)}
            >
              <img src={api.imageUrl(r.id)} alt="" />
            </button>
          ))}
          <label className="btn small">
            + New source
            <input type="file" accept="image/*" hidden onChange={(e) => e.target.files?.[0] && uploadSource(e.target.files[0])} />
          </label>
          <span className="hint">or drop / paste an image anywhere</span>
        </header>

        {status?.pod.reachable && status.pod.modelLoaded && idleMin >= 15 && (
          <div className="banner warn">
            Pod idle for {Math.floor(idleMin)} min at ${status.session.rateUsdHr}/hr. If you're done, <b>terminate the pod</b> in the RunPod console.
          </div>
        )}

        <Thread
          nodes={nodes}
          byId={byId}
          active={active}
          onSelect={setActiveId}
          onReuseSeed={reuseSeed}
          onReroll={reroll}
          onRetry={retry}
          onDelete={remove}
        />

        <Composer
          active={active}
          settings={settings}
          refNode={refNode}
          onClearRef={() => setRefNode(null)}
          onAttachRef={uploadRef}
          onSubmit={submitEdit}
          onFlash={flash}
          ready={!!status?.pod.modelLoaded}
        />
      </main>

      <aside className="right">
        <Inspector
          status={status}
          settings={settings}
          onSettings={setSettings}
          params={params}
          onParams={setParams}
          active={active}
          onFlash={flash}
        />
      </aside>

      {toast && <div className="toast">{toast}</div>}
      {dragging && <div className="drop-overlay">Drop image to start a new thread</div>}
    </div>
  );
}

function isAncestorOrSelf(rootId: string, n: NodeRow | null, byId: Map<string, NodeRow>): boolean {
  let cur: NodeRow | null | undefined = n;
  while (cur) {
    if (cur.id === rootId) return true;
    cur = cur.parent_id ? byId.get(cur.parent_id) : null;
  }
  return false;
}
