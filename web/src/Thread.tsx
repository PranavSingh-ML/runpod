import { useEffect, useRef } from "react";
import { api } from "./api";
import type { NodeRow } from "./types";

interface Props {
  nodes: NodeRow[];
  byId: Map<string, NodeRow>;
  active: NodeRow | null;
  onSelect: (id: string) => void;
  onReuseSeed: (n: NodeRow) => void;
  onReroll: (n: NodeRow) => void;
  onRetry: (n: NodeRow) => void;
  onDelete: (n: NodeRow) => void;
}

// The thread is the path root -> active. Every image is clickable: clicking sets it as the
// active source, which forks the conversation from that point. Siblings (other branches
// off the same parent) are shown as thumbnails under each card.
export function Thread({ nodes, byId, active, onSelect, onReuseSeed, onReroll, onRetry, onDelete }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const path: NodeRow[] = [];
  let cur: NodeRow | null | undefined = active;
  while (cur) {
    path.unshift(cur);
    cur = cur.parent_id ? byId.get(cur.parent_id) : null;
  }
  const childrenOf = (id: string) => nodes.filter((n) => n.parent_id === id);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [active?.id, active?.status]);

  if (!path.length) {
    return (
      <div className="thread empty">
        <div>
          <h2>No source image</h2>
          <p>Drop an image here, paste one from the clipboard, or use “New source”.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="thread">
      {path.map((n, i) => {
        const next = path[i + 1];
        const branches = childrenOf(n.id).filter((c) => c.id !== next?.id);
        return (
          <div key={n.id} className="turn">
            {n.kind === "edit" && (
              <div className="bubble user">
                <span>{n.prompt}</span>
                {n.ref_node_id && byId.get(n.ref_node_id) && (
                  <img className="ref-chip" src={api.imageUrl(n.ref_node_id)} title="reference image" alt="" />
                )}
              </div>
            )}
            <Card
              n={n}
              isTip={n.id === active?.id}
              onSelect={onSelect}
              onReuseSeed={onReuseSeed}
              onReroll={onReroll}
              onRetry={onRetry}
              onDelete={onDelete}
            />
            {branches.length > 0 && (
              <div className="branches">
                <span className="label">{next ? "other branches" : "branches"}</span>
                {branches.map((b) => (
                  <button key={b.id} className={`branch ${b.status}`} title={b.prompt ?? ""} onClick={() => onSelect(b.id)}>
                    {b.status === "done" ? <img src={api.imageUrl(b.id)} alt="" /> : <span className="ph">{b.status}</span>}
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}
      <div ref={endRef} />
    </div>
  );
}

function Card({
  n,
  isTip,
  onSelect,
  onReuseSeed,
  onReroll,
  onRetry,
  onDelete,
}: {
  n: NodeRow;
  isTip: boolean;
  onSelect: (id: string) => void;
  onReuseSeed: (n: NodeRow) => void;
  onReroll: (n: NodeRow) => void;
  onRetry: (n: NodeRow) => void;
  onDelete: (n: NodeRow) => void;
}) {
  const running = n.status === "queued" || n.status === "running";
  return (
    <div className={`card ${n.status} ${isTip ? "tip" : ""}`}>
      <div className="imgwrap" onClick={() => n.status === "done" && onSelect(n.id)} title={n.status === "done" ? "Use as source (fork here)" : ""}>
        {n.status === "done" && n.image_path ? (
          <img src={api.imageUrl(n.id)} alt="" />
        ) : (
          <div className="placeholder">
            {running && (
              <>
                <div className="bar">
                  <div style={{ width: `${Math.round((n.progress ?? 0) * 100)}%` }} />
                </div>
                <div className="small">
                  {n.status === "queued" ? "queued on pod…" : `generating ${Math.round((n.progress ?? 0) * 100)}%`}
                  {n.elapsed_s ? ` · ${n.elapsed_s.toFixed(1)}s` : ""}
                </div>
                {n.error && <div className="small dim">{n.error}</div>}
              </>
            )}
            {n.status === "error" && (
              <>
                <div className="err">error</div>
                <pre className="small">{n.error}</pre>
              </>
            )}
          </div>
        )}
      </div>
      <div className="meta">
        {n.kind === "source" ? (
          <span className="dim">source · {n.width && n.height ? `${n.width}×${n.height}` : "?"}</span>
        ) : (
          <>
            <span className="dim">
              {n.width && n.height ? `${n.width}×${n.height}` : ""} · {n.steps ?? "?"} steps · cfg {n.guidance ?? "?"}
              {n.elapsed_s && n.status === "done" ? ` · ${n.elapsed_s.toFixed(1)}s` : ""}
            </span>
            <span className="seed">
              seed <b>{n.seed ?? "…"}</b>
              <button className="link" disabled={n.seed == null} onClick={() => onReuseSeed(n)} title="Put this seed in the inspector">
                reuse
              </button>
              <button className="link" disabled={running} onClick={() => onReroll(n)} title="Same prompt, same parent, new random seed">
                reroll
              </button>
              {n.status === "error" && (
                <button className="link" onClick={() => onRetry(n)}>
                  retry
                </button>
              )}
              <button className="link danger" disabled={running} onClick={() => confirm("Delete this result and everything under it?") && onDelete(n)}>
                delete
              </button>
            </span>
          </>
        )}
        {isTip && n.status === "done" && <span className="pill tip">active source</span>}
      </div>
    </div>
  );
}
