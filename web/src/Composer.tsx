import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { NodeRow, PublicSettings } from "./types";

interface Props {
  active: NodeRow | null;
  settings: PublicSettings | null;
  refNode: NodeRow | null;
  onClearRef: () => void;
  onAttachRef: (f: File) => void;
  onSubmit: (prompt: string) => Promise<void> | void;
  onFlash: (m: string) => void;
  ready: boolean;
  podRewriter: boolean; // v3 pod has PE-I2I loaded (sees the image); else Claude text-only needs a key
}

export function Composer({ active, settings, refNode, onClearRef, onAttachRef, onSubmit, onFlash, ready, podRewriter }: Props) {
  const [text, setText] = useState("");
  const [rewriting, setRewriting] = useState(false);
  const [rewritten, setRewritten] = useState<string | null>(null);
  const [rewriteSource, setRewriteSource] = useState<"pod" | "claude" | null>(null);
  const ta = useRef<HTMLTextAreaElement>(null);
  const canSend = !!active && active.status === "done" && text.trim().length > 0;
  const rewriterOn = !!settings?.rewriteEnabled && (podRewriter || !!settings?.hasAnthropicKey);

  useEffect(() => {
    ta.current?.focus();
  }, [active?.id]);

  const send = async () => {
    if (!canSend) return;
    const p = text.trim();
    setText("");
    setRewritten(null);
    await onSubmit(p);
  };

  const rewrite = async () => {
    if (!active || !text.trim()) return;
    setRewriting(true);
    try {
      const r = await api.rewrite(active.id, text.trim());
      setRewritten(text);
      setRewriteSource(r.source);
      setText(r.instruction); // shown for editing; never auto-sent
      ta.current?.focus();
    } catch (e: any) {
      onFlash(`rewrite failed: ${e.message}`);
    } finally {
      setRewriting(false);
    }
  };

  return (
    <div className="composer">
      {refNode && (
        <div className="refbar">
          <img src={api.imageUrl(refNode.id)} alt="" />
          <span>reference image attached (sent as image2)</span>
          <button className="link" onClick={onClearRef}>
            remove
          </button>
        </div>
      )}
      {rewritten !== null && (
        <div className="rewrote">
          {rewriteSource === "pod" ? "enhanced by the pod's rewriter (it saw the image)" : "rewritten by Claude"} from “{rewritten}” — edit if needed, then send
          <button
            className="link"
            onClick={() => {
              setText(rewritten);
              setRewritten(null);
            }}
          >
            undo
          </button>
        </div>
      )}
      <div className="row">
        <textarea
          ref={ta}
          value={text}
          placeholder={
            !active
              ? "Upload a source image first"
              : active.status !== "done"
                ? "Waiting for the active image to finish…"
                : ready
                  ? "Describe the edit (Enter to send, Shift+Enter for newline)"
                  : "Pod not ready — you can still queue; it runs once the model is loaded"
          }
          rows={2}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
        />
        <div className="col">
          <button className="btn primary" disabled={!canSend} onClick={() => void send()}>
            Send
          </button>
          {rewriterOn && (
            <button
              className="btn"
              disabled={!canSend || rewriting}
              onClick={() => void rewrite()}
              title={
                podRewriter
                  ? "Enhance on the pod: PE-I2I looks at the image and writes a detailed, creative edit instruction (shown before sending; ~10-30s)"
                  : "Rewrite into a precise edit instruction with Claude (shown before sending)"
              }
            >
              {rewriting ? "…" : podRewriter ? "Enhance" : "Rewrite"}
            </button>
          )}
          <label className="btn small" title="Attach a second reference image (multi-image edit)">
            + ref
            <input type="file" accept="image/*" hidden onChange={(e) => e.target.files?.[0] && onAttachRef(e.target.files[0])} />
          </label>
        </div>
      </div>
    </div>
  );
}
