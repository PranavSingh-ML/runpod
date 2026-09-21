import type { NodeRow, PodInfo, PublicSettings, Status } from "./types";

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `${res.status}`;
    try {
      msg = ((await res.json()) as any).error ?? msg;
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
  return (await res.json()) as T;
}

export const api = {
  settings: () => fetch("/api/settings").then((r) => j<PublicSettings>(r)),
  saveSettings: (patch: Record<string, unknown>) =>
    fetch("/api/settings", { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(patch) }).then((r) =>
      j<PublicSettings>(r),
    ),
  status: () => fetch("/api/status").then((r) => j<Status>(r)),
  resetSession: () => fetch("/api/session/reset", { method: "POST" }).then((r) => j<unknown>(r)),
  podUp: (gpu: string) =>
    fetch("/api/pod/up", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ gpu }) }).then((r) =>
      j<PodInfo & { gpu: string; cloud: string }>(r),
    ),
  podDown: () => fetch("/api/pod/down", { method: "POST" }).then((r) => j<{ deleted: string[] }>(r)),
  nodes: () => fetch("/api/nodes").then((r) => j<NodeRow[]>(r)),
  upload: (file: File | Blob, kind: "source" | "ref" = "source", name?: string) => {
    const fd = new FormData();
    fd.append("image", file, name ?? (file instanceof File ? file.name : "pasted.png"));
    fd.append("kind", kind);
    if (name) fd.append("name", name);
    return fetch("/api/upload", { method: "POST", body: fd }).then((r) => j<NodeRow>(r));
  },
  edit: (body: {
    parentId: string;
    prompt: string;
    negative?: string | null;
    steps?: number;
    guidance?: number;
    seed?: number;
    size?: string | null;
    refNodeId?: string | null;
  }) =>
    fetch("/api/edit", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }).then((r) =>
      j<NodeRow>(r),
    ),
  retry: (id: string) => fetch(`/api/nodes/${id}/retry`, { method: "POST" }).then((r) => j<NodeRow>(r)),
  remove: (id: string) => fetch(`/api/nodes/${id}`, { method: "DELETE" }).then((r) => j<{ deleted: string[] }>(r)),
  rewrite: (parentId: string, message: string) =>
    fetch("/api/rewrite", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ parentId, message }) }).then(
      (r) => j<{ instruction: string }>(r),
    ),
  imageUrl: (id: string) => `/api/images/${id}`,
};
