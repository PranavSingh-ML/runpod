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
  resolution: number | null;
  ref_node_id: string | null;
  image_path: string | null;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  error: string | null;
  job_id: string | null;
  elapsed_s: number | null;
  created_at: string;
}

export interface Status {
  pod: {
    configured: boolean;
    reachable: boolean;
    modelLoaded: boolean;
    loadError: string | null;
    gpu: string | null;
    capability: number[] | null;
    quant: string | null;
    model: string | null;
    pipeline: string | null; // qwen_image_21 (v3) | qwen_edit_plus (v2) | null (unknown / old pod)
    lora: string | null;
    rewriterLoaded: boolean;
    rewriterError: string | null;
    vramUsedGb: number | null;
    vramTotalGb: number | null;
    queueDepth: number;
    defaults: { steps: number; guidance: number; max_side: number; resolution?: number; max_resolution?: number; max_steps?: number } | null;
    lastError: string | null;
  };
  session: {
    active: boolean;
    startedAt: number | null;
    seconds: number;
    rateUsdHr: number;
    costUsd: number;
    edits: number;
    totalSpendUsd: number;
    idleSeconds: number;
    autoStopMin: number;
    autoStopInSeconds: number | null;
  };
  control: {
    available: boolean;
    pods: PodInfo[];
    balance: number | null;
    spendPerHr: number | null;
    busy: "starting" | "stopping" | null;
    lastAction: string | null;
    lastError: string | null;
    image: string;
  };
}

export interface PodInfo {
  id: string;
  name: string;
  status: string;
  costPerHr: number;
  uptimeSeconds: number;
  url: string;
}

export interface PublicSettings {
  podUrl: string;
  hasToken: boolean;
  tokenMasked: string;
  hasAnthropicKey: boolean;
  anthropicKeyMasked: string;
  rateUsdHr: number;
  rewriteEnabled: boolean;
  autoStopMin: number;
  podImage: string;
}

export interface Params {
  steps: number;
  guidance: number;
  seed: number; // -1 = random
  longSide: number; // v2 (qwen_edit_plus): post-resize long side; 0 = match input (pod default)
  resolution: number; // v3 (qwen_image_21): generation budget 1024/1536/2048; 0 = pod default
  pipeline?: string; // which pod pipeline steps/guidance were last defaulted from
}
