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
    lora: string | null;
    vramUsedGb: number | null;
    vramTotalGb: number | null;
    queueDepth: number;
    defaults: { steps: number; guidance: number; max_side: number } | null;
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
  longSide: number; // 0 = match input (pod default)
}
