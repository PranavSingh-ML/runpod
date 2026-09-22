// Optional prompt rewriter (spec section 5). Turns a casual chat message plus the last few
// turns into one precise, literal edit instruction. The result is shown to the user for
// editing - the app never sends a silently-rewritten prompt.
//
// Two backends:
//   pod    (v3) Qwen-Image-2.1-PE-I2I on the pod: SEES the image, so it can invent details that fit
//          it (the Grok-Imagine style enhancement). Used whenever the pod reports rewriter_loaded.
//   claude text-only Haiku with the edit history; the v2 fallback and the safety net if the pod fails.
import fs from "node:fs";
import path from "node:path";
import Anthropic from "@anthropic-ai/sdk";
import * as db from "./db.js";
import * as pod from "./pod.js";
import { podState } from "./jobs.js";
import { IMAGES_DIR, settings } from "./settings.js";

const POD_REWRITE_TIMEOUT_MS = 180_000; // thinking mode can run a couple of thousand tokens on a 9B model
const POLL_MS = 1000;

export interface Rewrite {
  instruction: string;
  source: "pod" | "claude";
  thinking?: string;
}

export function podRewriterAvailable(): boolean {
  return !!(podState.reachable && podState.health?.model_loaded && podState.health?.rewriter_loaded);
}

async function rewriteOnPod(parentId: string, message: string): Promise<Rewrite> {
  const parent = db.getNode(parentId);
  if (!parent?.image_path) throw new Error("active image has no file yet");
  const image = fs.readFileSync(path.join(IMAGES_DIR, parent.image_path));
  const { job_id } = await pod.submitRewrite(image, null, message);
  podState.lastEditAt = Date.now(); // counts as activity for auto-stop
  const t0 = Date.now();
  try {
    for (;;) {
      await new Promise((r) => setTimeout(r, POLL_MS));
      const st = await pod.jobStatus(job_id);
      if (st.status === "error") throw new Error(st.error?.split("\n").pop() ?? "pod rewrite failed");
      if (st.status === "done") {
        const text = st.result?.rewritten_prompt?.trim();
        if (!text) throw new Error("pod rewrite returned no prompt");
        return { instruction: text, source: "pod", thinking: st.result?.thinking };
      }
      if (Date.now() - t0 > POD_REWRITE_TIMEOUT_MS) throw new Error("pod rewrite timed out");
    }
  } finally {
    void pod.deleteJob(job_id);
  }
}

export async function rewritePrompt(parentId: string, message: string): Promise<Rewrite> {
  if (podRewriterAvailable()) {
    try {
      return await rewriteOnPod(parentId, message);
    } catch (e: any) {
      if (!settings.ANTHROPIC_API_KEY) throw e;
      console.warn(`pod rewriter failed (${e?.message ?? e}); falling back to Claude`);
    }
  }
  return { instruction: await rewriteWithClaude(parentId, message), source: "claude" };
}

// The spec asks for a cheap text model; override with REWRITE_MODEL if you want more quality.
const MODEL = process.env.REWRITE_MODEL ?? "claude-haiku-4-5";

const SYSTEM = `You convert casual chat messages into precise instructions for an image-editing model (Qwen-Image-Edit).
The editing model is literal: it edits exactly what it is told and preserves everything else.

Rules:
- Output ONE instruction, 1-3 sentences, imperative mood, no preamble, no quotes, no options.
- Name the subject explicitly (e.g. "the woman in the red coat", "the text on the sign").
- Name the change explicitly (what is added / removed / replaced / recoloured / moved).
- Say what must stay unchanged (identity, pose, composition, lighting, other objects) when relevant.
- Resolve pronouns and references like "it", "that", "the same" using the conversation history.
- If the user refers to a previous edit, phrase the instruction relative to the CURRENT image, not the original.
- Do not invent details the user did not ask for. Do not add style words unless asked.`;

async function rewriteWithClaude(parentId: string, message: string): Promise<string> {
  if (!settings.ANTHROPIC_API_KEY) throw new Error("no rewriter: the pod has none loaded and ANTHROPIC_API_KEY is not set");
  const client = new Anthropic({ apiKey: settings.ANTHROPIC_API_KEY });

  // last few turns along the path root -> parent
  const chain: db.NodeRow[] = [];
  let cur = db.getNode(parentId);
  while (cur) {
    chain.unshift(cur);
    cur = cur.parent_id ? db.getNode(cur.parent_id) : undefined;
  }
  const turns = chain
    .filter((n) => n.kind === "edit" && n.prompt)
    .slice(-6)
    .map((n, i) => `${i + 1}. ${n.prompt}`)
    .join("\n");

  const user =
    (turns ? `Previous edit instructions applied to reach the current image (oldest first):\n${turns}\n\n` : "The current image is the user's original upload.\n\n") +
    `User's new message: ${message}\n\nWrite the single edit instruction.`;

  try {
    const res = await client.messages.create({
      model: MODEL,
      max_tokens: 400,
      system: SYSTEM,
      messages: [{ role: "user", content: user }],
    });
    const text = res.content
      .filter((b): b is Anthropic.TextBlock => b.type === "text")
      .map((b) => b.text)
      .join("")
      .trim();
    if (!text) throw new Error("empty rewrite");
    return text.replace(/^["'`]+|["'`]+$/g, "");
  } catch (e) {
    if (e instanceof Anthropic.AuthenticationError) throw new Error("Anthropic API key rejected");
    if (e instanceof Anthropic.RateLimitError) throw new Error("Anthropic rate limit - try again in a moment");
    if (e instanceof Anthropic.APIError) throw new Error(`Anthropic API error ${e.status}: ${e.message}`);
    throw e;
  }
}
