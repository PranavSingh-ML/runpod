// Optional prompt rewriter (spec section 5). Turns a casual chat message plus the last few
// turns into one precise, literal edit instruction. The result is shown to the user for
// editing - the app never sends a silently-rewritten prompt.
import Anthropic from "@anthropic-ai/sdk";
import * as db from "./db.js";
import { settings } from "./settings.js";

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

export async function rewritePrompt(parentId: string, message: string): Promise<string> {
  if (!settings.ANTHROPIC_API_KEY) throw new Error("ANTHROPIC_API_KEY not set");
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
