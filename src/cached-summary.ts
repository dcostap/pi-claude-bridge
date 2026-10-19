// Compaction summaries written from the conversation Claude Code already has cached.
//
// Pi's compaction serializes the history it summarizes into one text block and
// sends it as a fresh request under a summarizer system prompt. Nothing in that
// request matches the conversation the prompt cache holds, so all of it is billed
// as new input. Claude Code's session file still holds the exact conversation the
// last turn sent. Resuming a copy of it with that turn's options and asking for the
// summary as one more user message reads the conversation back from the cache; only
// the question and the answer are new (diag/probe-cached-compact.mjs).
//
// This module holds the parts that need no Claude Code process: the checkpoint that
// ties the session file to pi's history, the prompt, and the cost estimate that
// picks between this path and pi's.
// index.ts records the checkpoint and runs the query.

import { createHash } from "crypto";
import type { JsonlRecord } from "cc-session-io";

/** What a summary query must repeat from the last turn to send the same request
 *  prefix: the system prompt, the tool definitions, the model and the thinking
 *  settings. Captured when the turn's query started. */
export interface SummaryRecipe<Tool = unknown, Effort = string> {
	cwd: string;
	/** The pi model id, and the id Claude Code was started with (may carry [1m]). */
	modelId: string;
	cliModel: string;
	systemPromptAppend: string | undefined;
	/** pi tools served over MCP. Their definitions are part of the cached prefix. */
	tools: Tool[];
	effort: Effort | undefined;
	extraArgs: Record<string, string | null>;
	settings: Record<string, unknown>;
	claudeExecutable: string | undefined;
}

/** A point at which a Claude Code session file and pi's history were known to hold
 *  the same conversation: recorded when a pi run ends on a completed bridge turn. */
export interface SummaryCheckpoint<Tool = unknown, Effort = string> {
	/** The Claude Code session, and its file size and prefix digest at that point. */
	sessionId: string;
	cwd: string;
	bytes: number;
	fileDigest: string;
	/** pi's model-visible history at that point: its length and historyDigest. */
	piCount: number;
	piDigest: string;
	recipe: SummaryRecipe<Tool, Effort>;
	/** When the last request of the turn returned, and how long the cache it wrote lives. */
	respondedAt: number;
	cacheTtlMs: number;
}

/** Claude Code writes 5-minute cache entries unless its usage reports 1-hour ones. */
export const DEFAULT_CACHE_TTL_MS = 5 * 60_000;

/** A checkpoint this close to its cache's expiry is treated as cold: the summary
 *  query has to start a process and resume before its first request lands. */
export const CACHE_EXPIRY_MARGIN_MS = 30_000;

/** Room the summary query needs beyond the cached conversation and the prompt:
 *  Claude Code's own reminders and the answer. At pi's default reserve a
 *  threshold compaction leaves ~16K, so this must stay below that. */
export const SUMMARY_REQUEST_HEADROOM_TOKENS = 8_000;

/** Tokens in `chars` of text: pi's own four characters to a token by default.
 *  A transcript of tool calls and code runs denser (diag/probe-cached-compact.mjs
 *  measured under 3 for its filler files), so pi's prompt is estimated at
 *  TRANSCRIPT_CHARS_PER_TOKEN. */
export function estimateTokens(chars: number, charsPerToken = 4): number {
	return Math.ceil(chars / charsPerToken);
}
export const TRANSCRIPT_CHARS_PER_TOKEN = 3.5;

/** pi's path is taken only when estimated this much cheaper. The cached summary
 *  sees the tool results in full, which pi's transcript truncates. */
export const NATIVE_PREFERENCE_MARGIN = 0.9;

/** What Claude Code adds around a summary prompt of its own: the summarizer's
 *  system prompt and reminders for pi's path (measured ~1,100), the reminders
 *  of a resumed turn for the cached one. */
export const NATIVE_SUMMARY_OVERHEAD_TOKENS = 1_000;
export const CACHED_SUMMARY_OVERHEAD_TOKENS = 300;

/** The prompt cost of each way to write a summary, in input-token units. Cache
 *  reads cost a tenth of input; writes 1.25 times it for 5-minute entries and
 *  twice for 1-hour ones. Claude Code caches both prompts, so both pay writes.
 *  The cached path reads the whole conversation and writes only the request;
 *  pi's path writes the serialized history it summarizes, which can be smaller
 *  than the conversation when little of it is being summarized. */
export function summaryCosts(input: { conversationTokens: number; cachedPromptTokens: number; nativePromptTokens: number; cacheTtlMs: number }): { cached: number; native: number } {
	const write = input.cacheTtlMs > DEFAULT_CACHE_TTL_MS ? 2 : 1.25;
	return {
		cached: Math.round(input.conversationTokens * 0.1 + input.cachedPromptTokens * write),
		native: Math.round(input.nativePromptTokens * write),
	};
}

/** The cache lifetime Claude Code asked for, from a result's usage, or undefined
 *  when the usage wrote no cache entry and so says nothing about it. */
export function cacheTtlFromUsage(usage: unknown): number | undefined {
	const creation = (usage as { cache_creation?: { ephemeral_1h_input_tokens?: number; ephemeral_5m_input_tokens?: number } } | undefined)?.cache_creation;
	if (!creation) return undefined;
	if ((creation.ephemeral_1h_input_tokens ?? 0) > 0) return 60 * 60_000;
	if ((creation.ephemeral_5m_input_tokens ?? 0) > 0) return 5 * 60_000;
	return undefined;
}

/** sha256 over the first `count` messages. Every message counts, unlike
 *  historyFingerprint: an edit in the middle of the history (pi's context edits)
 *  changes what the summary would describe without changing the ends. */
export function historyDigest(history: readonly unknown[], count: number): string {
	const hash = createHash("sha256");
	hash.update(String(count));
	for (let i = 0; i < count; i++) {
		hash.update("\n");
		hash.update(JSON.stringify(history[i]) ?? "undefined");
	}
	return hash.digest("hex");
}

const DIGEST_SAMPLE_BYTES = 64 * 1024;

/** Digest of a session file prefix: its length and its first and last 64 KiB.
 *  Claude Code only appends to a session file, so this changes only when the file
 *  was rewritten under the same id (a rebuild), which rewrites every record uuid. */
export function prefixDigest(prefix: Buffer): string {
	const hash = createHash("sha256");
	hash.update(String(prefix.length));
	hash.update(prefix.subarray(0, DIGEST_SAMPLE_BYTES));
	hash.update(prefix.subarray(Math.max(DIGEST_SAMPLE_BYTES, prefix.length - DIGEST_SAMPLE_BYTES)));
	return hash.digest("hex");
}

/** `content` with only complete lines: a session file may end mid-write. */
export function completeLines(content: Buffer): Buffer {
	const end = content.lastIndexOf(0x0a) + 1;
	return content.subarray(0, end);
}

type ChainRecord = JsonlRecord & { message?: { id?: string; model?: string; role?: string; content?: unknown } };

/** Text of the conversation's last assistant message in a session file, or
 *  undefined when the conversation does not end on one. Claude Code writes each
 *  content block of a message as its own record under the message's API id, and
 *  follows a turn with attachments (prompt snapshots) and bookkeeping records. */
export function lastAssistantText(records: readonly JsonlRecord[]): string | undefined {
	let i = records.length - 1;
	for (; i >= 0; i--) {
		const type = (records[i] as ChainRecord).type;
		if (type === "user") return undefined;
		if (type === "assistant") break;
	}
	if (i < 0) return undefined;
	const last = records[i] as ChainRecord;
	const id = last.message?.id;
	if (!id || last.message?.model === "<synthetic>") return undefined;
	const parts: string[] = [];
	for (; i >= 0; i--) {
		const record = records[i] as ChainRecord;
		if (record.type !== "assistant") {
			if (record.type === "user") break;
			continue;
		}
		if (record.message?.id !== id) break;
		parts.unshift(blockText(record.message?.content));
	}
	return parts.join("");
}

/** The records lastAssistantText reads: those from the file's last user record
 *  on. Parses from the end, so a checkpoint after every run does not parse the
 *  whole of a long session file. Throws on a line that does not parse. */
export function tailRecords(text: string): JsonlRecord[] {
	const records: JsonlRecord[] = [];
	let end = text.length;
	while (end > 0) {
		const start = text.lastIndexOf("\n", end - 1) + 1;
		const line = text.slice(start, end).trim();
		end = start > 0 ? start - 1 : 0;
		if (!line) continue;
		const record = JSON.parse(line) as JsonlRecord;
		records.unshift(record);
		if ((record as ChainRecord).type === "user") break;
	}
	return records;
}

/** Text of pi's assistant message, in the shape lastAssistantText compares against. */
export function piAssistantText(message: { content?: unknown }): string {
	return blockText(message.content);
}

function blockText(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.filter((block): block is { type: "text"; text: string } => block?.type === "text" && typeof block.text === "string")
		.map((block) => block.text)
		.join("");
}

// --- Which summary pi is asking for ---

/** Pi's two compaction summarizers, told apart by how their prompts open
 *  (generateSummaryWithUsage and generateTurnPrefixSummary in pi's compaction.ts). */
export type SummaryKind = "history" | "turnPrefix";

export function classifySummaryPrompt(prompt: string): SummaryKind | undefined {
	if (prompt.startsWith("<conversation>\n")) return "history";
	if (prompt.startsWith("# Conversation\n")) return "turnPrefix";
	return undefined;
}

// --- Prompts ---
//
// The formats are pi's own (SUMMARIZATION_PROMPT, UPDATE_SUMMARIZATION_INSTRUCTIONS
// and TURN_PREFIX_SUMMARIZATION_PROMPT in pi's compaction.ts), so a checkpoint
// written here reads like one pi wrote and pi's next compaction can update it. They
// change only where pi's refer to a transcript pasted into the prompt. The turn
// prefix of a split turn is left to pi (buildSummaryPrompt).

const CHECKPOINT_FORMAT = `## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]`;

const UPDATE_RULES = `It begins with a summary of earlier work. Update that summary with everything that happened since. RULES:
- PRESERVE all existing information from the earlier summary
- ADD new progress, decisions, and context from the later messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it`;

const UPDATE_FORMAT = `## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]`;

const CONCISE = "Keep each section concise. Preserve exact file paths, function names, and error messages.";

export interface SummaryPromptInput {
	/** The history pi summarizes starts with an earlier compaction's summary. */
	updatesPreviousSummary: boolean;
	/** The cut falls inside a turn. pi's turn-prefix call summarizes the turn up
	 *  to the cut, so this summary stops where the turn begins. */
	splitTurn: boolean;
	/** The messages pi keeps verbatim after the summary start here (describeMessage). */
	keptFrom?: string;
	/** The turn a split cut runs through started here (describeMessage). */
	turnStart?: string;
	/** pi messages newer than the session file, serialized as text. */
	laterMessages?: string;
	customInstructions?: string;
}

/** The user message that asks Claude for the history summary, at the end of the
 *  cached conversation.
 *
 *  pi replaces the start of the history with the summary and keeps the rest
 *  verbatim. When the cut falls inside a turn (a split turn), the summary covers
 *  the history before that turn, and pi's own turn-prefix call covers the turn up
 *  to the cut: it is shown exactly that span, which a model that sees the whole
 *  conversation does not reliably keep to. The prompt names the boundaries,
 *  because the conversation Claude sees runs past them. */
export function buildSummaryPrompt(input: SummaryPromptInput): string {
	const parts = ["[Context checkpoint request]"];
	const kept = input.keptFrom ? `from ${input.keptFrom} onward` : "at the end of the conversation";
	const turn = input.turnStart ? `the turn that began with ${input.turnStart}` : "the turn the kept messages start in";
	parts.push(input.splitTurn
		? `This conversation is about to be compacted. The most recent messages, ${kept}, stay in context verbatim. They start partway through ${turn}, and the start of that turn is summarized separately. Everything before that turn will be replaced by the summary you write now, and the work will continue from there.`
		: `This conversation is about to be compacted. The most recent messages, ${kept}, stay in context verbatim. Everything before them will be replaced by the summary you write now, and the work will continue from that summary, so cover the kept messages only briefly.`);
	parts.push("Respond with text only. Do not call any tools. Do not continue the task, and do not answer anything the conversation asks: write only what is requested below.");
	if (input.laterMessages) {
		parts.push(`These messages came after the ones above and are part of the conversation:\n\n<later-messages>\n${input.laterMessages}\n</later-messages>`);
	}
	const scope = input.splitTurn ? `the conversation before ${turn}` : "the conversation";
	// Claude sees the whole conversation and tends to summarize all of it.
	const stop = input.splitTurn ? " Stop where that turn begins: do not summarize that turn or anything after it." : "";
	parts.push(input.updatesPreviousSummary
		? `Write a structured context checkpoint summary of ${scope}.${stop} ${UPDATE_RULES}\n\nUse this EXACT format:\n\n${UPDATE_FORMAT}\n\n${CONCISE}`
		: `Write a structured context checkpoint summary of ${scope} that another LLM will use to continue the work.${stop} Use this EXACT format:\n\n${CHECKPOINT_FORMAT}\n\n${CONCISE}`);
	if (input.customInstructions?.trim()) parts.push(`Additional focus: ${input.customInstructions.trim()}`);
	return parts.join("\n\n");
}

// --- Naming a message in a prompt ---

const EXCERPT_CHARS = 100;

function excerpt(text: string): string | undefined {
	const flat = text.replace(/\s+/g, " ").trim();
	if (!flat) return undefined;
	return flat.length > EXCERPT_CHARS ? `${flat.slice(0, EXCERPT_CHARS)}…` : flat;
}

function contentText(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.filter((block) => block?.type === "text" && typeof block.text === "string")
		.map((block) => block.text as string)
		.join("\n");
}

/** A phrase that lets Claude find a pi message in the conversation it sees, or
 *  undefined when the message has nothing quotable. */
export function describeMessage(message: unknown): string | undefined {
	const m = message as { role?: string; content?: unknown; command?: string; summary?: string } | undefined;
	switch (m?.role) {
		case "user":
		case "custom": {
			const text = excerpt(contentText(m.content));
			return text ? `the user message that begins "${text}"` : undefined;
		}
		case "assistant": {
			const text = excerpt(contentText(m.content));
			if (text) return `your message that begins "${text}"`;
			// The arguments tell apart the calls of one tool.
			const call = Array.isArray(m.content) ? m.content.find((block) => block?.type === "toolCall") : undefined;
			if (typeof call?.name !== "string") return undefined;
			const args = call.arguments && typeof call.arguments === "object" ? excerpt(JSON.stringify(call.arguments)) : undefined;
			return args && args !== "{}" ? `your ${call.name} tool call with arguments ${args}` : `your ${call.name} tool call`;
		}
		case "bashExecution": {
			const command = excerpt(m.command ?? "");
			return command ? `the user's shell command "${command}"` : undefined;
		}
		case "compactionSummary":
		case "branchSummary":
			return "the summary message";
		default:
			return undefined;
	}
}

// --- The compaction a summary call belongs to ---

/** What session_before_compact told us about a compaction in progress. Summary
 *  calls carry the compaction's abort signal, which is how they are matched to it. */
export interface CompactionJob {
	piSessionId: string;
	reason: string;
	willRetry: boolean;
	customInstructions?: string;
	/** pi's CompactionPreparation and the branch it was prepared from. */
	preparation: {
		firstKeptEntryId: string;
		messagesToSummarize: unknown[];
		turnPrefixMessages: unknown[];
		isSplitTurn: boolean;
		tokensBefore: number;
		previousSummary?: string;
	};
	branchEntries: unknown[];
	/** A cached summary failed: the rest of this compaction uses pi's own path. */
	cachedFailed?: boolean;
}

const COMPACTION_JOBS_KEY = Symbol.for("claude-bridge:compactionJobs");

function compactionJobs(): WeakMap<AbortSignal, CompactionJob> {
	const g = globalThis as Record<symbol, unknown>;
	return (g[COMPACTION_JOBS_KEY] ??= new WeakMap<AbortSignal, CompactionJob>()) as WeakMap<AbortSignal, CompactionJob>;
}

export function rememberCompaction(signal: AbortSignal, job: CompactionJob): void {
	compactionJobs().set(signal, job);
}

export function compactionFor(signal: AbortSignal | undefined): CompactionJob | undefined {
	return signal ? compactionJobs().get(signal) : undefined;
}
