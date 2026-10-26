// Cut a Claude Code session file back to an earlier size, under a new session id.
//
// After an abort the bridge used to rebuild the whole session from pi's history.
// A rebuild drops the records pi never sees (CC's hidden `environment`, `model`,
// `date` and similar attachments), so the request diverges at messages[0] and the
// prompt cache misses for the whole conversation. Claude Code only appends to its
// session file, so the bytes it held before the aborted query are still the exact
// history it sent. Copying that prefix keeps the cached request byte-identical up
// to the aborted turn (diag/probe-fork-fidelity.mjs).
//
// The cut does not stop where the aborted query started. It keeps every tool
// round of that query which pi also holds in full (liveExtent). A rebuilt round
// differs from the live one: it lacks CC's hidden attachments, and the API drops
// thinking blocks whose prefix changed. The API also finds a cached prefix only
// about 20 content blocks back from a request's cache breakpoint. So a long
// query that pi re-appends from its start misses the whole conversation.
//
// Cached compaction summaries (cached-summary.ts) use the same copy: they resume
// it so the summary request reads the conversation from the prompt cache.
//
// The copy takes fresh uuids, like the SDK's own forkSession: the killed CC
// process may still append to the source file, and the copy must not share
// record ids with it.
//
// The copy can go to another project directory, when the pi session moved to
// another cwd. It keeps the recorded system prompt, which then carries the old
// directory's project context until compaction. Claude Code appends the new
// working directory itself. We chose the prompt cache over fresh instructions.

import { createHash, randomUUID } from "crypto";
import { mkdirSync, readFileSync, writeFileSync } from "fs";
import { dirname } from "path";
import { Session, getSessionPath, normalizeProjectPath, type JsonlRecord } from "cc-session-io";
import { prefixDigest } from "./cached-summary.js";

/** Where an aborted query started: enough to cut its session file back. */
export interface ForkPoint {
	/** The CC session the aborted query resumed and appended to. */
	sessionId: string;
	/** Size of that session file before the aborted query wrote to it. */
	bytes: number;
	/** Length of pi's history (system messages excluded) the prefix mirrors. */
	piTurnStart: number;
	/** historyFingerprint of pi's history at that length. */
	fingerprint: string;
}

/**
 * Identifies pi's history up to `length`. A tree path to a node is unique, so the
 * last message fixes the whole prefix after a tree navigation. The first message
 * catches a compaction, which replaces the start and shifts the rest.
 */
export function historyFingerprint(history: readonly unknown[], length: number): string {
	const hash = createHash("sha256");
	hash.update(String(length));
	if (length > 0) {
		hash.update(JSON.stringify(history[0]));
		hash.update(JSON.stringify(history[length - 1]));
	}
	return hash.digest("hex");
}

// Top-level record fields that hold the uuid of another record.
const UUID_FIELDS = ["uuid", "parentUuid", "logicalParentUuid", "sourceToolAssistantUUID", "leafUuid", "messageId"];

const isChainRecord = (r: JsonlRecord) => r.type === "user" || r.type === "assistant" || r.type === "attachment";

/** How much of an aborted query's live session file can stay. */
export interface LiveExtent {
	/** Where to cut the file: at or after the query's start. */
	bytes: number;
	/** How many of pi's messages from the query's start those bytes hold. */
	piMessages: number;
}

type Block = { type?: string; id?: string; tool_use_id?: string; text?: string };
type PiMessage = { role?: string; content?: unknown; toolCallId?: string };

const blocksOf = (content: unknown): Block[] => (Array.isArray(content) ? content : []);

/** A user turn as text, images as markers: how a steer is compared across the two sides. */
function promptText(content: unknown): string {
	if (typeof content === "string") return content;
	return blocksOf(content).map((b) => (b.type === "text" ? b.text ?? "" : `[${b.type}]`)).join("\n");
}

/**
 * Walk the records an aborted query wrote after `start` and match them, one tool
 * round at a time, to pi's messages from the query's start (`piTail`). A round is
 * one API response with tool calls, its tool results, and the attachments CC wrote
 * with them. It matches when pi holds the same calls (by id) and all their results,
 * and the same steers CC drained as `queued_command` attachments.
 *
 * Returns the end of the last round that matched, or `start` when none did. The
 * walk stops at anything else: a response without tool calls, a missing result,
 * the interrupt marker CC writes on abort, a partial line.
 */
export function liveExtent(source: Buffer, start: number, piTail: readonly unknown[]): LiveExtent {
	const none: LiveExtent = { bytes: start, piMessages: 0 };
	const tail = piTail as readonly PiMessage[];

	// Chain records after `start`, with where each line ends. A cut after a record
	// keeps the bookkeeping lines (queue operations, last-prompt) before it.
	const records: { record: JsonlRecord; end: number }[] = [];
	let offset = start;
	while (offset < source.length) {
		const nl = source.indexOf(0x0a, offset);
		if (nl === -1) break;
		const line = source.subarray(offset, nl).toString("utf8").trim();
		offset = nl + 1;
		if (!line) continue;
		let record: JsonlRecord;
		try {
			record = JSON.parse(line) as JsonlRecord;
		} catch {
			break;
		}
		if (isChainRecord(record)) records.push({ record, end: offset });
	}
	const message = (r: JsonlRecord) => (r as { message?: { id?: string; model?: string; content?: unknown } }).message;
	const isToolResults = (r: JsonlRecord) => r.type === "user" && blocksOf(message(r)?.content).some((b) => b.type === "tool_result");

	// The query's prompt: pi's leading user messages, CC's first user record and
	// the attachments CC wrote with it. A resumed file that ended on a user record
	// gets CC's own "No response requested." before the prompt.
	let p = 0;
	while (p < tail.length && tail[p].role === "user") p++;
	let k = 0;
	while (records[k]?.record.type === "assistant" && message(records[k].record)?.model === "<synthetic>") k++;
	if (p === 0 || records[k]?.record.type !== "user" || isToolResults(records[k].record)) return none;
	k++;
	while (records[k]?.record.type === "attachment") k++;

	let extent = none;
	while (records[k]?.record.type === "assistant") {
		// The response: CC writes one record per content block.
		const id = message(records[k].record)?.id;
		const calls: string[] = [];
		for (; records[k]?.record.type === "assistant" && message(records[k].record)?.id === id; k++) {
			for (const b of blocksOf(message(records[k].record)?.content)) if (b.type === "tool_use" && b.id) calls.push(b.id);
		}
		const piCalls = tail[p]?.role === "assistant" ? blocksOf(tail[p].content).filter((b) => b.type === "toolCall").map((b) => b.id) : [];
		if (calls.length === 0 || piCalls.join("\n") !== calls.join("\n")) break;

		// CC's side of the results: tool results and attachments, steers among them.
		const results = new Set<string>();
		const steers: string[] = [];
		let end = -1;
		for (; records[k] && (records[k].record.type === "attachment" || isToolResults(records[k].record)); k++) {
			const { record } = records[k];
			if (record.type === "user") for (const b of blocksOf(message(record)?.content)) if (b.tool_use_id) results.add(b.tool_use_id);
			const attachment = (record as { attachment?: { type?: string; prompt?: unknown } }).attachment;
			if (attachment?.type === "queued_command") steers.push(promptText(attachment.prompt));
			end = records[k].end;
		}
		// pi's side: the results, then any steers.
		const piResults = new Set<string>();
		const piSteers: string[] = [];
		let q = p + 1;
		for (; tail[q]?.role === "toolResult" || tail[q]?.role === "user"; q++) {
			if (tail[q].role === "toolResult") piResults.add(tail[q].toolCallId ?? "");
			else piSteers.push(promptText(tail[q].content));
		}
		const answered = (ids: Set<string>) => ids.size === calls.length && calls.every((c) => ids.has(c));
		if (!answered(results) || !answered(piResults) || steers.join("\n") !== piSteers.join("\n")) break;
		extent = { bytes: end, piMessages: q };
		p = q;
	}
	return extent;
}

/**
 * Copy the first `bytes` of a CC session file into a new session and open it for
 * appending. Throws when the source cannot serve as a prefix: missing, shorter
 * than `bytes`, unparseable, or without a message to chain from.
 */
export function forkSessionPrefix(opts: {
	sourceSessionId: string;
	bytes: number;
	cwd: string;
	/** Where the source file lives, when it is not `cwd`. */
	sourceCwd?: string;
	claudeDir?: string;
	model?: string;
	/** prefixDigest the first `bytes` must have: guards against a file that was
	 *  rewritten under the same id since the size was taken. */
	digest?: string;
}): Session {
	const projectPath = normalizeProjectPath(opts.cwd);
	const sourcePath = getSessionPath(opts.sourceSessionId, normalizeProjectPath(opts.sourceCwd ?? opts.cwd), opts.claudeDir);
	const source = readFileSync(sourcePath);
	if (source.length < opts.bytes) {
		throw new Error(`source session is ${source.length} bytes, expected at least ${opts.bytes}`);
	}
	if (opts.digest !== undefined && prefixDigest(source.subarray(0, opts.bytes)) !== opts.digest) {
		throw new Error("source session was rewritten since the cut point was taken");
	}
	// The size was read while the previous CC process could still be writing its
	// trailing bookkeeping records. Keep only complete lines.
	const prefix = source.subarray(0, opts.bytes).toString("utf8");
	const end = prefix.lastIndexOf("\n") + 1;
	const lines = prefix.slice(0, end).split("\n").filter((line) => line.trim().length > 0);
	const records = lines.map((line) => JSON.parse(line) as JsonlRecord);
	if (!records.some(isChainRecord)) throw new Error("source prefix has no message to chain from");

	const sessionId = randomUUID();
	const uuidMap = new Map<string, string>();
	for (const r of records) {
		if (typeof r.uuid === "string") uuidMap.set(r.uuid, randomUUID());
	}
	const copied = records.map((r) => {
		const out: Record<string, unknown> = { ...r };
		if ("sessionId" in out) out.sessionId = sessionId;
		for (const field of UUID_FIELDS) {
			const value = out[field];
			if (typeof value === "string" && uuidMap.has(value)) out[field] = uuidMap.get(value);
		}
		return out as JsonlRecord;
	});

	const jsonlPath = getSessionPath(sessionId, projectPath, opts.claudeDir);
	mkdirSync(dirname(jsonlPath), { recursive: true });
	writeFileSync(jsonlPath, copied.map((r) => JSON.stringify(r) + "\n").join(""), "utf-8");

	const first = copied.find((r) => r.type === "user" || r.type === "assistant") as Record<string, unknown> | undefined;
	return new Session({
		sessionId,
		projectPath,
		jsonlPath,
		slug: first?.slug as string | undefined,
		cwd: opts.sourceCwd === undefined ? first?.cwd as string | undefined : opts.cwd,
		version: first?.version as string | undefined,
		gitBranch: first?.gitBranch as string | undefined,
		model: opts.model,
		records: copied,
		fileExists: true,
	});
}
