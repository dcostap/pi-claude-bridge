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
// Cached compaction summaries (cached-summary.ts) use the same copy: they resume
// it so the summary request reads the conversation from the prompt cache.
//
// The copy takes fresh uuids, like the SDK's own forkSession: the killed CC
// process may still append to the source file, and the copy must not share
// record ids with it.

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

/**
 * Copy the first `bytes` of a CC session file into a new session and open it for
 * appending. Throws when the source cannot serve as a prefix: missing, shorter
 * than `bytes`, unparseable, or without a message to chain from.
 */
export function forkSessionPrefix(opts: {
	sourceSessionId: string;
	bytes: number;
	cwd: string;
	claudeDir?: string;
	model?: string;
	/** prefixDigest the first `bytes` must have: guards against a file that was
	 *  rewritten under the same id since the size was taken. */
	digest?: string;
}): Session {
	const projectPath = normalizeProjectPath(opts.cwd);
	const sourcePath = getSessionPath(opts.sourceSessionId, projectPath, opts.claudeDir);
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
		cwd: first?.cwd as string | undefined,
		version: first?.version as string | undefined,
		gitBranch: first?.gitBranch as string | undefined,
		model: opts.model,
		records: copied,
		fileExists: true,
	});
}
