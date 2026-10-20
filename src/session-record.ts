// Session records: which Claude Code session file serves a pi session, kept on
// disk so a new pi process can resume that file instead of rebuilding it.
//
// The bridge keeps this mapping in memory (SessionState in index.ts). A new pi
// process starts with an empty map, so its first turn rebuilt the CC session from
// pi's history. A rebuild drops CC's hidden records and misses the prompt cache
// for the whole conversation. With a record, the first turn resumes the same
// file, and the request matches the cached one.
//
// Two kinds of record:
//   "start"    — written before each query. The file size and the pi history it
//                mirrors at that point: the same data as an abort's fork point.
//                The next process cuts the file back to that size and appends
//                pi's copy of the turn, as after an abort.
//   "complete" — written when a query ends normally. The file then holds pi's
//                history up to `cursor` plus the answer pi appends next, so the
//                next process resumes it as is.
//
// A record is a hint. The reader checks the cwd, the file and the history
// fingerprint, and the caller rebuilds on any mismatch.
//
// Logging-free like convert.ts: functions return what happened and index.ts logs it.

import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "fs";
import { join } from "path";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { getSessionPath, normalizeProjectPath } from "cc-session-io";
import type { ForkPoint } from "./session-fork.js";

export type SessionRecord =
	| ({ kind: "start"; cwd: string } & ForkPoint)
	| { kind: "complete"; cwd: string; sessionId: string; cursor: number; fingerprint: string };

/** Records older than this are removed at startup. Claude Code removes its own
 *  session files after 30 days by default (cleanupPeriodDays). */
const MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000;

export function sessionRecordDir(): string {
	return join(getAgentDir(), "claude-bridge-sessions");
}

function recordPath(piSessionId: string): string {
	return join(sessionRecordDir(), `${piSessionId.replace(/[^A-Za-z0-9_-]/g, "_")}.json`);
}

/** Writes the record through a temporary file, so a reader never sees half of it. */
export function writeSessionRecord(piSessionId: string, record: SessionRecord): void {
	const path = recordPath(piSessionId);
	mkdirSync(sessionRecordDir(), { recursive: true });
	const temp = `${path}.${process.pid}.tmp`;
	writeFileSync(temp, JSON.stringify(record), "utf-8");
	renameSync(temp, path);
}

export function deleteSessionRecord(piSessionId: string): void {
	rmSync(recordPath(piSessionId), { force: true });
}

/** The record for `piSessionId`, or null when there is none or it cannot be read. */
export function readSessionRecord(piSessionId: string): SessionRecord | null {
	let parsed: unknown;
	try {
		parsed = JSON.parse(readFileSync(recordPath(piSessionId), "utf-8"));
	} catch {
		return null;
	}
	return isSessionRecord(parsed) ? parsed : null;
}

function isSessionRecord(value: unknown): value is SessionRecord {
	if (typeof value !== "object" || value === null) return false;
	const r = value as Record<string, unknown>;
	if (typeof r.cwd !== "string" || typeof r.sessionId !== "string" || typeof r.fingerprint !== "string") return false;
	if (r.kind === "start") return typeof r.bytes === "number" && typeof r.piTurnStart === "number";
	if (r.kind === "complete") return typeof r.cursor === "number";
	return false;
}

/** True when the CC session file the record names still exists. */
export function sessionFileExists(record: SessionRecord, claudeDir?: string): boolean {
	return existsSync(getSessionPath(record.sessionId, normalizeProjectPath(record.cwd), claudeDir));
}

/** Removes records not written for MAX_AGE_MS. Returns how many it removed. */
export function pruneSessionRecords(now = Date.now()): number {
	let removed = 0;
	let names: string[];
	try {
		names = readdirSync(sessionRecordDir());
	} catch {
		return 0;
	}
	for (const name of names) {
		const path = join(sessionRecordDir(), name);
		try {
			if (now - statSync(path).mtimeMs > MAX_AGE_MS) {
				rmSync(path, { force: true });
				removed++;
			}
		} catch {
			// Another process removed it first.
		}
	}
	return removed;
}
