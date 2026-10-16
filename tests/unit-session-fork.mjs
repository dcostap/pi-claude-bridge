/**
 * After an abort, the next sync cuts Claude Code's session file back to where the
 * aborted query started instead of rebuilding it from pi's history. These tests
 * pin the cut (src/session-fork.ts) and the FORK path in syncSharedSession.
 */
import { describe, it, afterEach } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { appendFileSync, mkdtempSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createSession, openSession } from "cc-session-io";
import { forkSessionPrefix, historyFingerprint } from "../src/session-fork.js";

const { __test } = await import("../src/index.js");

const T = 1_700_000_000_000;
const hi = { role: "user", content: "Hi", timestamp: T };
const hello = { role: "assistant", content: [{ type: "text", text: "Hello." }], provider: "claude-bridge", stopReason: "stop", timestamp: T + 1 };
const doX = { role: "user", content: "Do X", timestamp: T + 2 };
const partial = { role: "assistant", content: [{ type: "text", text: "Starting X" }], provider: "claude-bridge", stopReason: "aborted", timestamp: T + 3 };

/** A session holding [Hi, Hello] plus CC-only records, then an aborted turn.
 *  Returns the size before the aborted turn. */
function seedAbortedSession(cwd) {
	const sessionId = randomUUID();
	const s = createSession({ sessionId, projectPath: cwd });
	s.addUserMessage("Hi");
	s.addAttachment({ type: "date", content: "2026-10-16" });
	s.addAssistantMessage([{ type: "text", text: "Hello." }]);
	s.dangerousAppendRecord({ type: "last-prompt", lastPrompt: "Hi", sessionId });
	s.save();
	const bytes = statSync(s.jsonlPath).size;
	s.addUserMessage("Do X");
	s.addAssistantMessage([{ type: "text", text: "Starting X" }]);
	s.save();
	return { sessionId, bytes, jsonlPath: s.jsonlPath };
}

/** Walk parentUuid links from the leaf; returns the chain root-first. */
function chain(records) {
	const byUuid = new Map(records.filter((r) => r.uuid).map((r) => [r.uuid, r]));
	const out = [];
	for (let r = [...records].reverse().find((x) => x.uuid); r; r = byUuid.get(r.parentUuid)) out.unshift(r);
	return out;
}

describe("forkSessionPrefix", () => {
	it("copies the prefix under a new id with fresh, consistent uuids", () => {
		const cwd = mkdtempSync(join(tmpdir(), "session-fork-"));
		try {
			const src = seedAbortedSession(cwd);
			const source = openSession({ sessionId: src.sessionId, projectPath: cwd });
			const forked = forkSessionPrefix({ sourceSessionId: src.sessionId, bytes: src.bytes, cwd });

			assert.notEqual(forked.sessionId, src.sessionId);
			assert.deepEqual(forked.records.map((r) => r.type), ["user", "attachment", "assistant", "last-prompt"]);
			assert.ok(forked.records.every((r) => r.sessionId === forked.sessionId), "every record names the new session");
			const sourceUuids = new Set(source.records.map((r) => r.uuid).filter(Boolean));
			assert.ok(forked.records.every((r) => !sourceUuids.has(r.uuid)), "no uuid is shared with the source");
			assert.deepEqual(chain(forked.records).map((r) => r.type), ["user", "attachment", "assistant"], "the chain survives the remap");

			forked.importMessages([{ role: "user", content: "Next" }]);
			forked.save();
			const reread = openSession({ sessionId: forked.sessionId, projectPath: cwd });
			assert.deepEqual(chain(reread.records).map((r) => r.type), ["user", "attachment", "assistant", "user"], "an import chains onto the prefix leaf");
		} finally {
			rmSync(cwd, { recursive: true, force: true });
		}
	});

	it("keeps only complete lines when the size falls inside a record", () => {
		const cwd = mkdtempSync(join(tmpdir(), "session-fork-"));
		try {
			const src = seedAbortedSession(cwd);
			const forked = forkSessionPrefix({ sourceSessionId: src.sessionId, bytes: src.bytes + 10, cwd });
			assert.equal(forked.records.length, 4);
		} finally {
			rmSync(cwd, { recursive: true, force: true });
		}
	});

	it("throws when the source is shorter than the cut", () => {
		const cwd = mkdtempSync(join(tmpdir(), "session-fork-"));
		try {
			const src = seedAbortedSession(cwd);
			assert.throws(() => forkSessionPrefix({ sourceSessionId: src.sessionId, bytes: statSync(src.jsonlPath).size + 1, cwd }), /bytes/);
		} finally {
			rmSync(cwd, { recursive: true, force: true });
		}
	});
});

describe("syncSharedSession after an abort", () => {
	afterEach(() => __test.resetSharedSession());

	function arm(cwd, src, piHistory) {
		__test.setSharedSession(null, {
			sessionId: src.sessionId, cursor: 2, cwd, needsRebuild: true, forceRotate: true,
			forkPoint: { sessionId: src.sessionId, bytes: src.bytes, piTurnStart: 2, fingerprint: historyFingerprint(piHistory, 2), cwd },
		});
	}

	it("cuts the file back and appends pi's record of the aborted turn", () => {
		const cwd = mkdtempSync(join(tmpdir(), "session-fork-"));
		try {
			const src = seedAbortedSession(cwd);
			// The killed process writes late; the cut must not see it.
			appendFileSync(src.jsonlPath, JSON.stringify({ type: "user", uuid: randomUUID(), parentUuid: null, sessionId: src.sessionId, message: { role: "user", content: "[Request interrupted by user]" } }) + "\n");
			arm(cwd, src, [hi, hello]);

			const result = __test.syncSharedSession([hi, hello, doX, partial, { role: "user", content: "Next", timestamp: T + 4 }], cwd);

			assert.ok(result.sessionId && result.sessionId !== src.sessionId, "the fork takes a new id");
			const state = __test.getSharedSession(null);
			assert.equal(state.sessionId, result.sessionId);
			assert.equal(state.cursor, 4);
			assert.equal(state.forkPoint, undefined, "a fork point is used once");
			assert.ok(!state.needsRebuild && !state.forceRotate);
			const forked = openSession({ sessionId: result.sessionId, projectPath: cwd });
			assert.deepEqual(
				chain(forked.records).map((r) => `${r.type}:${r.type === "attachment" ? r.attachment.type : JSON.stringify(r.message.content).slice(0, 60)}`),
				['user:"Hi"', "attachment:date", 'assistant:[{"type":"text","text":"Hello."}]', 'user:"Do X"', 'assistant:[{"type":"text","text":"Starting X"}]'],
				"CC's own records stay, pi's aborted turn follows",
			);
		} finally {
			rmSync(cwd, { recursive: true, force: true });
		}
	});

	it("appends nothing when pi dropped the aborted turn (edit last prompt)", () => {
		const cwd = mkdtempSync(join(tmpdir(), "session-fork-"));
		try {
			const src = seedAbortedSession(cwd);
			arm(cwd, src, [hi, hello]);
			const result = __test.syncSharedSession([hi, hello, { role: "user", content: "Do Y", timestamp: T + 5 }], cwd);
			const forked = openSession({ sessionId: result.sessionId, projectPath: cwd });
			assert.equal(forked.records.length, 4, "the prefix only");
			assert.equal(__test.getSharedSession(null).cursor, 2);
		} finally {
			rmSync(cwd, { recursive: true, force: true });
		}
	});

	it("rebuilds when pi's history no longer starts with the cut prefix", () => {
		const cwd = mkdtempSync(join(tmpdir(), "session-fork-"));
		try {
			const src = seedAbortedSession(cwd);
			arm(cwd, src, [hi, hello]);
			const summary = { role: "user", content: "Summary of earlier work", timestamp: T + 6 };
			const result = __test.syncSharedSession([summary, hello, { role: "user", content: "Next", timestamp: T + 7 }], cwd);
			assert.ok(result.sessionId && result.sessionId !== src.sessionId);
			const rebuilt = openSession({ sessionId: result.sessionId, projectPath: cwd });
			assert.deepEqual(rebuilt.records.map((r) => r.type), ["user", "assistant"], "no CC-only records: this is a rebuild");
			assert.equal(__test.getSharedSession(null).forkPoint, undefined);
		} finally {
			rmSync(cwd, { recursive: true, force: true });
		}
	});
});
