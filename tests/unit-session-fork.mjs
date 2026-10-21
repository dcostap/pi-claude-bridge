/**
 * After an abort, the next sync cuts Claude Code's session file back to where the
 * aborted query started instead of rebuilding it from pi's history. These tests
 * pin the cut (src/session-fork.ts) and the FORK path in syncSharedSession.
 */
import { describe, it, afterEach } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { appendFileSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createSession, openSession } from "cc-session-io";
import { forkSessionPrefix, historyFingerprint, liveExtent } from "../src/session-fork.js";

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

// --- An aborted query with tool rounds, as Claude Code writes it ---
//
// CC writes one assistant record per content block, all with the API message
// id, then one user record per tool result, then attachments such as the
// total_tokens_reminder. pi holds one assistant message per response and one
// toolResult per call.

const piUser = (text) => ({ role: "user", content: text, timestamp: T });
const piCall = (...ids) => ({ role: "assistant", content: [{ type: "thinking", thinking: "hm", thinkingSignature: "sig" }, ...ids.map((id) => ({ type: "toolCall", id, name: "bash", arguments: {} }))], provider: "claude-bridge", stopReason: "toolUse", timestamp: T });
const piResult = (id) => ({ role: "toolResult", toolCallId: id, toolName: "bash", content: [{ type: "text", text: `out ${id}` }], isError: false, timestamp: T });

/** Raw CC records for a query, chained by uuid; `end` marks a cut. */
function ccLog(sessionId, parent = null) {
	const lines = [];
	let n = 0;
	const push = (record) => {
		const uuid = randomUUID();
		lines.push(JSON.stringify({ parentUuid: parent, uuid, sessionId, timestamp: new Date(T + n++).toISOString(), ...record }));
		parent = uuid;
	};
	const log = {
		lines,
		prompt(text) { push({ type: "user", message: { role: "user", content: text } }); return log; },
		synthetic(text) { push({ type: "assistant", message: { id: randomUUID(), model: "<synthetic>", role: "assistant", content: [{ type: "text", text }] } }); return log; },
		response(...ids) {
			const id = `msg_${randomUUID()}`;
			push({ type: "assistant", message: { id, role: "assistant", content: [{ type: "thinking", thinking: "hm", signature: "sig" }] } });
			for (const tool of ids) push({ type: "assistant", message: { id, role: "assistant", content: [{ type: "tool_use", id: tool, name: "mcp__custom-tools__bash", input: {} }] } });
			return log;
		},
		result(id) { push({ type: "user", message: { role: "user", content: [{ type: "tool_result", tool_use_id: id, content: `out ${id}` }] } }); return log; },
		attachment(attachment) { push({ type: "attachment", attachment }); return log; },
		reminder() { return log.attachment({ type: "total_tokens_reminder" }); },
		bookkeeping(type) { lines.push(JSON.stringify({ type, sessionId })); return log; },
		interrupt() { push({ type: "user", message: { role: "user", content: [{ type: "text", text: "[Request interrupted by user]" }] } }); return log; },
		get text() { return lines.map((l) => l + "\n").join(""); },
	};
	return log;
}

/** The byte offset after the first `count` lines of `log`. */
const bytesAfter = (log, count) => Buffer.byteLength(log.lines.slice(0, count).map((l) => l + "\n").join(""));

describe("liveExtent", () => {
	const sid = "s";

	it("keeps every round pi holds in full, and stops at a call without its result", () => {
		const log = ccLog(sid).prompt("Go").reminder()
			.response("a").result("a").reminder()
			.response("b").result("b").reminder();
		const kept = log.lines.length;
		log.bookkeeping("last-prompt").response("c"); // the process died running c
		const pi = [piUser("Go"), piCall("a"), piResult("a"), piCall("b"), piResult("b"), piCall("c")];

		assert.deepEqual(liveExtent(Buffer.from(log.text), 0, pi), { bytes: bytesAfter(log, kept), piMessages: 5 });
	});

	it("cuts before the interrupt marker an abort writes", () => {
		const log = ccLog(sid).prompt("Go").response("a").result("a").reminder();
		const kept = log.lines.length;
		log.bookkeeping("mode").interrupt();
		const pi = [piUser("Go"), piCall("a"), piResult("a"), { role: "assistant", content: [], provider: "claude-bridge", stopReason: "aborted", timestamp: T }];

		assert.deepEqual(liveExtent(Buffer.from(log.text), 0, pi), { bytes: bytesAfter(log, kept), piMessages: 3 });
	});

	it("starts at the given offset and skips CC's synthetic reply before the prompt", () => {
		const before = ccLog(sid).prompt("Earlier").response("x").result("x");
		const log = ccLog(sid).synthetic("No response requested.").prompt("Go").response("a").result("a").reminder();
		const start = Buffer.byteLength(before.text);
		const source = Buffer.from(before.text + log.text);

		assert.deepEqual(liveExtent(source, start, [piUser("Go"), piCall("a"), piResult("a")]), { bytes: source.length, piMessages: 3 });
	});

	it("needs every result of a parallel call on both sides", () => {
		const log = ccLog(sid).prompt("Go").response("a", "b").result("a").result("b").reminder();
		const source = Buffer.from(log.text);

		assert.equal(liveExtent(source, 0, [piUser("Go"), piCall("a", "b"), piResult("b"), piResult("a")]).piMessages, 4, "order does not matter");
		assert.equal(liveExtent(source, 0, [piUser("Go"), piCall("a", "b"), piResult("a")]).piMessages, 0, "pi lacks b's result");
		const ccLacksB = ccLog(sid).prompt("Go").response("a", "b").result("a");
		assert.equal(liveExtent(Buffer.from(ccLacksB.text), 0, [piUser("Go"), piCall("a", "b"), piResult("a"), piResult("b")]).piMessages, 0, "CC lacks b's result");
	});

	it("keeps a round with a steer only when CC drained the same steer", () => {
		const log = ccLog(sid).prompt("Go")
			.response("a").result("a").reminder()
			.response("b").bookkeeping("queue-operation").result("b")
			.attachment({ type: "queued_command", prompt: [{ type: "text", text: "also check y" }] }).reminder()
			.response("c").result("c");
		const source = Buffer.from(log.text);
		const pi = (steer) => [piUser("Go"), piCall("a"), piResult("a"), piCall("b"), piResult("b"), piUser(steer), piCall("c"), piResult("c")];

		assert.deepEqual(liveExtent(source, 0, pi("also check y")), { bytes: source.length, piMessages: 8 });
		assert.equal(liveExtent(source, 0, pi("something else")).piMessages, 3, "a different steer stops before its round");
		const undelivered = [piUser("Go"), piCall("a"), piResult("a"), piUser("never reached CC"), piCall("b"), piResult("b")];
		assert.equal(liveExtent(Buffer.from(ccLog(sid).prompt("Go").response("a").result("a").response("b").result("b").text), 0, undelivered).piMessages, 0);
	});

	it("stops at calls pi does not hold, a response without calls, and a partial line", () => {
		const log = ccLog(sid).prompt("Go").response("a").result("a");
		const source = Buffer.from(log.text);
		assert.equal(liveExtent(source, 0, [piUser("Go"), piCall("z"), piResult("z")]).piMessages, 0, "other ids");
		assert.equal(liveExtent(source, 0, [piCall("a"), piResult("a")]).piMessages, 0, "no prompt on pi's side");
		const cut = Buffer.from(log.text.slice(0, -20));
		assert.equal(liveExtent(cut, 0, [piUser("Go"), piCall("a"), piResult("a")]).piMessages, 0, "the result line is incomplete");
		const text = ccLog(sid).prompt("Go");
		text.lines.push(JSON.stringify({ type: "assistant", uuid: randomUUID(), message: { id: "m", role: "assistant", content: [{ type: "text", text: "Done" }] } }));
		assert.equal(liveExtent(Buffer.from(text.text), 0, [piUser("Go"), { role: "assistant", content: [{ type: "text", text: "Done" }] }]).piMessages, 0);
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

	it("keeps the aborted query's live rounds and appends only what follows them", () => {
		const cwd = mkdtempSync(join(tmpdir(), "session-fork-"));
		try {
			const src = seedAbortedSession(cwd);
			// Replace seedAbortedSession's aborted turn with a live query of two rounds.
			const full = readFileSync(src.jsonlPath);
			writeFileSync(src.jsonlPath, full.subarray(0, src.bytes));
			const leaf = chain(openSession({ sessionId: src.sessionId, projectPath: cwd }).records).at(-1).uuid;
			const log = ccLog(src.sessionId, leaf).prompt("Do X").response("a").result("a").reminder().response("b").result("b").reminder();
			log.response("c").interrupt();
			appendFileSync(src.jsonlPath, log.text);
			arm(cwd, src, [hi, hello]);

			const aborted = { role: "assistant", content: [], provider: "claude-bridge", stopReason: "aborted", timestamp: T };
			const history = [hi, hello, doX, piCall("a"), piResult("a"), piCall("b"), piResult("b"), piCall("c"), aborted, { role: "user", content: "Next", timestamp: T + 4 }];
			const result = __test.syncSharedSession(history, cwd);

			assert.equal(__test.getSharedSession(null).cursor, 9);
			const forked = openSession({ sessionId: result.sessionId, projectPath: cwd });
			const describe = (r) => r.type === "attachment" ? `attachment:${r.attachment.type}`
				: `${r.type}:${r.message.content.map?.((b) => b.type).join("+") ?? "text"}`;
			assert.deepEqual(chain(forked.records).slice(3).map(describe), [
				"user:text",
				"assistant:thinking", "assistant:tool_use", "user:tool_result", "attachment:total_tokens_reminder",
				"assistant:thinking", "assistant:tool_use", "user:tool_result", "attachment:total_tokens_reminder",
				"assistant:thinking+tool_use", "user:tool_result",
			], "CC's two live rounds stay, with their attachments; pi appends only c, paired with a result");
			const live = forked.records.filter((r) => r.type === "assistant" && String(r.message.id).startsWith("msg_") && !String(r.message.id).startsWith("msg_syn"));
			assert.equal(live.length, 4, "the kept rounds keep CC's own message ids");
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
