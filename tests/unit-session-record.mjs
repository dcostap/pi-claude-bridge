/**
 * Session records let a new pi process resume the Claude Code session file of
 * a pi session it reopens, instead of rebuilding it from pi's history.
 *
 * A rebuild drops CC's hidden records, so the request misses the prompt cache
 * for the whole conversation on the first turn after every restart. These pin
 * the record's life: a query that ends normally is resumed as is, one that
 * never ended is cut back to where it began, and any mismatch rebuilds.
 *
 * "Restart" here is forgetSessionMirrors(): the in-memory mirrors go, the
 * records on disk stay, which is what a new process sees.
 */
import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { createSession, getSessionPath, normalizeProjectPath, openSession } from "cc-session-io";

const claudeDir = mkdtempSync(join(tmpdir(), "claude-bridge-session-record-cc-"));
process.env.CLAUDE_CONFIG_DIR = claudeDir;
process.on("exit", () => rmSync(claudeDir, { recursive: true, force: true }));

const mod = await import("../src/index.js");
const { readSessionRecord, sessionRecordDir } = await import("../src/session-record.js");
const { setQuery, resetSharedSession, forgetSessionMirrors, activeQueryContexts, getSharedSession } = mod.__test;
let providerConfig;
mod.default({
	on: () => {},
	registerProvider: (_name, config) => { providerConfig = config; },
	registerTool: () => {},
});
const streamSimple = providerConfig.streamSimple;
const model = providerConfig.models[0];
const cwd = process.cwd();
const ccPath = (id) => getSessionPath(id, normalizeProjectPath(cwd), claudeDir);

const calls = [];
const scripts = [];
const init = (id) => ({ type: "system", subtype: "init", session_id: id });
const ev = (event) => ({ type: "stream_event", event });
const text = (t) => [
	ev({ type: "message_start", message: { id: "msg_text", usage: {} } }),
	ev({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
	ev({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: t } }),
	ev({ type: "content_block_stop", index: 0 }),
	ev({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: {} }),
	ev({ type: "message_stop" }),
];
const result = (t) => ({ type: "result", subtype: "success", is_error: false, result: t });

let clock = 1;
const user = (t) => ({ role: "user", content: t, timestamp: clock++ });
const assistant = (t) => ({ role: "assistant", content: [{ type: "text", text: t }],
	api: "claude-bridge", provider: "claude-bridge", model: model.id,
	usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
	stopReason: "stop", timestamp: clock++ });
const settle = () => new Promise((r) => setTimeout(r, 20));

/** What CC does on a turn: create or append to its session file. */
function ccTurn(id, prompt, answer) {
	return () => {
		const session = existsSync(ccPath(id))
			? openSession({ sessionId: id, projectPath: cwd, claudeDir })
			: createSession({ sessionId: id, projectPath: cwd, claudeDir });
		session.importMessages([{ role: "user", content: prompt }, { role: "assistant", content: [{ type: "text", text: `${answer} [cc record]` }] }]);
		session.save();
	};
}

/** One normal turn: CC writes `id`'s file and answers. */
function turn(id, prompt, answer) {
	return { steps: [ccTurn(id, prompt, answer), init(id), ...text(answer), result(answer)] };
}

beforeEach(() => {
	resetSharedSession();
	activeQueryContexts.clear();
	calls.length = 0;
	scripts.length = 0;
	setQuery(({ prompt, options }) => {
		const script = scripts.shift();
		if (!script) throw new Error("no fake script queued");
		calls.push({ resume: options.resume });
		let closed = false, onClose;
		const killed = new Promise((r) => { onClose = r; });
		const gen = (async function* () {
			await prompt[Symbol.asyncIterator]().next();
			for (const step of script.steps) {
				if (closed) return;
				if (typeof step === "function") { await step(killed); continue; }
				yield step;
			}
		})();
		gen.interrupt = async () => {};
		gen.close = () => { closed = true; onClose(); };
		return gen;
	});
});

afterEach(() => setQuery(null));

/** What a move from `from` into this cwd leaves: the record and CC's file under `from`. */
function movedFrom(piSession, id, from) {
	const path = join(sessionRecordDir(), `${piSession}.json`);
	writeFileSync(path, JSON.stringify({ ...JSON.parse(readFileSync(path, "utf-8")), cwd: from }));
	const old = getSessionPath(id, normalizeProjectPath(from), claudeDir);
	mkdirSync(dirname(old), { recursive: true });
	renameSync(ccPath(id), old);
}

/** Turn one of a pi session, ending normally on CC session `id`. */
async function firstTurn(piSession, id) {
	const u1 = user("hi");
	scripts.push(turn(id, "hi", "hello"));
	const a1 = await streamSimple(model, { messages: [u1], tools: [] }, { sessionId: piSession }).result();
	return [u1, { ...a1, timestamp: clock++ }];
}

describe("session records", () => {
	it("a new process resumes the file of a query that ended normally", async () => {
		const id = randomUUID();
		const history = await firstTurn("pi-resume", id);
		const record = readSessionRecord("pi-resume");
		assert.equal(record?.kind, "complete");
		assert.equal(record?.sessionId, id);
		assert.equal(record?.cursor, 1, "the file holds the prompt; pi appends the answer next");

		forgetSessionMirrors();
		const before = readFileSync(ccPath(id), "utf-8");
		scripts.push(turn(id, "next", "done"));
		await streamSimple(model, { messages: [...history, user("next")], tools: [] }, { sessionId: "pi-resume" }).result();

		assert.equal(calls[1].resume, id, "the second process resumes the same file");
		assert.ok(readFileSync(ccPath(id), "utf-8").startsWith(before), "the file was resumed, not rewritten");
	});

	it("a new process cuts back the file of a query that never ended", async () => {
		const id = randomUUID();
		const history = await firstTurn("pi-cut", id);

		// Turn two writes to the file, then its process dies before the query ends.
		const u2 = user("long job");
		scripts.push({ steps: [ccTurn(id, "long job", "partial"), init(id), (killed) => killed] });
		const sizeBefore = readFileSync(ccPath(id)).length;
		void streamSimple(model, { messages: [...history, u2], tools: [] }, { sessionId: "pi-cut" });
		await settle();
		const record = readSessionRecord("pi-cut");
		assert.equal(record?.kind, "start");
		assert.equal(record?.bytes, sizeBefore, "the start record holds the size before the query wrote");
		forgetSessionMirrors();
		activeQueryContexts.clear();

		const a2 = assistant("pi kept this");
		scripts.push(turn(randomUUID(), "again", "ok"));
		await streamSimple(model, { messages: [...history, u2, a2, user("again")], tools: [] }, { sessionId: "pi-cut" }).result();

		const forked = calls[2].resume;
		assert.ok(forked && forked !== id, "the cut goes to a new file, like after an abort");
		const copy = openSession({ sessionId: forked, projectPath: cwd, claudeDir });
		assert.match(JSON.stringify(copy.messages[1]), /hello \[cc record\]/, "the copy keeps CC's own records before the cut, which a rebuild from pi's history would not");
		assert.doesNotMatch(JSON.stringify(copy.messages), /partial/, "the records after the cut are gone");
		assert.match(JSON.stringify(copy.messages.at(-1)), /pi kept this/, "pi's copy of the turn follows the cut");
	});

	it("rebuilds when pi's history no longer matches the record", async () => {
		const id = randomUUID();
		const history = await firstTurn("pi-compacted", id);
		forgetSessionMirrors();

		// A compaction in the other process replaced the start of the history.
		scripts.push(turn(randomUUID(), "next", "done"));
		await streamSimple(model, { messages: [user("summary of earlier work"), history[1], user("next")], tools: [] }, { sessionId: "pi-compacted" }).result();

		assert.notEqual(calls[1].resume, id, "a mismatched history must not resume the old file");
		assert.ok(calls[1].resume, "it rebuilds a file from pi's history");
	});

	it("a session moved from another cwd resumes a copy of its file", async () => {
		const id = randomUUID();
		const history = await firstTurn("pi-moved", id);
		const before = readFileSync(ccPath(id), "utf-8");
		movedFrom("pi-moved", id, join(cwd, "elsewhere"));
		forgetSessionMirrors();

		scripts.push(turn(id, "next", "done"));
		await streamSimple(model, { messages: [...history, user("next")], tools: [] }, { sessionId: "pi-moved" }).result();

		assert.equal(calls[1].resume, id, "it resumes the same session in the new cwd");
		assert.ok(readFileSync(ccPath(id), "utf-8").startsWith(before), "the new cwd holds a copy of the file, not a rebuild");
	});

	it("a session moved during a query cuts its file into the new cwd", async () => {
		const id = randomUUID();
		const history = await firstTurn("pi-moved-cut", id);
		const u2 = user("move");
		scripts.push({ steps: [ccTurn(id, "move", "partial"), init(id), (killed) => killed] });
		void streamSimple(model, { messages: [...history, u2], tools: [] }, { sessionId: "pi-moved-cut" });
		await settle();
		assert.equal(readSessionRecord("pi-moved-cut")?.kind, "start");
		movedFrom("pi-moved-cut", id, join(cwd, "elsewhere"));
		forgetSessionMirrors();
		activeQueryContexts.clear();

		scripts.push(turn(randomUUID(), "again", "ok"));
		await streamSimple(model, { messages: [...history, u2, assistant("pi kept this"), user("again")], tools: [] }, { sessionId: "pi-moved-cut" }).result();

		const forked = calls[2].resume;
		assert.ok(forked && forked !== id, "the cut goes to a new file, like after an abort");
		const copy = openSession({ sessionId: forked, projectPath: cwd, claudeDir });
		assert.match(JSON.stringify(copy.messages[1]), /hello \[cc record\]/, "the copy in the new cwd keeps CC's own records");
		assert.doesNotMatch(JSON.stringify(copy.messages), /partial/);
	});

	it("rebuilds when the file is gone", async () => {
		const gone = randomUUID();
		const history = await firstTurn("pi-gone", gone);
		rmSync(ccPath(gone));
		forgetSessionMirrors();
		scripts.push(turn(randomUUID(), "next", "done"));
		await streamSimple(model, { messages: [...history, user("next")], tools: [] }, { sessionId: "pi-gone" }).result();
		assert.notEqual(calls[1].resume, gone);
	});

	it("a query that fails removes the record", async () => {
		const id = randomUUID();
		const history = await firstTurn("pi-failed", id);
		scripts.push({ steps: [init(id), () => { throw new Error("boom"); }] });
		await streamSimple(model, { messages: [...history, user("next")], tools: [] }, { sessionId: "pi-failed" }).result();
		assert.equal(readSessionRecord("pi-failed"), null, "the file may be the cause, so no process may resume it");
		assert.equal(getSharedSession("pi-failed"), null);
	});

	it("a clean start removes an old record", async () => {
		const id = randomUUID();
		await firstTurn("pi-new", id);
		assert.ok(readSessionRecord("pi-new"));
		// A new process where the same pi session starts over from no history (a
		// host reusing the id). In the old process the mirror would keep the record:
		// a shorter context there is an ephemeral query that preserves the session.
		forgetSessionMirrors();
		scripts.push({ steps: [init(randomUUID()), (killed) => killed] });
		void streamSimple(model, { messages: [user("fresh")], tools: [] }, { sessionId: "pi-new" });
		await settle();
		assert.equal(readSessionRecord("pi-new"), null);
	});
});
