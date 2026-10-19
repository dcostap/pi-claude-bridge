/**
 * Cached compaction summaries (src/cached-summary.ts and the wiring in index.ts).
 *
 * The helpers are pinned directly. The wiring runs a real pi session, extension
 * runner and compaction against the bridge provider, with only Claude Code's
 * query() replaced: provider turns write a real session file the way Claude Code
 * would, so the checkpoint, the copy and the fallbacks run against real files.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createSession, getSessionPath, normalizeProjectPath, openSession } from "cc-session-io";
import { createAgentSession, DefaultResourceLoader, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import {
	buildSummaryPrompt,
	cacheTtlFromUsage,
	classifySummaryPrompt,
	completeLines,
	describeMessage,
	historyDigest,
	lastAssistantText,
	prefixDigest,
	rememberCompaction,
	summaryCosts,
	tailRecords,
} from "../src/cached-summary.js";
import { forkSessionPrefix } from "../src/session-fork.js";
import activate, { __test } from "../src/index.js";

describe("cached summary helpers", () => {
	it("tells pi's two summarizer prompts apart", () => {
		assert.equal(classifySummaryPrompt("<conversation>\n[User]: hi\n</conversation>\n\nSummarize"), "history");
		assert.equal(classifySummaryPrompt("# Conversation\n[User]: hi\n\n# Instructions\n..."), "turnPrefix");
		assert.equal(classifySummaryPrompt("Summarize this branch"), undefined);
	});

	it("reads the last answer of a session file across split blocks and trailing records", () => {
		const a = (id, text, extra = {}) => ({ type: "assistant", message: { id, model: "claude-x", role: "assistant", content: [{ type: "text", text }] }, ...extra });
		const thinking = { type: "assistant", message: { id: "m2", model: "claude-x", content: [{ type: "thinking", thinking: "hmm" }] } };
		const records = [
			{ type: "user", message: { role: "user", content: "hi" } },
			a("m1", "old"),
			{ type: "user", message: { role: "user", content: "again" } },
			thinking, a("m2", "Hello "), a("m2", "there."),
			{ type: "attachment", attachment: { type: "prompt_snapshot" } },
			{ type: "last-prompt", lastPrompt: "again" },
		];
		assert.equal(lastAssistantText(records), "Hello there.");
		assert.equal(lastAssistantText([...records, { type: "user", message: { role: "user", content: "[Request interrupted by user]" } }]), undefined);
		assert.equal(lastAssistantText([records[0], { type: "assistant", message: { id: "s", model: "<synthetic>", content: [{ type: "text", text: "No response requested." }] } }]), undefined);
		assert.equal(lastAssistantText([records[0]]), undefined);

		// Parsed from the end: from the last user record on, so a bad line before it goes unread.
		const text = ["{not json", ...records.map((r) => JSON.stringify(r)), ""].join("\n");
		assert.deepEqual(tailRecords(text), records.slice(2));
		assert.equal(lastAssistantText(tailRecords(text)), "Hello there.");
		assert.deepEqual(tailRecords(records.slice(3).map((r) => JSON.stringify(r)).join("\n")), records.slice(3), "no user record: every line");
		assert.throws(() => tailRecords(`${JSON.stringify(records[2])}\n{not json\n`));
		assert.deepEqual(tailRecords(""), []);
	});

	it("reads the cache lifetime from usage, and nothing from usage that wrote no cache", () => {
		assert.equal(cacheTtlFromUsage({ cache_creation: { ephemeral_1h_input_tokens: 10, ephemeral_5m_input_tokens: 0 } }), 3_600_000);
		assert.equal(cacheTtlFromUsage({ cache_creation: { ephemeral_1h_input_tokens: 0, ephemeral_5m_input_tokens: 10 } }), 300_000);
		assert.equal(cacheTtlFromUsage({ cache_creation: { ephemeral_1h_input_tokens: 0, ephemeral_5m_input_tokens: 0 } }), undefined);
		assert.equal(cacheTtlFromUsage({ input_tokens: 3 }), undefined);
		assert.equal(cacheTtlFromUsage(undefined), undefined);
	});

	it("digests every message of the history, not just its ends", () => {
		const history = [{ role: "user", content: "a" }, { role: "assistant", content: "b" }, { role: "user", content: "c" }];
		const edited = [history[0], { role: "assistant", content: "B" }, history[2]];
		assert.notEqual(historyDigest(history, 3), historyDigest(edited, 3));
		assert.equal(historyDigest(history, 2), historyDigest([...history.slice(0, 2), { role: "user", content: "other" }], 2));
	});

	it("keeps complete lines only", () => {
		assert.equal(completeLines(Buffer.from("a\nb\nc")).toString(), "a\nb\n");
		assert.equal(completeLines(Buffer.from("partial")).length, 0);
	});

	it("names messages so Claude can find them", () => {
		assert.equal(describeMessage({ role: "user", content: "Fix   the\nbug" }), 'the user message that begins "Fix the bug"');
		assert.match(describeMessage({ role: "user", content: "x".repeat(300) }), /x{100}…"$/);
		assert.equal(describeMessage({ role: "assistant", content: [{ type: "toolCall", name: "bash" }] }), "your bash tool call");
		assert.equal(describeMessage({ role: "assistant", content: [{ type: "toolCall", name: "read", arguments: { path: "a.txt" } }] }), 'your read tool call with arguments {"path":"a.txt"}');
		assert.equal(describeMessage({ role: "bashExecution", command: "ls -la" }), 'the user\'s shell command "ls -la"');
		assert.equal(describeMessage({ role: "toolResult", content: [] }), undefined);
		assert.equal(describeMessage({ role: "user", content: "   " }), undefined);
	});

	it("builds the prompt pi's format asks for, with what the cache does not hold", () => {
		const split = buildSummaryPrompt({ updatesPreviousSummary: false, splitTurn: true, keptFrom: "K", turnStart: "T", laterMessages: "[User]: later", customInstructions: " focus on tests " });
		assert.match(split, /^\[Context checkpoint request\]/);
		assert.match(split, /The most recent messages, from K onward, stay in context verbatim\. They start partway through the turn that began with T, and the start of that turn is summarized separately\. Everything before that turn will be replaced/);
		assert.match(split, /<later-messages>\n\[User\]: later\n<\/later-messages>/);
		assert.match(split, /summary of the conversation before the turn that began with T that another LLM will use to continue the work\. Stop where that turn begins: do not summarize that turn or anything after it\./);
		assert.match(split, /## Goal[\s\S]*## Critical Context/);
		assert.match(split, /Additional focus: focus on tests$/);
		assert.doesNotMatch(split, /PRESERVE all existing information|## Original Request/);

		const plain = buildSummaryPrompt({ updatesPreviousSummary: false, splitTurn: false, keptFrom: "K" });
		assert.match(plain, /from K onward, stay in context verbatim\. Everything before them will be replaced by the summary you write now/);
		assert.match(plain, /summary of the conversation that another LLM/);
		assert.doesNotMatch(plain, /Stop where|partway/);

		const update = buildSummaryPrompt({ updatesPreviousSummary: true, splitTurn: false });
		assert.match(update, /PRESERVE all existing information/);
		assert.match(update, /at the end of the conversation/);
		assert.doesNotMatch(update, /later-messages|Additional focus/);
	});

	it("prices the cached request against pi's, with the cache's write rate", () => {
		// 200k cached conversation: 20k to read, plus the new prompt written to cache.
		assert.deepEqual(summaryCosts({ conversationTokens: 200_000, cachedPromptTokens: 2_000, nativePromptTokens: 150_000, cacheTtlMs: 3_600_000 }), { cached: 24_000, native: 300_000 });
		assert.deepEqual(summaryCosts({ conversationTokens: 200_000, cachedPromptTokens: 2_000, nativePromptTokens: 150_000, cacheTtlMs: 300_000 }), { cached: 22_500, native: 187_500 });
		// Little to summarize: reading the whole conversation costs more than sending the span.
		const small = summaryCosts({ conversationTokens: 200_000, cachedPromptTokens: 2_000, nativePromptTokens: 5_000, cacheTtlMs: 3_600_000 });
		assert.ok(small.cached > small.native);
	});

	it("refuses to copy a session file rewritten since its checkpoint", () => {
		const cwd = mkdtempSync(join(tmpdir(), "cached-summary-fork-"));
		try {
			const s = createSession({ sessionId: randomUUID(), projectPath: cwd });
			s.addUserMessage("Hi");
			s.addAssistantMessage([{ type: "text", text: "Hello." }]);
			s.save();
			const prefix = readFileSync(s.jsonlPath);
			const digest = prefixDigest(prefix);
			assert.ok(forkSessionPrefix({ sourceSessionId: s.sessionId, bytes: prefix.length, cwd, digest }));
			writeFileSync(s.jsonlPath, prefix.toString().replace("Hello.", "Howdy."));
			assert.throws(() => forkSessionPrefix({ sourceSessionId: s.sessionId, bytes: prefix.length, cwd, digest }), /rewritten/);
		} finally {
			rmSync(cwd, { recursive: true, force: true });
		}
	});
});

// --- A real pi session over the bridge, with Claude Code's query() replaced ---

const model = { id: "claude-sonnet-4-6", name: "Controlled bridge", api: "claude-bridge", provider: "claude-bridge", baseUrl: "claude-bridge", reasoning: false, input: ["text"], contextWindow: 200000, maxTokens: 8192, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
const claudeDir = () => process.env.CLAUDE_CONFIG_DIR;
const ev = (event) => ({ type: "stream_event", event });

/** The SDK messages of one answered request. */
function answerEvents(text, usage) {
	return [
		ev({ type: "message_start", message: { id: `msg_${randomUUID()}`, usage } }),
		ev({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
		ev({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text } }),
		ev({ type: "content_block_stop", index: 0 }),
		ev({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 20 } }),
		ev({ type: "message_stop" }),
		{ type: "result", subtype: "success", is_error: false, result: text, usage },
	];
}

function fakeQuery(steps, onClose) {
	return {
		async *[Symbol.asyncIterator]() {
			for (const step of steps) {
				if (typeof step === "function") await step();
				else yield step;
			}
		},
		close() { onClose?.(); },
		async interrupt() {},
	};
}

/**
 * A pi session that has run `turns` bridge turns. Each provider turn appends its
 * exchange to a real session file. `summarize` answers the cached summary query.
 */
async function bridgeSession(t, { turns = 2, summarize, beforeCompact } = {}) {
	const cwd = mkdtempSync(join(tmpdir(), "cached-summary-e2e-"));
	const ccSessionId = randomUUID();
	const calls = [];
	const notices = [];
	let turn = 0;
	const sessionFile = (id) => getSessionPath(id, normalizeProjectPath(cwd), claudeDir());
	const handle = { calls, notices, cwd, ccSessionId, sessionFile, state: () => __test.getSharedSession(handle.piSessionId) };
	__test.setQuery(({ prompt, options }) => {
		if (typeof prompt !== "string") {
			const answer = `ANSWER-${turn++} ` + "detail ".repeat(150);
			calls.push({ kind: "turn", options });
			const id = options.resume ?? ccSessionId;
			return fakeQuery([
				() => {
					const s = options.resume
						? openSession({ sessionId: id, projectPath: cwd, claudeDir: claudeDir() })
						: createSession({ sessionId: id, projectPath: cwd, claudeDir: claudeDir() });
					s.addUserMessage(`question ${turn}`);
					s.addAssistantMessage([{ type: "text", text: answer }]);
					s.dangerousAppendRecord({ type: "last-prompt", lastPrompt: `question ${turn}`, sessionId: id });
					s.save();
				},
				{ type: "system", subtype: "init", session_id: id },
				...answerEvents(answer, { input_tokens: 5, cache_read_input_tokens: 1000, cache_creation_input_tokens: 200, cache_creation: { ephemeral_1h_input_tokens: 200, ephemeral_5m_input_tokens: 0 } }),
			]);
		}
		if (options.persistSession === false) {
			calls.push({ kind: "isolated", prompt, options });
			return fakeQuery([{ type: "result", subtype: "success", result: `ISOLATED-${calls.filter((c) => c.kind === "isolated").length}` }]);
		}
		const call = { kind: "cached", prompt, options, copyExisted: existsSync(sessionFile(options.resume)), closed: false };
		calls.push(call);
		return fakeQuery(summarize(call), () => { call.closed = true; });
	});

	const manager = SessionManager.inMemory(cwd);
	const settings = SettingsManager.inMemory({ compaction: { enabled: false, reserveTokens: 1000, keepRecentTokens: 50 }, retry: { enabled: false } });
	const loader = new DefaultResourceLoader({
		cwd, agentDir: process.env.PI_CODING_AGENT_DIR, settingsManager: settings,
		noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
		// Loaded after the bridge, so its session_before_compact runs after the
		// bridge has refreshed the checkpoint.
		extensionFactories: [activate, (pi) => pi.on("session_before_compact", (event) => { beforeCompact?.(handle, event); })],
	});
	await loader.reload();
	assert.deepEqual(loader.getExtensions().errors, []);
	const { session } = await createAgentSession({ cwd, agentDir: process.env.PI_CODING_AGENT_DIR, resourceLoader: loader, sessionManager: manager, settingsManager: settings, noTools: "all", model });
	await session.bindExtensions({ onError: (error) => { throw new Error(error.error); } });
	Object.assign(handle, { session, manager, settings, piSessionId: manager.getSessionId() });
	__test.setPiUI({ notify: (message, level) => notices.push({ message, level }) });
	t.after(async () => {
		await session.extensionRunner.emit({ type: "session_shutdown" });
		session.dispose();
		__test.setQuery(null);
		__test.setPiUI(null);
		rmSync(cwd, { recursive: true, force: true });
	});
	for (let i = 0; i < turns; i++) await session.prompt(`question ${i} ` + "context ".repeat(150));
	return handle;
}

/** A cached answer: the history summary. */
const cachedAnswer = (call, usage = { input_tokens: 300, output_tokens: 80, cache_read_input_tokens: 5000, cache_creation_input_tokens: 0 }) => {
	return [
		{ type: "system", subtype: "init", session_id: call.options.resume },
		{ type: "assistant", message: { id: "msg_s", role: "assistant", content: [{ type: "text", text: "x" }], stop_reason: null, usage } },
		// As the SDK streams it: the turn's stop reason arrives on the result only.
		{ type: "result", subtype: "success", is_error: false, stop_reason: "end_turn", result: "## Goal\nCACHED-HISTORY", usage },
	];
};

describe("cached compaction summaries", () => {
	it("a run ending on a bridge answer records a checkpoint at the end of the session file", { timeout: 20000 }, async (t) => {
		const s = await bridgeSession(t, { summarize: cachedAnswer });
		const checkpoint = s.state()?.checkpoint;
		assert.ok(checkpoint, "checkpoint recorded at agent_end");
		assert.equal(checkpoint.sessionId, s.ccSessionId);
		assert.equal(checkpoint.bytes, statSync(s.sessionFile(s.ccSessionId)).size);
		assert.equal(checkpoint.piCount, 4);
		assert.equal(checkpoint.cacheTtlMs, 3_600_000, "lifetime read from the usage Claude Code reported");
		assert.equal(checkpoint.recipe.cliModel, "claude-sonnet-4-6");
	});

	it("summarizes from a copy of the session file with the last turn's options", { timeout: 20000 }, async (t) => {
		const s = await bridgeSession(t, { summarize: cachedAnswer });
		const sourceBefore = readFileSync(s.sessionFile(s.ccSessionId));
		const result = await s.session.compact("keep the test names");

		const cached = s.calls.filter((c) => c.kind === "cached");
		assert.equal(cached.length, 1, "one cached query, for the history");
		const isolated = s.calls.filter((c) => c.kind === "isolated");
		assert.equal(isolated.length, 1, "the split turn's prefix stays on pi's path");
		assert.ok(isolated[0].prompt.startsWith("# Conversation"));
		const [call] = cached;
		const lastTurn = s.calls.filter((c) => c.kind === "turn").at(-1).options;
		assert.notEqual(call.options.resume, s.ccSessionId, "resumes a copy, never the conversation's own file");
		assert.ok(call.copyExisted, "the copy exists while the query runs");
		assert.equal(existsSync(s.sessionFile(call.options.resume)), false, "the copy is deleted afterwards");
		assert.ok(call.closed);
		assert.deepEqual(readFileSync(s.sessionFile(s.ccSessionId)), sourceBefore, "the conversation's file is untouched");
		assert.deepEqual(call.options.systemPrompt, lastTurn.systemPrompt);
		assert.deepEqual(call.options.settings, lastTurn.settings);
		assert.deepEqual(call.options.extraArgs, lastTurn.extraArgs);
		assert.deepEqual(call.options.tools, []);
		assert.equal(call.options.effort, lastTurn.effort);
		assert.equal(call.options.cwd, lastTurn.cwd);
		assert.equal(call.options.persistSession, undefined);
		assert.match(call.prompt, /^\[Context checkpoint request\]/);
		assert.match(call.prompt, /Additional focus: keep the test names/);
		assert.doesNotMatch(call.prompt, /<later-messages>/);

		assert.match(result.summary, /^## Goal\nCACHED-HISTORY\n\n---\n\n\*\*Turn Context \(split turn\):\*\*\n\nISOLATED-1$/);
		assert.equal(s.state().checkpoint, undefined, "the compaction consumes the checkpoint");
		assert.deepEqual(s.notices, []);
	});

	it("passes messages newer than the checkpoint as text", { timeout: 20000 }, async (t) => {
		const s = await bridgeSession(t, { summarize: cachedAnswer });
		s.manager.appendMessage({ role: "user", content: "LATER-NOTE about the deploy", timestamp: Date.now() });
		await s.session.compact();
		const [call] = s.calls.filter((c) => c.kind === "cached");
		assert.match(call.prompt, /<later-messages>\n[\s\S]*LATER-NOTE about the deploy[\s\S]*<\/later-messages>/);
	});

	it("falls back to pi's summary when the cache has likely expired", { timeout: 20000 }, async (t) => {
		const s = await bridgeSession(t, { summarize: cachedAnswer, beforeCompact: (h) => { h.state().checkpoint.respondedAt = Date.now() - 3_600_000 + 10_000; } });
		const result = await s.session.compact();
		assert.equal(s.calls.filter((c) => c.kind === "cached").length, 0);
		assert.equal(s.calls.filter((c) => c.kind === "isolated").length, 2);
		assert.match(result.summary, /ISOLATED-1/);
		assert.deepEqual(s.notices, [], "an expected miss is not a warning");
	});

	it("falls back when the session file was rewritten since the checkpoint", { timeout: 20000 }, async (t) => {
		const s = await bridgeSession(t, {
			summarize: cachedAnswer,
			beforeCompact: (h) => {
				const path = h.sessionFile(h.ccSessionId);
				writeFileSync(path, readFileSync(path, "utf8").replaceAll("question", "QUESTION"));
			},
		});
		const result = await s.session.compact();
		assert.equal(s.calls.filter((c) => c.kind === "cached").length, 0);
		assert.match(result.summary, /ISOLATED-1/);
	});

	it("takes pi's summary when it is estimated cheaper than reading the whole conversation", { timeout: 20000 }, async (t) => {
		// A large cached conversation of which only a small span is summarized.
		const s = await bridgeSession(t, { summarize: cachedAnswer, beforeCompact: (_h, event) => { event.preparation.tokensBefore = 150_000; } });
		const result = await s.session.compact();
		assert.equal(s.calls.filter((c) => c.kind === "cached").length, 0);
		assert.equal(s.calls.filter((c) => c.kind === "isolated").length, 2);
		assert.match(result.summary, /ISOLATED-1/);
		assert.deepEqual(s.notices, [], "a cost choice is not a warning");
	});

	it("falls back when the request would not fit the context window", { timeout: 20000 }, async (t) => {
		const s = await bridgeSession(t, { summarize: cachedAnswer, beforeCompact: (_h, event) => { event.preparation.tokensBefore = 995_000; } });
		await s.session.compact();
		assert.equal(s.calls.filter((c) => c.kind === "cached").length, 0);
		assert.equal(s.calls.filter((c) => c.kind === "isolated").length, 2);
	});

	it("falls back when pi's branch no longer starts with the checkpointed conversation", { timeout: 20000 }, async (t) => {
		const s = await bridgeSession(t, { summarize: cachedAnswer, beforeCompact: (h) => { h.state().checkpoint.piDigest = "0".repeat(64); } });
		await s.session.compact();
		assert.equal(s.calls.filter((c) => c.kind === "cached").length, 0);
		assert.equal(s.calls.filter((c) => c.kind === "isolated").length, 2);
	});

	it("falls back, with a warning, when the cached query fails; the rest of the compaction stays on pi's path", { timeout: 20000 }, async (t) => {
		const s = await bridgeSession(t, { summarize: () => [{ type: "result", subtype: "success", is_error: true, result: "API Error: 529 overloaded" }] });
		const result = await s.session.compact();
		assert.equal(s.calls.filter((c) => c.kind === "cached").length, 1);
		assert.equal(s.calls.filter((c) => c.kind === "isolated").length, 2, "history retried isolated, turn prefix isolated");
		assert.match(result.summary, /ISOLATED-1/);
		assert.equal(s.notices.length, 1);
		assert.match(s.notices[0].message, /529 overloaded/);
		const [call] = s.calls.filter((c) => c.kind === "cached");
		assert.equal(existsSync(s.sessionFile(call.options.resume)), false, "the copy is deleted after a failure too");
	});

	it("falls back when the answer was cut off", { timeout: 20000 }, async (t) => {
		const s = await bridgeSession(t, {
			summarize: (call) => cachedAnswer(call).map((m) => m.type === "result" ? { ...m, stop_reason: "max_tokens" } : m),
		});
		const result = await s.session.compact();
		assert.match(result.summary, /ISOLATED-1/);
		assert.match(s.notices[0].message, /max_tokens/);
	});

	it("an aborted compaction stops the cached query and does not fall back", { timeout: 20000 }, async (t) => {
		let session;
		const s = await bridgeSession(t, {
			summarize: (call) => [
				() => { session.abortCompaction(); },
				() => new Promise((resolve) => setTimeout(resolve, 50)),
				...cachedAnswer(call),
			],
		});
		session = s.session;
		await assert.rejects(s.session.compact(), /abort|cancel/i);
		const [call] = s.calls.filter((c) => c.kind === "cached");
		assert.ok(call.closed);
		assert.equal(s.calls.filter((c) => c.kind === "isolated").length, 0);
		assert.equal(existsSync(s.sessionFile(call.options.resume)), false);
		assert.deepEqual(s.notices, []);
		assert.equal(s.manager.getBranch().filter((e) => e.type === "compaction").length, 0);
	});

	/** Turn pi's auto-compaction on after the setup turns, for one compaction, and prompt once more. */
	async function autoCompact(s) {
		const reasons = [];
		const unsubscribe = s.session.subscribe((event) => {
			if (event.type !== "compaction_end") return;
			reasons.push({ reason: event.reason, willRetry: event.willRetry, error: event.errorMessage });
			// The fake turns all report the same usage, which would compact again after the next one.
			s.settings.applyOverrides({ compaction: { enabled: false } });
		});
		s.settings.applyOverrides({ compaction: { enabled: true } });
		await s.session.prompt("one more question " + "context ".repeat(150));
		unsubscribe();
		return { reasons, compactions: s.manager.getBranch().filter((e) => e.type === "compaction") };
	}

	it("serves pi's threshold auto-compaction from the cache", { timeout: 20000 }, async (t) => {
		const s = await bridgeSession(t, { summarize: cachedAnswer });
		// Each turn reports ~1.2k context tokens: past a 200k window less 199k reserve.
		s.settings.applyOverrides({ compaction: { reserveTokens: 199_000 } });
		const { reasons, compactions } = await autoCompact(s);
		assert.deepEqual(reasons, [{ reason: "threshold", willRetry: false, error: undefined }]);
		assert.equal(s.calls.filter((c) => c.kind === "cached").length, 1);
		assert.equal(compactions.length, 1);
		assert.match(compactions[0].summary, /CACHED-HISTORY/);
		assert.deepEqual(s.notices, []);
	});

	it("serves an overflow auto-compaction of a completed answer from the cache", { timeout: 20000 }, async (t) => {
		// pi calls it "overflow", with no retry, when a completed answer's input passed
		// pi's registered window: a run that grew past the threshold before agent_end.
		// The served window (200k for this model) still holds the conversation.
		const s = await bridgeSession(t, { summarize: cachedAnswer });
		// What provider.contextWindow does: register less than the 200k served for this model.
		s.session.agent.state.model = { ...s.session.model, contextWindow: 1000 };
		const { reasons, compactions } = await autoCompact(s);
		assert.deepEqual(reasons, [{ reason: "overflow", willRetry: false, error: undefined }]);
		assert.equal(s.calls.filter((c) => c.kind === "cached").length, 1);
		assert.equal(compactions.length, 1);
		assert.match(compactions[0].summary, /CACHED-HISTORY/);
	});

	it("leaves overflow recovery to pi's summary", { timeout: 20000 }, async (t) => {
		const s = await bridgeSession(t, { summarize: cachedAnswer });
		assert.ok(s.state().checkpoint);
		// Straight to the provider: a real overflow needs a turn the API rejected.
		const controller = new AbortController();
		const branchEntries = s.manager.getBranch();
		rememberCompaction(controller.signal, {
			piSessionId: s.piSessionId, reason: "overflow", willRetry: true, branchEntries,
			preparation: { firstKeptEntryId: branchEntries.at(-1).id, messagesToSummarize: [], turnPrefixMessages: [], isSplitTurn: false, tokensBefore: 1000 },
		});
		const prompt = { role: "user", content: "<conversation>\n[User]: hi\n</conversation>\n\nSummarize", timestamp: 0 };
		const message = await __test.streamClaudeAgentSdk(model, { systemPrompt: "Summarize", messages: [prompt] }, { cacheRetention: "none", signal: controller.signal, sessionId: s.piSessionId }).result();
		assert.equal(message.content[0].text, "ISOLATED-1");
		assert.equal(s.calls.filter((c) => c.kind === "cached").length, 0);
	});

	it("is off when provider.cachedCompaction is false", { timeout: 20000 }, async (t) => {
		const s = await bridgeSession(t, { summarize: cachedAnswer });
		const configDir = join(s.cwd, ".pi");
		const { mkdirSync } = await import("node:fs");
		mkdirSync(configDir, { recursive: true });
		writeFileSync(join(configDir, "claude-bridge.json"), JSON.stringify({ provider: { cachedCompaction: false } }));
		await s.session.compact();
		assert.equal(s.calls.filter((c) => c.kind === "cached").length, 0);
	});
});
