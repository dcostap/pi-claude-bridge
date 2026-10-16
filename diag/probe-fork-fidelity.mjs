#!/usr/bin/env node
// Does a session cut with the SDK's forkSession(upToMessageId) read back the
// prompt cache that the live session left?
//
// After an abort the bridge rotates to a new session id and rebuilds Claude
// Code's session from pi's history, and the rebuild misses the cache from
// messages[0] on (diag/AUDIT.md). The proposed fix keeps the new id but makes
// the new file with forkSession, which copies CC's own records up to a chosen
// message. That is worth building only if CC sends the forked prefix with the
// same bytes it sent live. This probe tests that directly with the Agent SDK,
// without pi or the bridge, behind diag/capture-proxy.mjs.
//
//   node diag/probe-fork-fidelity.mjs [--model claude-haiku-5-5] [--effort low] [--runs 1] [--out DIR]
//
// Each run is one query per prompt, as the bridge runs them:
//   P1  new session
//   P2  resume            control: a normal continuation, expected warm
//   P3  resume, aborted   killed while its Bash call sleeps
//   fork                  forkSession(session, { upToMessageId: P2's last assistant record })
//   P4  resume the fork   the test
//
// Two comparisons for P4's first request:
//   cache   cacheRead against what P2's last request left cached (cacheRead + cacheWrite).
//           P3's first request also cached, but its breakpoint sits after the [P3] prompt,
//           which P4 does not have, so P2's last request is the most P4 can read back.
//   bytes   P3's first request against P4's first request, element by element. P3 sent the
//           live prefix plus [P3]; a faithful fork differs first at the [P3] message.
//
// Differences from the bridge: CC's built-in Read and Bash tools instead of pi's tools over
// MCP, and settingSources [] so no user settings, hooks or plugins load in the child.
// The record shapes in question (text, thinking, tool_use, tool_result) are the same.

import { spawn } from "node:child_process";
import { closeSync, mkdirSync, mkdtempSync, openSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { connect, createServer } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { forkSession, getSessionMessages, query } from "@anthropic-ai/claude-agent-sdk";

const REPO = join(dirname(fileURLToPath(import.meta.url)), "..");

const args = process.argv.slice(2);
const flag = (name, fallback) => {
	const i = args.indexOf(`--${name}`);
	return i === -1 ? fallback : args[i + 1];
};
const MODEL = flag("model", "claude-haiku-5-5");
const EFFORT = flag("effort", "low");
const RUNS = Number(flag("runs", 1));
// realpath expands Windows 8.3 short names, so CC's project dir matches the cwd we pass.
const OUT = flag("out") ?? mkdtempSync(join(realpathSync.native(tmpdir()), "fork-fidelity-"));

const BILLING_HEADER = "x-anthropic-billing-header";

const PLAN = [
	{ tag: "[P1]", text: "Use the Read tool to read notes-a.txt and notes-b.txt. Issue both Read calls in a single message, in parallel. Then reply with just the two codewords." },
	// Arithmetic so that adaptive thinking puts signed thinking blocks into the forked history.
	{ tag: "[P2]", text: "Work out 347 * 29 - 4183 / 47 step by step, then use the Bash tool to run: echo <your result>. Reply with just the number." },
	{ tag: "[P3]", text: "Use the Bash tool to run: sleep 45 && echo finished. Reply with just: done." },
	{ tag: "[P4]", text: "Use the Bash tool to run: cat notes-b.txt. Reply with just its contents." },
];

// The child must not inherit this shell's Claude Code session (CLAUDECODE,
// CLAUDE_CODE_SESSION_ID, CLAUDE_CODE_PLUGIN_DIRS, ...).
function childEnv(port) {
	const env = Object.fromEntries(Object.entries(process.env).filter(([k]) => !/^CLAUDE/i.test(k) && !/^ANTHROPIC_/i.test(k)));
	return {
		...env,
		ANTHROPIC_BASE_URL: `http://127.0.0.1:${port}`,
		ENABLE_CLAUDEAI_MCP_SERVERS: "0",
		DISABLE_AUTO_COMPACT: "1",
		CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "1",
	};
}

// The bridge's provider-path options, minus its MCP tool server.
function options(cwd, port, extra) {
	return {
		cwd,
		model: MODEL,
		env: childEnv(port),
		tools: ["Read", "Bash"],
		permissionMode: "bypassPermissions",
		allowDangerouslySkipPermissions: true,
		settingSources: [],
		settings: { includeGitInstructions: false },
		systemPrompt: { type: "preset", preset: "claude_code" },
		effort: EFFORT,
		extraArgs: { "thinking-display": "summarized", "strict-mcp-config": null },
		...extra,
	};
}

/** Runs one prompt to its result, or aborts it once its Bash call has started. */
async function turn(step, cwd, port, extra, { abortOnBash = false } = {}) {
	const abortController = new AbortController();
	const q = query({ prompt: `${step.tag} ${step.text}`, options: options(cwd, port, { ...extra, abortController }) });
	let sessionId = null;
	let result = null;
	try {
		for await (const message of q) {
			if (message.session_id) sessionId = message.session_id;
			if (abortOnBash && message.type === "assistant" && message.message.content.some((b) => b.type === "tool_use" && b.name === "Bash")) {
				// Let the tool start, then kill the process as a pi abort does.
				await delay(3000);
				abortController.abort();
			}
			if (message.type === "result") result = message;
		}
	} catch (error) {
		if (!abortOnBash) throw error;
	}
	if (!abortOnBash && result?.is_error) throw new Error(`${step.tag} failed: ${result.subtype} ${JSON.stringify(result.errors ?? result.result)}`);
	return { sessionId, result };
}

async function freePort() {
	return new Promise((resolve, reject) => {
		const server = createServer();
		server.once("error", reject);
		server.listen(0, "127.0.0.1", () => {
			const { port } = server.address();
			server.close(() => resolve(port));
		});
	});
}

async function waitForPort(port) {
	const deadline = Date.now() + 10_000;
	while (Date.now() < deadline) {
		const open = await new Promise((resolve) => {
			const socket = connect(port, "127.0.0.1", () => { socket.end(); resolve(true); });
			socket.once("error", () => resolve(false));
		});
		if (open) return;
		await delay(100);
	}
	throw new Error(`capture proxy did not listen on ${port}`);
}

/** Main-loop /v1/messages requests in send order; CC's toolless side requests are dropped. */
function loadRequests(captureDir) {
	const rows = readFileSync(join(captureDir, "index.jsonl"), "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
	return rows
		.filter((row) => row.path?.startsWith("/v1/messages") && row.status === 200 && (row.tools ?? 0) > 0)
		.map((row) => ({ ...row, body: JSON.parse(readFileSync(join(captureDir, `req-${String(row.n).padStart(4, "0")}.json`), "utf8")) }));
}

const userTexts = (message) => message.role !== "user" ? []
	: typeof message.content === "string" ? [message.content]
	: message.content.filter((block) => block.type === "text").map((block) => block.text);

const opensWith = (message, tag) => userTexts(message).some((text) => text.trimStart().startsWith(tag));

const firstWith = (requests, tag) => requests.findIndex((r) => r.body.messages.some((m) => opensWith(m, tag)));

// --- Prefix comparison, as in diag/probe-rebuild-fidelity.mjs ---

function withoutCacheControl(value) {
	if (Array.isArray(value)) return value.map(withoutCacheControl);
	if (value && typeof value === "object") {
		return Object.fromEntries(Object.entries(value).filter(([k]) => k !== "cache_control").map(([k, v]) => [k, withoutCacheControl(v)]));
	}
	return value;
}

const kinds = (message) => typeof message.content === "string" ? "text" : message.content.map((b) => b.type).join("+");

function contentAsString(message) {
	const c = message.content;
	const plain = Array.isArray(c) && c.length === 1 && c[0].type === "text" && Object.keys(c[0]).length === 2;
	return plain ? { ...message, content: c[0].text } : message;
}

function elements(body) {
	const system = Array.isArray(body.system) ? body.system : body.system ? [{ type: "text", text: body.system }] : [];
	return [
		{ label: "tools", value: withoutCacheControl(body.tools ?? []) },
		...system.flatMap((block, i) => String(block.text ?? "").startsWith(BILLING_HEADER) ? []
			: [{ label: `system[${i}]`, value: withoutCacheControl(block) }]),
		...body.messages.map((message, i) => {
			const value = contentAsString(withoutCacheControl(message));
			return { label: `messages[${i}] ${value.role}/${kinds(value)}`, value };
		}),
	];
}

function firstDiff(a, b, path = "") {
	if (typeof a !== typeof b || Array.isArray(a) !== Array.isArray(b) || (a === null) !== (b === null)) return { path, a, b };
	if (Array.isArray(a)) {
		for (let i = 0; i < Math.min(a.length, b.length); i++) {
			const d = firstDiff(a[i], b[i], `${path}[${i}]`);
			if (d) return d;
		}
		return a.length === b.length ? null : { path: `${path}.length`, a: a.length, b: b.length };
	}
	if (a && typeof a === "object") {
		const ka = Object.keys(a);
		const kb = Object.keys(b);
		for (const k of ka) if (!(k in b)) return { path: `${path}.${k}`, a: a[k], b: "<absent>" };
		for (const k of kb) if (!(k in a)) return { path: `${path}.${k}`, a: "<absent>", b: b[k] };
		for (const k of ka) {
			const d = firstDiff(a[k], b[k], `${path}.${k}`);
			if (d) return d;
		}
		return ka.join() === kb.join() ? null : { path, keyOrder: true, a: ka, b: kb };
	}
	return a === b ? null : { path, a, b };
}

function describeDiff(d) {
	if (d.keyOrder) return `${d.path} key order only: [${d.a}] -> [${d.b}]`;
	if (typeof d.a === "string" && typeof d.b === "string") {
		let i = 0;
		while (i < d.a.length && i < d.b.length && d.a[i] === d.b[i]) i++;
		const around = (s) => JSON.stringify(s.slice(Math.max(0, i - 60), i + 60));
		return `${d.path} differs at char ${i} (length ${d.a.length} -> ${d.b.length})\n      before: ${around(d.a)}\n      after:  ${around(d.b)}`;
	}
	const show = (v) => {
		const s = JSON.stringify(v);
		return s.length > 240 ? `${s.slice(0, 240)}...(${s.length}b)` : s;
	};
	return `${d.path}\n      before: ${show(d.a)}\n      after:  ${show(d.b)}`;
}

/** First element of `prev`'s prefix that `curr` changed, or null when `curr` extends it unchanged. */
function divergence(prev, curr) {
	const before = elements(prev.body);
	const after = elements(curr.body);
	for (let k = 0; k < before.length; k++) {
		if (k >= after.length) return { label: `prompt shrank ${before.length} -> ${after.length} elements` };
		const d = firstDiff(before[k].value, after[k].value);
		if (d) return { label: before[k].label === after[k].label ? before[k].label : `${before[k].label} -> ${after[k].label}`, detail: describeDiff({ ...d, path: `${before[k].label.split(" ")[0]}${d.path}` }) };
	}
	return null;
}

/** Content block types in a request's history, to show which record shapes a run covered. */
function coverage(body) {
	const counts = {};
	for (const message of body.messages) {
		if (typeof message.content === "string") continue;
		for (const block of message.content) counts[block.type] = (counts[block.type] ?? 0) + 1;
	}
	return counts;
}

function cache(prev, curr) {
	const expected = prev.usage.cacheRead + prev.usage.cacheWrite;
	return { cacheRead: curr.usage.cacheRead, cacheWrite: curr.usage.cacheWrite, expected, shortfall: expected - curr.usage.cacheRead };
}

async function run(runIndex) {
	const runDir = join(OUT, `run-${runIndex}`);
	const captureDir = join(runDir, "capture");
	const cwd = join(runDir, "cwd");
	mkdirSync(captureDir, { recursive: true });
	mkdirSync(cwd, { recursive: true });
	const word = () => `${["amber", "cobalt", "saffron", "teal"][Math.floor(Math.random() * 4)]}${Math.random().toString(36).slice(2, 6)}`;
	writeFileSync(join(cwd, "notes-a.txt"), `codeword: ${word()}\n`);
	writeFileSync(join(cwd, "notes-b.txt"), `codeword: ${word()}\n`);

	const port = await freePort();
	const proxyLog = openSync(join(runDir, "proxy.log"), "w");
	const proxy = spawn(process.execPath, [join(REPO, "diag/capture-proxy.mjs"), "--port", String(port), "--out", captureDir], {
		stdio: ["ignore", "ignore", proxyLog],
	});
	let sessionId;
	let forkId;
	let cutAt;
	try {
		await waitForPort(port);
		({ sessionId } = await turn(PLAN[0], cwd, port, {}));
		await turn(PLAN[1], cwd, port, { resume: sessionId });
		await turn(PLAN[2], cwd, port, { resume: sessionId }, { abortOnBash: true });
		// The killed process can still flush records; the fork must not depend on them.
		await delay(2000);

		const records = await getSessionMessages(sessionId, { dir: cwd });
		const p3 = records.findIndex((r) => r.type === "user" && JSON.stringify(r.message).includes("[P3]"));
		if (p3 === -1) throw new Error(`run ${runIndex}: [P3] not found in session ${sessionId}`);
		cutAt = records.slice(0, p3).findLast((r) => r.type === "assistant");
		if (!cutAt) throw new Error(`run ${runIndex}: no assistant record before [P3]`);
		({ sessionId: forkId } = await forkSession(sessionId, { dir: cwd, upToMessageId: cutAt.uuid }));

		await turn(PLAN[3], cwd, port, { resume: forkId });
		await delay(500); // the proxy writes its index after it answers
	} finally {
		proxy.kill();
		closeSync(proxyLog);
	}

	const requests = loadRequests(captureDir);
	const at = Object.fromEntries(PLAN.map((step) => [step.tag, firstWith(requests, step.tag)]));
	const missing = PLAN.filter((step) => at[step.tag] === -1).map((step) => step.tag);
	if (missing.length) throw new Error(`run ${runIndex}: no captured request carries ${missing.join(", ")}; see ${captureDir}`);

	const p1Last = requests[at["[P2]"] - 1];
	const p2First = requests[at["[P2]"]];
	const p2Last = requests[at["[P3]"] - 1];
	const p3First = requests[at["[P3]"]];
	const p4First = requests[at["[P4]"]];
	return {
		runDir,
		sessionId,
		forkId,
		cutAt: cutAt.uuid,
		requests: requests.length,
		control: { ...cache(p1Last, p2First), divergence: divergence(p1Last, p2First) },
		fork: { ...cache(p2Last, p4First), divergence: divergence(p3First, p4First) },
		coverage: coverage(p4First.body),
		p3MessageIndex: p3First.body.messages.findIndex((m) => opensWith(m, "[P3]")),
	};
}

function print(r, runIndex) {
	console.log(`\nrun ${runIndex}  ${r.runDir}`);
	console.log(`  session ${r.sessionId}, fork ${r.forkId} cut at ${r.cutAt}; ${r.requests} main-loop requests`);
	console.log(`  history covered: ${JSON.stringify(r.coverage)}`);
	const row = (name, c) => console.log(`  ${name.padEnd(22)} cacheRead ${String(c.cacheRead).padStart(7)}  expected ${String(c.expected).padStart(7)}  shortfall ${String(c.shortfall).padStart(7)}`);
	row("P2 resume (control)", r.control);
	console.log(`    prefix vs P1's last request: ${r.control.divergence ? `diverges at ${r.control.divergence.label}` : "extended unchanged"}`);
	if (r.control.divergence?.detail) console.log(`      ${r.control.divergence.detail}`);
	row("P4 on fork (test)", r.fork);
	const d = r.fork.divergence;
	const faithful = d?.label.startsWith(`messages[${r.p3MessageIndex}] `);
	console.log(`    prefix vs P3's first request: ${d ? `diverges at ${d.label}` : "identical"}${faithful ? "  (the [P3] prompt: fork is faithful)" : d ? "  (BEFORE the cut: fork changed bytes)" : ""}`);
	if (d?.detail && !faithful) console.log(`      ${d.detail}`);
}

mkdirSync(OUT, { recursive: true });
console.log(`model ${MODEL}, effort ${EFFORT}, ${RUNS} run(s), output ${OUT}`);
const results = [];
for (let i = 1; i <= RUNS; i++) {
	const result = await run(i);
	print(result, i);
	results.push(result);
}
writeFileSync(join(OUT, "report.json"), JSON.stringify({ model: MODEL, effort: EFFORT, results }, null, 1));
console.log(`\nreport: ${join(OUT, "report.json")}`);
