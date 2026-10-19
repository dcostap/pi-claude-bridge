#!/usr/bin/env node
// Does a compaction summary read the conversation back from the prompt cache?
//
// Runs one pi session in RPC mode through the bridge under diag/capture-proxy.mjs:
//
//   P0     reads of ~20 KB files, one per turn (--early N, default 1): whole
//          turns before the cut point, so the summary covers history as well
//          as the split turn's prefix. pi's summary shows tool results
//          truncated, so the cached path wins only with enough history.
//   P1-P3  read tool calls over five more such files, enough for pi's default
//          keepRecentTokens (20K) to cut inside P1's turn
//   P4     a bash tool call
//   /compact
//   P5     a question only the summary can answer
//
// For the summary request it reports what the cache served against what the last
// request before it left cached (cacheRead + cacheWrite), the first element of
// that request that the summary request changed, and the summary's token cost in
// input-token units. --native runs the same session with provider.cachedCompaction
// off, for pi's own summary; --compare runs both.
//
//   node diag/probe-cached-compact.mjs [--model claude-bridge/claude-haiku-5-5]
//       [--thinking low|off] [--bridge DIR] [--pi pi] [--out DIR] [--early N] [--native | --compare]

import { spawn } from "node:child_process";
import { closeSync, existsSync, mkdirSync, mkdtempSync, openSync, readFileSync, writeFileSync } from "node:fs";
import { connect, createServer } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";

const REPO = join(dirname(fileURLToPath(import.meta.url)), "..");

const args = process.argv.slice(2);
const flag = (name, fallback) => {
	const i = args.indexOf(`--${name}`);
	return i === -1 ? fallback : args[i + 1];
};
const MODEL = flag("model", "claude-bridge/claude-haiku-5-5");
const THINKING = flag("thinking", "low");
const BRIDGE = flag("bridge", REPO);
const PI = flag("pi", "pi");
const OUT = flag("out") ?? mkdtempSync(join(tmpdir(), "cached-compact-"));
const MODES = args.includes("--compare") ? ["cached", "native"] : args.includes("--native") ? ["native"] : ["cached"];

const BILLING_HEADER = "x-anthropic-billing-header";
const CHECKPOINT_REQUEST = "[Context checkpoint request]";

const EARLY = Number(flag("early", "1"));
const EARLY_FILES = Array.from({ length: EARLY }, (_, i) => `f${i}`);
const FILES = ["a", "b", "c", "d", "e", ...EARLY_FILES];
const PLAN = [
	...EARLY_FILES.map((name, i) => ({ tag: `[P0.${i}]`, text: `Use the read tool to read notes-${name}.txt. Then reply with just its codeword.` })),
	{ tag: "[P1]", text: "Use the read tool to read notes-a.txt and notes-b.txt. Issue both read calls in a single message. Then reply with just the two codewords." },
	{ tag: "[P2]", text: "Use the read tool to read notes-c.txt and notes-d.txt. Issue both read calls in a single message. Then reply with just the two codewords." },
	{ tag: "[P3]", text: "Use the read tool to read notes-e.txt. Then reply with just its codeword." },
	{ tag: "[P4]", text: "Use the bash tool to run: ls -1 | wc -l. Reply with just the number." },
	{ compact: true },
	{ tag: "[P5]", text: "What were all the codewords, from every notes file? Reply with just them." },
];

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

/** Drive pi over RPC: one step at a time. */
async function drive(cwd, env, log) {
	const pi = spawn(PI, ["--mode", "rpc", "--no-session", "-ne", "-e", BRIDGE, "--model", MODEL,
		...(THINKING === "off" ? [] : ["--thinking", THINKING])], { cwd, env, stdio: ["pipe", "pipe", log] });
	const send = (command) => pi.stdin.write(JSON.stringify(command) + "\n");
	let buffer = "";
	let onEvent = () => {};
	const exited = new Promise((resolve) => pi.once("exit", resolve));
	pi.stdout.on("data", (chunk) => {
		buffer += chunk;
		let nl;
		while ((nl = buffer.indexOf("\n")) !== -1) {
			const line = buffer.slice(0, nl).trim();
			buffer = buffer.slice(nl + 1);
			if (!line) continue;
			try { onEvent(JSON.parse(line)); } catch { /* not JSON */ }
		}
	});
	const steps = [];
	try {
		for (const step of PLAN) {
			const done = new Promise((resolve, reject) => {
				const timer = setTimeout(() => reject(new Error(`${step.tag ?? "compact"} did not finish in 5 minutes`)), 5 * 60_000);
				onEvent = (event) => {
					const finished = step.compact
						? event.type === "response" && event.command === "compact"
						: event.type === "agent_settled";
					if (finished) {
						clearTimeout(timer);
						resolve(event);
					}
				};
				exited.then((code) => reject(new Error(`pi exited ${code} during ${step.tag ?? "compact"}`)));
			});
			const started = Date.now();
			send(step.compact ? { type: "compact" } : { type: "prompt", message: `${step.tag} ${step.text}` });
			const event = await done;
			steps.push({ step: step.tag ?? "compact", ms: Date.now() - started, ...(step.compact ? { success: event.success, error: event.error, summary: event.data?.summary } : {}) });
		}
	} finally {
		pi.stdin.end();
		pi.kill();
	}
	return steps;
}

function loadRequests(captureDir) {
	const rows = readFileSync(join(captureDir, "index.jsonl"), "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
	return rows
		.filter((row) => row.path?.startsWith("/v1/messages") && !row.path.includes("count_tokens") && row.status === 200)
		.map((row) => ({ ...row, body: JSON.parse(readFileSync(join(captureDir, `req-${String(row.n).padStart(4, "0")}.json`), "utf8")) }));
}

const userTexts = (message) => message.role !== "user" ? []
	: typeof message.content === "string" ? [message.content]
	: message.content.filter((block) => block.type === "text").map((block) => block.text);
// Claude Code may follow the prompt with system-role messages, so search every message.
const anyUserText = (r, test) => r.body.messages.some((m) => userTexts(m).some(test));
const isCachedSummary = (r) => anyUserText(r, (text) => text.includes(CHECKPOINT_REQUEST));
const isNativeSummary = (r) => (r.tools ?? 0) === 0 && anyUserText(r, (text) => /^(<conversation>|# Conversation)\n/.test(text));
const isMainLoop = (r) => (r.tools ?? 0) > 0 && !isCachedSummary(r);

function withoutCacheControl(value) {
	if (Array.isArray(value)) return value.map(withoutCacheControl);
	if (value && typeof value === "object") {
		return Object.fromEntries(Object.entries(value).filter(([k]) => k !== "cache_control").map(([k, v]) => [k, withoutCacheControl(v)]));
	}
	return value;
}

function contentAsString(message) {
	const c = message.content;
	const plain = Array.isArray(c) && c.length === 1 && c[0].type === "text" && Object.keys(c[0]).length === 2;
	return plain ? { ...message, content: c[0].text } : message;
}

const kinds = (message) => typeof message.content === "string" ? "text" : message.content.map((b) => b.type).join("+");

function elements(body) {
	const system = Array.isArray(body.system) ? body.system : body.system ? [{ type: "text", text: body.system }] : [];
	return [
		{ label: "tools", value: withoutCacheControl(body.tools ?? []) },
		{ label: "thinking", value: body.thinking ?? null },
		...system.flatMap((block, i) => String(block.text ?? "").startsWith(BILLING_HEADER) ? []
			: [{ label: `system[${i}]`, value: withoutCacheControl(block) }]),
		...body.messages.map((message, i) => {
			const value = contentAsString(withoutCacheControl(message));
			return { label: `messages[${i}] ${value.role}/${kinds(value)}`, value };
		}),
	];
}

/** The first element of `prev` that `curr` changed, and how many came before it. */
function divergence(prev, curr) {
	const before = elements(prev.body);
	const after = elements(curr.body);
	for (let k = 0; k < before.length; k++) {
		if (k >= after.length) return { at: k, of: before.length, label: `prompt shrank to ${after.length} elements` };
		if (JSON.stringify(before[k].value) !== JSON.stringify(after[k].value)) {
			const text = (v) => JSON.stringify(v).slice(0, 300);
			return { at: k, of: before.length, label: before[k].label, before: text(before[k].value), after: text(after[k].value) };
		}
	}
	return { at: before.length, of: before.length, label: null };
}

/** Input-token units: what the prompt cost relative to sending it all uncached. */
function cost(usage, writeRate) {
	if (!usage) return null;
	return Math.round(usage.input + usage.cacheRead * 0.1 + usage.cacheWrite * writeRate);
}

async function run(mode) {
	const out = join(OUT, mode);
	const captureDir = join(out, "capture");
	const cwd = join(out, "cwd");
	mkdirSync(captureDir, { recursive: true });
	mkdirSync(cwd, { recursive: true });
	const word = () => `${["amber", "cobalt", "saffron", "teal"][Math.floor(Math.random() * 4)]}${Math.random().toString(36).slice(2, 6)}`;
	// Under Claude Code's inline limit for a tool result, which saves larger ones to a file.
	const filler = (name) => Array.from({ length: 450 }, (_, i) => `${name} filler line ${i}, padding the file out to size`).join("\n");
	for (const name of FILES) writeFileSync(join(cwd, `notes-${name}.txt`), `codeword: ${word()}\n${filler(name)}\n`);
	if (mode === "native") {
		mkdirSync(join(cwd, ".pi"), { recursive: true });
		writeFileSync(join(cwd, ".pi", "claude-bridge.json"), JSON.stringify({ provider: { cachedCompaction: false } }));
	}

	const port = await freePort();
	const proxyLog = openSync(join(out, "proxy.log"), "w");
	const proxy = spawn(process.execPath, [join(REPO, "diag/capture-proxy.mjs"), "--port", String(port), "--out", captureDir], { stdio: ["ignore", "ignore", proxyLog] });
	const debugLog = join(out, "bridge-debug.log");
	const piErr = openSync(join(out, "pi.stderr"), "w");
	let steps;
	try {
		await waitForPort(port);
		steps = await drive(cwd, { ...process.env, ANTHROPIC_BASE_URL: `http://127.0.0.1:${port}`, CLAUDE_BRIDGE_DEBUG: "1", CLAUDE_BRIDGE_DEBUG_PATH: debugLog }, piErr);
	} finally {
		proxy.kill();
		closeSync(proxyLog);
		closeSync(piErr);
	}

	const log = existsSync(debugLog) ? readFileSync(debugLog, "utf8").split("\n") : [];
	const pick = (re) => log.filter((line) => re.test(line)).map((line) => line.replace(/^\[[^\]]+\] \[\w+\] /, ""));
	const requests = loadRequests(captureDir);
	const ttl = Number(pick(/summary checkpoint .*cache ttl (\d+)m/).at(-1)?.match(/cache ttl (\d+)m/)?.[1] ?? 5);
	const writeRate = ttl >= 60 ? 2 : 1.25;

	console.log(`\n=== ${mode} (${out}) ===`);
	for (const s of steps) console.log(`${s.step}: ${s.ms} ms${s.success === undefined ? "" : s.success ? " ok" : ` FAILED ${s.error}`}`);
	for (const line of pick(/summary checkpoint|cached summary|compaction summary call|compact summary: (spawn|done)/)) console.log(`  log: ${line}`);

	// In cached mode a split turn's prefix still goes through pi's summary.
	const summaries = requests.filter(mode === "cached" ? (r) => isCachedSummary(r) || isNativeSummary(r) : isNativeSummary);
	const report = { mode, steps, ttlMinutes: ttl, summaries: [] };
	for (const curr of summaries) {
		const i = requests.indexOf(curr);
		const prev = requests.slice(0, i).filter(isMainLoop).at(-1);
		const expected = prev ? (prev.usage?.cacheRead ?? 0) + (prev.usage?.cacheWrite ?? 0) : 0;
		const u = curr.usage;
		console.log(`${isCachedSummary(curr) ? "cached" : "pi's"} summary request ${curr.n}: input ${u?.input} cacheRead ${u?.cacheRead} cacheWrite ${u?.cacheWrite} → ~${cost(u, writeRate)} input-token units`);
		const entry = { n: curr.n, usage: u, cost: cost(u, writeRate) };
		if (prev && isCachedSummary(curr)) {
			const d = divergence(prev, curr);
			console.log(`  vs request ${prev.n} (${expected} cached): shortfall ${expected - (u?.cacheRead ?? 0)}`);
			console.log(d.label === null
				? `  prefix identical for all ${d.of} elements`
				: `  first change at element ${d.at} of ${d.of}: ${d.label}\n    before: ${d.before}\n    after:  ${d.after}`);
			Object.assign(entry, { prev: prev.n, expected, divergence: d });
		}
		report.summaries.push(entry);
	}
	if (!summaries.length) console.log("no summary request found");
	const compact = steps.find((s) => s.step === "compact");
	if (compact?.summary) console.log(`summary text (${compact.summary.length} chars):\n${compact.summary.slice(0, 1500)}`);
	const after = requests.filter(isMainLoop).at(-1);
	if (after) console.log(`P5 request ${after.n}: input ${after.usage?.input} cacheRead ${after.usage?.cacheRead} cacheWrite ${after.usage?.cacheWrite}`);
	writeFileSync(join(out, "report.json"), JSON.stringify({ ...report, requests: requests.map((r) => ({ n: r.n, tools: r.tools, messages: r.messages, usage: r.usage })) }, null, 1));
	return report;
}

mkdirSync(OUT, { recursive: true });
console.log(`model ${MODEL}, thinking ${THINKING}, early turns ${EARLY}, bridge ${BRIDGE}, output ${OUT}`);
const reports = [];
for (const mode of MODES) reports.push(await run(mode));
if (reports.length === 2) {
	const total = (r) => r.summaries.reduce((sum, s) => sum + (s.cost ?? 0), 0);
	console.log(`\nsummary prompt cost: cached ~${total(reports[0])} vs native ~${total(reports[1])} input-token units`);
}
