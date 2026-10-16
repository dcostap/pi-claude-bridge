#!/usr/bin/env node
// Does the turn after an abort read back the prompt cache?
//
// Runs one pi session in RPC mode through the bridge under diag/capture-proxy.mjs
// and aborts a prompt while its tool runs, as Escape does:
//
//   P1  clean start
//   P2  reuse    control
//   P3  reuse    runs `sleep 45`; the probe aborts 3 s after the tool starts
//   P4  fork     the turn after the abort
//   P5  reuse    control after the fork
//
// For P4 it reports the sync path, cacheRead against what P3's last request left
// cached (cacheRead + cacheWrite), and the first element of that request's prompt
// that P4's request changed. P3's own prompt message can differ: Claude Code's
// hidden attachments for it are cut with the aborted turn. Everything before it
// should be identical. Pass --bridge to measure another checkout, such as one
// without the fork path, where P4 rebuilds.
//
//   node diag/probe-abort-fork.mjs [--model claude-bridge/claude-haiku-5-5]
//       [--thinking low|off] [--bridge DIR] [--pi pi] [--out DIR]

import { spawn } from "node:child_process";
import { closeSync, mkdirSync, mkdtempSync, openSync, readFileSync, writeFileSync } from "node:fs";
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
const OUT = flag("out") ?? mkdtempSync(join(tmpdir(), "abort-fork-"));

const BILLING_HEADER = "x-anthropic-billing-header";

const PLAN = [
	{ tag: "[P1]", text: "Use the read tool to read notes-a.txt and notes-b.txt. Issue both read calls in a single message. Then reply with just the two codewords." },
	{ tag: "[P2]", text: "Use the bash tool to run: ls -1 | wc -l. Reply with just the number." },
	{ tag: "[P3]", abort: true, text: "Use the bash tool to run: sleep 45. Then reply with just: slept." },
	{ tag: "[P4]", text: "Use the bash tool to run: cat notes-b.txt. Reply with just its contents." },
	{ tag: "[P5]", text: "Reply with just: ok." },
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

/** Drive pi over RPC: one prompt at a time, waiting for agent_settled. */
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
	const settled = [];
	try {
		for (const step of PLAN) {
			const done = new Promise((resolve, reject) => {
				const timer = setTimeout(() => reject(new Error(`${step.tag} did not settle in 5 minutes`)), 5 * 60_000);
				let aborting = false;
				onEvent = (event) => {
					if (step.abort && !aborting && event.type === "tool_execution_start" && String(event.args?.command ?? "").includes("sleep")) {
						aborting = true;
						setTimeout(() => send({ type: "abort" }), 3000);
					}
					if (event.type === "agent_settled") {
						clearTimeout(timer);
						resolve(event);
					}
				};
				exited.then((code) => reject(new Error(`pi exited ${code} during ${step.tag}`)));
			});
			send({ type: "prompt", message: `${step.tag} ${step.text}` });
			settled.push({ tag: step.tag, aborted: (await done).aborted });
		}
	} finally {
		pi.stdin.end();
		pi.kill();
	}
	return settled;
}

function loadRequests(captureDir) {
	const rows = readFileSync(join(captureDir, "index.jsonl"), "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
	return rows
		.filter((row) => row.path?.startsWith("/v1/messages") && row.status === 200 && (row.tools ?? 0) > 0)
		.map((row) => ({ ...row, body: JSON.parse(readFileSync(join(captureDir, `req-${String(row.n).padStart(4, "0")}.json`), "utf8")) }));
}

function syncPaths(debugLog) {
	const re = /syncResult: path=(\S+)(.*)$/;
	return readFileSync(debugLog, "utf8").split("\n").map((line) => line.match(re)).filter(Boolean).map((m) => {
		const kind = m[2].match(/ (first|preserved|rotated-post-abort|preserve-shared)\b/);
		return kind ? `${m[1]} ${kind[1]}` : m[1];
	});
}

const userTexts = (message) => message.role !== "user" ? []
	: typeof message.content === "string" ? [message.content]
	: message.content.filter((block) => block.type === "text").map((block) => block.text);
const carries = (body, tag) => body.messages.some((m) => userTexts(m).some((text) => text.includes(tag)));

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
			const text = (v) => JSON.stringify(v).slice(0, 200);
			return { at: k, of: before.length, label: before[k].label, before: text(before[k].value), after: text(after[k].value) };
		}
	}
	return { at: before.length, of: before.length, label: null };
}

mkdirSync(OUT, { recursive: true });
const captureDir = join(OUT, "capture");
const cwd = join(OUT, "cwd");
mkdirSync(captureDir, { recursive: true });
mkdirSync(cwd, { recursive: true });
const word = () => `${["amber", "cobalt", "saffron", "teal"][Math.floor(Math.random() * 4)]}${Math.random().toString(36).slice(2, 6)}`;
writeFileSync(join(cwd, "notes-a.txt"), `codeword: ${word()}\n`);
writeFileSync(join(cwd, "notes-b.txt"), `codeword: ${word()}\n`);
console.log(`model ${MODEL}, thinking ${THINKING}, bridge ${BRIDGE}, output ${OUT}`);

const port = await freePort();
const proxyLog = openSync(join(OUT, "proxy.log"), "w");
const proxy = spawn(process.execPath, [join(REPO, "diag/capture-proxy.mjs"), "--port", String(port), "--out", captureDir], { stdio: ["ignore", "ignore", proxyLog] });
const debugLog = join(OUT, "bridge-debug.log");
const piErr = openSync(join(OUT, "pi.stderr"), "w");
let settled;
try {
	await waitForPort(port);
	settled = await drive(cwd, { ...process.env, ANTHROPIC_BASE_URL: `http://127.0.0.1:${port}`, CLAUDE_BRIDGE_DEBUG: "1", CLAUDE_BRIDGE_DEBUG_PATH: debugLog }, piErr);
} finally {
	proxy.kill();
	closeSync(proxyLog);
	closeSync(piErr);
}

const paths = syncPaths(debugLog);
const requests = loadRequests(captureDir);
console.log(`settled: ${settled.map((s) => `${s.tag}${s.aborted ? " (aborted)" : ""}`).join(", ")}`);
console.log(`sync paths: ${paths.join(", ")}`);
console.log(`${requests.length} main-loop requests\n`);

for (const tag of ["[P4]", "[P5]"]) {
	const i = requests.findIndex((r) => carries(r.body, tag));
	if (i <= 0) {
		console.log(`${tag}: no request found`);
		continue;
	}
	const prev = requests[i - 1];
	const curr = requests[i];
	const expected = (prev.usage?.cacheRead ?? 0) + (prev.usage?.cacheWrite ?? 0);
	const d = divergence(prev, curr);
	console.log(`${tag} cacheRead ${curr.usage?.cacheRead} of ${expected} cached by request ${prev.n} (shortfall ${expected - (curr.usage?.cacheRead ?? 0)}), cacheWrite ${curr.usage?.cacheWrite}`);
	console.log(d.label === null
		? `      prefix identical for all ${d.of} elements`
		: `      first change at element ${d.at} of ${d.of}: ${d.label}\n        before: ${d.before}\n        after:  ${d.after}`);
}
writeFileSync(join(OUT, "report.json"), JSON.stringify({ model: MODEL, thinking: THINKING, bridge: BRIDGE, settled, paths, requests: requests.map((r) => ({ n: r.n, usage: r.usage })) }, null, 1));
