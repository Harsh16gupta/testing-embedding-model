import joplin from 'api';
import { PANEL_TITLE } from './modelConfig';

const LOG = '[hg]';

function logErr(...args: unknown[]) {
	console.error(LOG, ...args);
}

const CORPUS_FILENAME = 'corpus.jsonl';

const MAX_NOTES = 0;
const MAX_CHARS_PER_NOTE = 200000;

type BenchmarkItem = { text: string; label: string };

function parseJsonlLine(line: string, lineIndex: number): BenchmarkItem | null {
	const trimmed = line.trim();
	if (!trimmed) return null;
	try {
		const o = JSON.parse(trimmed) as Record<string, unknown>;
		if (typeof o.text === 'string' && o.text.length > 0) {
			const label = typeof o.label === 'string' ? o.label : `Note ${lineIndex + 1}`;
			return { text: clampText(o.text), label };
		}
		const body = typeof o.body === 'string' ? o.body : '';
		const title = typeof o.title === 'string' ? o.title : '';
		const combined = title && body ? `${title}\n\n${body}` : body || title;
		if (!combined) return null;
		const label = typeof o.label === 'string' ? o.label : (title || `Note ${lineIndex + 1}`);
		return { text: clampText(combined), label };
	} catch {
		return null;
	}
}

function clampText(s: string): string {
	if (s.length <= MAX_CHARS_PER_NOTE) return s;
	return s.slice(0, MAX_CHARS_PER_NOTE);
}

function parseJsonl(raw: string): BenchmarkItem[] {
	const out: BenchmarkItem[] = [];
	const lines = raw.split(/\r?\n/);
	for (let i = 0; i < lines.length; i++) {
		const item = parseJsonlLine(lines[i], i);
		if (item) out.push(item);
	}
	return out;
}

async function loadCorpus(installDir: string): Promise<BenchmarkItem[]> {
	const res = await fetch(`${installDir}/${CORPUS_FILENAME}`);
	if (!res.ok) throw new Error(`corpus.jsonl (${res.status})`);
	const items = parseJsonl(await res.text());
	if (!items.length) throw new Error('corpus.jsonl is empty or invalid');
	return items;
}

function applyMaxNotes(items: BenchmarkItem[]): BenchmarkItem[] {
	if (MAX_NOTES > 0 && items.length > MAX_NOTES) return items.slice(0, MAX_NOTES);
	return items;
}

function renderLoading(statusText: string, progress: number): string {
	const coffeeMessage = progress > 0 && progress < 95 ? '<div class="coffee">Grab a coffee, this will take a while...</div>' : '';
	
	return `
		<h1 class="title">${PANEL_TITLE}</h1>
		<p class="muted">${statusText}</p>
		<div class="progress"><div class="progress-fill" style="width: ${progress}%"></div></div>
		${coffeeMessage}
	`;
}

function renderResults(data: {
	initTimeMs: number,
	warmupMs: number,
	timings: number[],
	avgNoWarmup: number,
	totalMs: number,
	noteCount: number,
}): string {
	const totalSec = (data.totalMs / 1000).toFixed(1);

	return `
		<h1 class="title">${PANEL_TITLE}</h1>
		<ul class="metrics">
			<li><span>Count</span><span>${data.noteCount}</span></li>
			<li><span>Load</span><span>${(data.initTimeMs / 1000).toFixed(1)} s</span></li>
			<li><span>Warmup</span><span>${Math.round(data.warmupMs)} ms</span></li>
			<li><span>Avg (excl. first)</span><span>${data.avgNoWarmup} ms</span></li>
			<li><span>Total embed time</span><span>${totalSec} s </span></li>
		</ul>
	`;
}

joplin.plugins.register({
	onStart: async function () {
		const installDir = await joplin.plugins.installationDir();

		const panel = await joplin.views.panels.create('embedding-benchmark-panel');
		await joplin.views.panels.addScript(panel, './panel.css');
		await joplin.views.panels.setHtml(panel, renderLoading('Reading corpus…', 5));
		await joplin.views.panels.show(panel);

		let benchmarkItems: BenchmarkItem[];
		try {
			benchmarkItems = applyMaxNotes(await loadCorpus(installDir));
		} catch (e) {
			const msg = e instanceof Error ? e.message : String(e);
			logErr('corpus:', msg);
			await joplin.views.panels.setHtml(panel, `
				<h1 class="title">${PANEL_TITLE}</h1>
				<p class="err">${msg}</p>
			`);
			return;
		}

		const worker = new Worker(`${installDir}/worker.js`);

		let initTimeMs = 0;
		let warmupMs = 0;
		const timings: number[] = new Array(benchmarkItems.length).fill(0);

		worker.onerror = (err) => {
			logErr('worker (non-fatal):', err.message || err);
		};

		worker.onmessage = async (event) => {
			const data = event.data;

			if (data.type === 'load-result') {
				if (data.success) {
					initTimeMs = data.loadTime;
					warmupMs = data.warmupTime;

					const total = benchmarkItems.length;
					await joplin.views.panels.setHtml(panel, renderLoading(
						`Embedding 1/${total}: ${benchmarkItems[0].label}`,
						25
					));

					worker.postMessage({
						type: 'embed',
						text: benchmarkItems[0].text,
						index: 0,
						label: benchmarkItems[0].label,
					});
				} else {
					logErr('load:', data.error);
					await joplin.views.panels.setHtml(panel, `
						<h1 class="title">${PANEL_TITLE}</h1>
						<p class="err">Model load failed: ${data.error}</p>
					`);
				}
				return;
			}

			if (data.type === 'embed-result') {
				if (data.success) {
					timings[data.index] = data.inferenceTime;
					const i = data.index;
					const total = benchmarkItems.length;
					const next = i + 1;

					if (next < total) {
						const progress = 25 + ((i + 1) / total) * 70;
						await joplin.views.panels.setHtml(panel, renderLoading(
							`Embedding ${next + 1}/${total}: ${benchmarkItems[next].label}`,
							progress
						));
						worker.postMessage({
							type: 'embed',
							text: benchmarkItems[next].text,
							index: next,
							label: benchmarkItems[next].label,
						});
					} else {
						const totalMs = timings.reduce((a, b) => a + b, 0);
						let avgNoWarmup: number;
						if (timings.length > 1) {
							avgNoWarmup = Math.round(
								timings.slice(1).reduce((a, b) => a + b, 0) / (timings.length - 1)
							);
						} else {
							avgNoWarmup = Math.round(timings[0] || 0);
						}

						await joplin.views.panels.setHtml(panel, renderResults({
							initTimeMs,
							warmupMs,
							timings,
							avgNoWarmup,
							totalMs,
							noteCount: benchmarkItems.length,
						}));
					}
				} else {
					logErr('embed:', data.label, data.error);
					await joplin.views.panels.setHtml(panel, `
						<h1 class="title">${PANEL_TITLE}</h1>
						<p class="err">Embed failed (${data.label}): ${data.error}</p>
					`);
				}
			}
		};

		await joplin.views.panels.setHtml(panel, renderLoading('Loading model…', 10));

		worker.postMessage({ type: 'load' });
	},
});
