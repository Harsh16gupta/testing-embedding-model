import joplin from 'api';
import { PANEL_TITLE } from './modelConfig';
import { clusterEmbeddings, ClusterResult } from './clustering';

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

const CLUSTER_COLORS = [
	'#6495ed', '#ed6495', '#95ed64', '#ed9564', '#6464ed',
	'#64edd5', '#d564ed', '#eddb64', '#64a5ed', '#ed6464',
];

function renderClusterResults(
	benchmark: {
		initTimeMs: number,
		warmupMs: number,
		timings: number[],
		avgNoWarmup: number,
		totalMs: number,
		noteCount: number,
	},
	clusters: ClusterResult,
	labels: string[],
): string {
	const totalSec = (benchmark.totalMs / 1000).toFixed(1);

	// Group notes by cluster
	const groups = new Map<number, string[]>();
	for (let i = 0; i < clusters.assignments.length; i++) {
		const c = clusters.assignments[i];
		if (!groups.has(c)) groups.set(c, []);
		groups.get(c)!.push(labels[i]);
	}

	// Build cluster cards
	const sortedKeys = Array.from(groups.keys()).sort((a, b) => a - b);
	let clusterCards = '';
	for (const cid of sortedKeys) {
		const notes = groups.get(cid)!;
		const color = CLUSTER_COLORS[cid % CLUSTER_COLORS.length];
		const noteList = notes
			.map(n => `<li class="cluster-note">${escapeHtml(n)}</li>`)
			.join('');

		clusterCards += `
			<div class="cluster-card">
				<div class="cluster-header">
					<span class="cluster-dot" style="background:${color}"></span>
					<span class="cluster-label">Cluster ${cid + 1}</span>
					<span class="cluster-count">${notes.length} notes</span>
				</div>
				<ul class="cluster-notes">${noteList}</ul>
			</div>
		`;
	}

	const kScoreRows = clusters.kScores
		.map(ks => {
			const isBest = ks.k === clusters.k;
			const cls = isBest ? 'k-best' : '';
			return `<li class="${cls}"><span>k=${ks.k}</span><span>${ks.score.toFixed(3)}</span></li>`;
		})
		.join('');

	return `
		<h1 class="title">${PANEL_TITLE}</h1>

		<h2 class="section-title">Embedding Benchmark</h2>
		<ul class="metrics">
			<li><span>Count</span><span>${benchmark.noteCount}</span></li>
			<li><span>Load</span><span>${(benchmark.initTimeMs / 1000).toFixed(1)} s</span></li>
			<li><span>Warmup</span><span>${Math.round(benchmark.warmupMs)} ms</span></li>
			<li><span>Avg (excl. first)</span><span>${benchmark.avgNoWarmup} ms</span></li>
			<li><span>Total embed time</span><span>${totalSec} s</span></li>
		</ul>

		<h2 class="section-title">UMAP + K-Means Clustering</h2>
		<ul class="metrics">
			<li><span>Best k</span><span>${clusters.k}</span></li>
			<li><span>Silhouette score</span><span>${clusters.silhouetteAvg.toFixed(3)}</span></li>
			<li><span>UMAP time</span><span>${Math.round(clusters.umapTimeMs)} ms</span></li>
			<li><span>K-selection time</span><span>${Math.round(clusters.kmeansTimeMs)} ms</span></li>
		</ul>

		<h2 class="section-title">Clusters</h2>
		${clusterCards}

		<h2 class="section-title">K-Score Table</h2>
		<ul class="metrics k-scores">
			<li class="k-header"><span>k</span><span>Silhouette</span></li>
			${kScoreRows}
		</ul>
	`;
}

function escapeHtml(s: string): string {
	return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function wrap(html: string): string {
	return `<div class="scroll-wrap">${html}</div>`;
}

joplin.plugins.register({
	onStart: async function () {
		const installDir = await joplin.plugins.installationDir();

		const panel = await joplin.views.panels.create('embedding-benchmark-panel');
		await joplin.views.panels.addScript(panel, './panel.css');
		await joplin.views.panels.setHtml(panel, wrap(renderLoading('Reading corpus…', 5)));
		await joplin.views.panels.show(panel);

		let benchmarkItems: BenchmarkItem[];
		try {
			benchmarkItems = applyMaxNotes(await loadCorpus(installDir));
		} catch (e) {
			const msg = e instanceof Error ? e.message : String(e);
			logErr('corpus:', msg);
			await joplin.views.panels.setHtml(panel, wrap(`
				<h1 class="title">${PANEL_TITLE}</h1>
				<p class="err">${msg}</p>
			`));
			return;
		}

		const worker = new Worker(`${installDir}/worker.js`);

		let initTimeMs = 0;
		let warmupMs = 0;
		const timings: number[] = new Array(benchmarkItems.length).fill(0);
		const embeddings: number[][] = new Array(benchmarkItems.length);

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
					await joplin.views.panels.setHtml(panel, wrap(renderLoading(
						`Embedding 1/${total}: ${benchmarkItems[0].label}`,
						25
					)));

					worker.postMessage({
						type: 'embed',
						text: benchmarkItems[0].text,
						index: 0,
						label: benchmarkItems[0].label,
					});
				} else {
					logErr('load:', data.error);
					await joplin.views.panels.setHtml(panel, wrap(`
						<h1 class="title">${PANEL_TITLE}</h1>
						<p class="err">Model load failed: ${data.error}</p>
					`));
				}
				return;
			}

			if (data.type === 'embed-result') {
				if (data.success) {
					timings[data.index] = data.inferenceTime;
					embeddings[data.index] = data.embedding;
					const i = data.index;
					const total = benchmarkItems.length;
					const next = i + 1;

					if (next < total) {
						const progress = 25 + ((i + 1) / total) * 70;
						await joplin.views.panels.setHtml(panel, wrap(renderLoading(
							`Embedding ${next + 1}/${total}: ${benchmarkItems[next].label}`,
							progress
						)));
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

						const benchmarkData = {
							initTimeMs,
							warmupMs,
							timings,
							avgNoWarmup,
							totalMs,
							noteCount: benchmarkItems.length,
						};

						// clustering
						await joplin.views.panels.setHtml(panel,
							wrap(renderLoading('Running UMAP + K-Means clustering…', 97))
						);

						try {
							const clusterResult = clusterEmbeddings(embeddings);
							const noteLabels = benchmarkItems.map(item => item.label);

							await joplin.views.panels.setHtml(panel,
								wrap(renderClusterResults(benchmarkData, clusterResult, noteLabels))
							);
						} catch (e) {
							const msg = e instanceof Error ? e.message : String(e);
							logErr('clustering:', msg);
							await joplin.views.panels.setHtml(panel,
								wrap(renderResults(benchmarkData) +
								`<p class="err">Clustering failed: ${msg}</p>`)
							);
						}
					}
				} else {
					logErr('embed:', data.label, data.error);
					await joplin.views.panels.setHtml(panel, wrap(`
						<h1 class="title">${PANEL_TITLE}</h1>
						<p class="err">Embed failed (${data.label}): ${data.error}</p>
					`));
				}
			}
		};

		await joplin.views.panels.setHtml(panel, wrap(renderLoading('Loading model…', 10)));

		worker.postMessage({ type: 'load' });
	},
});
