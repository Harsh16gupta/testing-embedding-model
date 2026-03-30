// UMAP dimensionality reduction + K-Means clustering
// Uses DruidJS for UMAP and ml-kmeans for clustering

// @ts-ignore
import * as druid from '@saehrimnir/druidjs';
// @ts-ignore
import { kmeans } from 'ml-kmeans';

// UMAP config
const UMAP_NEIGHBORS = 15;
const UMAP_DIM = 5;
const UMAP_MIN_DIST = 0.0;
const UMAP_SEED = 42;
const UMAP_ITERS = 350;

function euclidean(a: number[], b: number[]): number {
	let sum = 0;
	for (let i = 0; i < a.length; i++) {
		const d = a[i] - b[i];
		sum += d * d;
	}
	return Math.sqrt(sum);
}

// silhouette score — measures how well-separated the clusters are
// returns [-1, 1], higher = better
function silhouetteScore(points: number[][], labels: number[]): number {
	const n = points.length;
	if (n <= 1) return 0;

	// pairwise distances (fine for small N after UMAP)
	const dist: number[][] = new Array(n);
	for (let i = 0; i < n; i++) {
		dist[i] = new Array(n);
		for (let j = 0; j < n; j++) {
			dist[i][j] = i === j ? 0 : euclidean(points[i], points[j]);
		}
	}

	// group indices by cluster
	const groups = new Map<number, number[]>();
	for (let i = 0; i < n; i++) {
		const c = labels[i];
		if (!groups.has(c)) groups.set(c, []);
		groups.get(c)!.push(i);
	}
	const cids = Array.from(groups.keys());

	let total = 0;
	for (let i = 0; i < n; i++) {
		const mine = labels[i];
		const myGroup = groups.get(mine)!;

		// a(i) — avg distance to own cluster
		let a = 0;
		if (myGroup.length > 1) {
			for (const j of myGroup) {
				if (j !== i) a += dist[i][j];
			}
			a /= (myGroup.length - 1);
		}

		// b(i) — min avg distance to any other cluster
		let b = Infinity;
		for (const cid of cids) {
			if (cid === mine) continue;
			const members = groups.get(cid)!;
			let avg = 0;
			for (const j of members) avg += dist[i][j];
			avg /= members.length;
			if (avg < b) b = avg;
		}

		if (b !== Infinity) {
			total += (b - a) / Math.max(a, b);
		}
	}

	return total / n;
}

export interface ClusterResult {
	assignments: number[];
	k: number;
	silhouetteAvg: number;
	kScores: { k: number; score: number }[];
	umapTimeMs: number;
	kmeansTimeMs: number;
	reducedPoints: number[][];
}

export function clusterEmbeddings(embeddings: number[][]): ClusterResult {
	const N = embeddings.length;

	if (N < 2) {
		return {
			assignments: new Array(N).fill(0),
			k: 1, silhouetteAvg: 0, kScores: [],
			umapTimeMs: 0, kmeansTimeMs: 0,
			reducedPoints: embeddings,
		};
	}

	// --- UMAP reduction ---
	const t0 = performance.now();
	const neighbors = Math.min(UMAP_NEIGHBORS, N - 1);

	let reduced: number[][];

	if (N <= 10) {
		// too few points for UMAP, just use raw vectors
		reduced = embeddings.map(v => v.slice());
	} else {
		const umap = new druid.UMAP(embeddings, {
			n_neighbors: neighbors,
			min_dist: UMAP_MIN_DIST,
			d: UMAP_DIM,
			metric: druid.cosine,
			seed: UMAP_SEED,
		});

		const result: any = umap.transform(UMAP_ITERS);

		// druid returns its own Matrix type, need to convert
		if (result && typeof result.to2dArray === 'function') {
			reduced = Array.from(result.to2dArray()).map(
				(row: any) => Array.from(row) as number[]
			);
		} else if (Array.isArray(result)) {
			reduced = result;
		} else {
			reduced = [];
			for (const row of result) {
				reduced.push(Array.from(row));
			}
		}
	}

	const umapTimeMs = performance.now() - t0;

	// --- try different k values, pick best silhouette ---
	const t1 = performance.now();

	const kMax = Math.max(2, Math.min(Math.floor(Math.sqrt(N)), N - 1));
	const kScores: { k: number; score: number }[] = [];
	let bestK = 2;
	let bestScore = -Infinity;

	for (let k = 2; k <= kMax; k++) {
		const res = kmeans(reduced, k, { initialization: 'kmeans++', seed: 42 });
		const score = silhouetteScore(reduced, res.clusters);
		kScores.push({ k, score });

		if (score > bestScore) {
			bestScore = score;
			bestK = k;
		}
	}

	// final run with best k
	const final = kmeans(reduced, bestK, { initialization: 'kmeans++', seed: 42 });
	const kmeansTimeMs = performance.now() - t1;

	return {
		assignments: final.clusters,
		k: bestK,
		silhouetteAvg: bestScore,
		kScores, umapTimeMs, kmeansTimeMs,
		reducedPoints: reduced,
	};
}
