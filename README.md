# Testing Transformers.js in Joplin

A quick Joplin plugin I built to see how well Transformers.js runs inside the plugin sandbox. Also tested UMAP and K-Means clustering in the same environment to see if the full pipeline works.

## What it does

- Loads an embedding model (MiniLM-L6 or BGE-small) using Transformers.js
- Reads a `corpus.jsonl` file with test notes (I ran a python script to get all the notes from https://en.wikipedia.org/w/api.php)
- Embeds each note one by one in a Web Worker so Joplin doesn't freeze
- After embedding, runs UMAP + K-Means to cluster the notes and picks the best k using silhouette scores
- Displays everything in a panel: load time, warmup, per-note avg, and the cluster assignments

## How to build

```bash
npm install
npm run dist
```

This creates a `.jpl` file in `publish/` that you can install in Joplin through Tools → Options → Plugins.

## The corpus

`src/corpus.jsonl`: one JSON object per line. Each line can have:
- `{ "title": "...", "body": "..." }`  combined as title + body (like Joplin notes)


I fetched ~1500 Wikipedia articles using the MediaWiki API(https://en.wikipedia.org/w/api.php) for testing.

## The result
| Description | Screenshot |
|-------------|-----------|
| BGE-small 125 notes | <img width="400" alt="Screenshot 2026-03-28 055901" src="https://github.com/user-attachments/assets/62ec6c24-85bb-483e-a21c-aa51c7cba4cf" /> |
| BGE-small 250 notes | <img width="400" alt="Screenshot 2026-03-28 055313" src="https://github.com/user-attachments/assets/365a38ca-b8aa-4f32-8a1d-269bec537d61" /> |
| BGE-small 500 notes | <img width="400" alt="Screenshot 2026-03-28 054323" src="https://github.com/user-attachments/assets/aa644f74-5e31-4034-9836-548ef2fa49cd" /> |
| BGE-small 1000 notes | <img width="400" alt="Screenshot 2026-03-28 053555" src="https://github.com/user-attachments/assets/be256008-b856-406e-9685-5c2196a8bcc6" /> |
| BGE-small 1500 notes | <img width="400" alt="Screenshot 2026-03-28 062917" src="https://github.com/user-attachments/assets/c8eadeb4-e174-4bcd-bbe1-4b1a9dfd6df8" /> |
| MiniLM-L6 150 notes | <img width="400" alt="Screenshot 2026-03-28 080610" src="https://github.com/user-attachments/assets/6f18ba5a-8d23-4c83-82c8-36154d3b95f3" /> |
| MiniLM-L6 500 notes | <img width="400" alt="Screenshot 2026-03-28 084941" src="https://github.com/user-attachments/assets/af345284-65e0-4856-ac96-adf6c2b49444" /> |
| MiniLM-L6 1000 notes | <img width="400" alt="Screenshot 2026-03-28 083744" src="https://github.com/user-attachments/assets/d93b0622-8d08-4057-ad1d-d06e60d2fba6" /> |
| MiniLM-L6 1500 notes | <img width="400" alt="Screenshot 2026-03-28 082612" src="https://github.com/user-attachments/assets/f22fe166-f816-4e14-9980-3c55ab4e1376" /> |
| MiniLM-L6 3000 notes | <img width="400" alt="Screenshot 2026-03-28 091421" src="https://github.com/user-attachments/assets/76ec13a7-f53f-4f37-84d2-66d73f97966d" /> |

Screen recording:

MiniLM-L6
https://github.com/user-attachments/assets/c69c4e63-1ccf-411a-ac92-88770d3267e2

BGE-small
https://github.com/user-attachments/assets/c8db40b8-022e-499d-98cb-398aed663e3e


## Clustering

After all notes are embedded, the plugin runs:
1. UMAP (via [DruidJS](https://github.com/saehm/DruidJS)) to reduce the 384-dim vectors down to 5 dimensions
2. K-Means (via [ml-kmeans](https://github.com/mljs/kmeans)) for k=2 to √N, picks whichever k has the best silhouette score

On 25 notes it takes ~60ms for UMAP and ~10ms for K-Means. The whole clustering part is basically free compared to embedding time.


