# Testing Transformers.js in Joplin

A quick Joplin plugin I built to see how well Transformers.js runs inside the plugin sandbox. The idea was to benchmark embedding models on a corpus of notes.

## What it does

- Loads an embedding model (MiniLM-L6 or BGE-small) using Transformers.js
- Reads a `corpus.jsonl` file with test notes (I ran a python script to get all the notes from https://en.wikipedia.org/w/api.php)
- Embeds each note one by one in a Web Worker so Joplin doesn't freeze
- At the end, displays load time, warmup, average per-note time, and total time

## How to build

```bash
npm install
npm run dist
```

This creates a `.jpl` file in `publish/` that you can install in Joplin through Tools → Options → Plugins.

## The corpus

`src/corpus.jsonl`: one JSON object per line. Each line can have:
- `{ "text": "..." }`  used directly
- `{ "title": "...", "body": "..." }`  combined as title + body (like Joplin notes)

I fetched ~1500 Wikipedia articles using the MediaWiki API(https://en.wikipedia.org/w/api.php) for testing.

