#!/usr/bin/env python3
"""
Build src/corpus.jsonl from English Wikipedia (API). Human-written text only.

Requires: Python 3.8+ (stdlib only).

Usage:
  python tools/build_corpus_wikipedia.py
  python tools/build_corpus_wikipedia.py 100

Default count is 1000. First run downloads article extracts; respect rate limits (small delay).
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

API = "https://en.wikipedia.org/w/api.php"
OUT_PATH = "src/corpus.jsonl"
DEFAULT_COUNT = 1000
DELAY_SEC = 0.1

UA = "JoplinEmbeddingBenchmark/1.0 (local benchmark; contact: your-email@example.com)"


def api_get(params: dict) -> dict:
	q = urllib.parse.urlencode(params)
	url = f"{API}?{q}"
	req = urllib.request.Request(url, headers={"User-Agent": UA})
	with urllib.request.urlopen(req, timeout=90) as r:
		return json.loads(r.read().decode("utf-8"))


def fetch_random_titles(limit: int) -> list[str]:
	data = api_get({
		"action": "query",
		"format": "json",
		"list": "random",
		"rnnamespace": 0,
		"rnlimit": min(limit, 500),
	})
	return [p["title"] for p in data["query"]["random"]]


def fetch_extracts(titles: list[str]) -> list[tuple[str, str]]:
	if not titles:
		return []
	data = api_get({
		"action": "query",
		"format": "json",
		"prop": "extracts",
		"explaintext": 1,
		"titles": "|".join(titles),
	})
	out: list[tuple[str, str]] = []
	for page in data.get("query", {}).get("pages", {}).values():
		if page.get("missing"):
			continue
		title = page.get("title", "")
		ex = page.get("extract", "").strip()
		if title and ex:
			out.append((title, ex))
	return out


def main() -> None:
	target = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_COUNT
	if target < 1:
		print("Count must be >= 1", file=sys.stderr)
		sys.exit(1)

	written = 0
	batch_size = 10
	with open(OUT_PATH, "w", encoding="utf-8") as f:
		while written < target:
			titles = fetch_random_titles(batch_size)
			time.sleep(DELAY_SEC)
			pairs = fetch_extracts(titles)
			time.sleep(DELAY_SEC)
			for title, body in pairs:
				if written >= target:
					break
				line = json.dumps({"title": title, "body": body}, ensure_ascii=False)
				f.write(line + "\n")
				written += 1
			if not pairs:
				continue
			print(f"\r{written}/{target}", end="", flush=True)
	print(f"\nWrote {written} lines to {OUT_PATH}")


if __name__ == "__main__":
	main()
