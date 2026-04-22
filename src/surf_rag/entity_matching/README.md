# Entity matching (lexicon + alias)

Non-LLM exact phrase matching over `alias_map.json` and `entity_lexicon.parquet` from the
corpus `OUTPUT_DIR`, with:

- **Trie** longest-prefix (greedy) scan on match-normalized queries
- **Alnum word-boundary** filter so we do not match inside longer tokens (e.g. `ate` in `paternal`)
- **`df` max** and **ranking** (span length, `df`, source type)
- **Standalone** English fragment blocklist (tunable) for low-`df` noise that still snuck
  into the lexicon

**Env vars** (overridden by the Python API and dev script flags):

- `ENTITY_MATCH_MAX_DF` (default `8`)
- `ENTITY_MATCH_MAX_PER_QUERY` (default `12`)
- `ENTITY_MATCH_MIN_KEY_LEN` (default `3`)

Dev inspection (writes under `temp/`; that directory is usually gitignored):

```bash
poetry run python scripts/dev/extract_query_entities_benchmark.py \
  --output-dir data/processed \
  --benchmark data/processed/benchmark.jsonl \
  --out temp/query_entity_lexicon_hits.txt
```

Wiring this into :class:`surf_rag.strategies.graph.GraphRetriever` is intentionally deferred
until the offline report looks good.
