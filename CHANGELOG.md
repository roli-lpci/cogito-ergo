# Changelog

## v0.3.0 — 2026-04-16

- New `recall_hybrid` path: BM25 + dense + RRF with tiered LLM escalation.
  Port of the architecture that reached **93.4% R@1 on LongMemEval_S** (up
  from the 56% mem0 baseline).
- New `POST /recall_hybrid` HTTP endpoint and `cogito recall-hybrid` CLI.
- New tiers: `zero_llm` (no LLM, fastest), `filter` (cheap rerank, default),
  `flagship` (stronger model, 4x larger snippets).
- New env vars: `COGITO_FLAGSHIP_ENDPOINT`, `COGITO_FLAGSHIP_TOKEN`,
  `COGITO_FLAGSHIP_MODEL`, `COGITO_FLAGSHIP_TIMEOUT_MS`,
  `COGITO_HYBRID_COSINE_WEIGHT`. Optional `[hybrid]` extra for `bm25s`.
- Existing `/recall` and `/recall_b` behavior is unchanged. The hybrid path
  is strictly opt-in.

## v0.2.0 — 2026-03-28

- Dual-pipeline recall: zero-LLM `recall_b` (RRF multi-query) feeds `recall` (integer-pointer LLM filter)
- Snapshot layer for high-fidelity context injection
- Combined eval harness (`bench/`)
- Technical extraction prompt baked into default config (no external prompt file needed)
- Qwen3/qwen3.5 support via native Ollama `/api/chat` with `think:false`

## v0.0.8 — 2026-03-28

- Fixed benchmark attribution: qwen3.5:2b filter model (not claude-haiku-4-5)
- Added Hermes Labs PyPI metadata (author, homepage, keywords)
- Added agents.md for AI agent discoverability
- Updated llms.txt with full API shapes and integration notes
- Added "Built by Hermes Labs" ecosystem section to README

## v0.1.0 — 2026-03

- Initial two-stage integer-pointer recall pipeline
- mem0 + ChromaDB vector store backend
- HTTP server with `/recall`, `/add`, `/snapshot` endpoints
- `cogito calibrate` for vocab map generation
