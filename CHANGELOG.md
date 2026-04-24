# Changelog

## v0.3.1 — 2026-04-24

**Headline: zero-LLM is now the default retrieval tier.** The repositioning
reflects what the benchmark numbers actually show — the zero-LLM path is
the production moat (83.2% R@1 at $0, fully local); the LLM tiers are
benchmark-tuned and held as experimental until calibration is fixed.

### Default change (breaking for callers relying on default tier)

- `recall_hybrid()` default tier flipped from `filter` to `zero_llm`.
  Callers that relied on the filter default must pass `tier="filter"`
  explicitly. Affects: `cogito recall-hybrid` CLI, `POST /recall_hybrid`,
  `recall_hybrid(...)` Python API.

### Features

- New `since` parameter on `/recall` and `cogito recall --since` —
  ISO-8601 timestamp filter applied after Stage 2.

### Observability & reliability

- New `cogito.telemetry` module: escalation-rate log + `rate()` summariser.
  Makes the 80%-vs-10% calibration miss measurable at runtime. Crash-safe.
- `degrade.replay_queue` corrupt-file branch now covered by tests.

### Tests

- `tests/test_zero_llm_regression.py` — 4 tests pin zero-LLM default:
  no filter/flagship invocation, works without LLM config, matches
  explicit-tier output, handles empty corpus.
- `tests/test_telemetry.py` — 5 tests for the new observability module.
- `tests/test_verify_guard.py` — regression pin for the documented
  verify-guard-inactive bug; xfail until fixed.
- `tests/test_graceful_degrade_corruption.py` — replay_queue corruption.
- Test baseline: 61 → 72 passing + 2 defensive skips.

### Benchmarks (for reference, not default-path claims)

- `recall_hybrid` at `tier="flagship"` hits 96.4% R@1 on LongMemEval_S
  (470 questions, runP-v35, 2026-04-18). Escalates on ~80% of queries
  under current calibration vs 10% intended — known limitation
  documented in STATUS.md and `docs/RELEASE-SCOPE.md`. Do not use the
  96.4% number as a cost-model baseline.
- Zero-LLM per-category R@1 (now the default): single-session-assistant
  100%, knowledge-update 96%, single-session-user 95%, multi-session 83%,
  single-session-preference 67%, temporal-reasoning 66%.

### Docs

- README repositioned: zero-LLM leads, LLM tiers labeled experimental,
  per-category table published.
- `docs/THRESHOLD-AUDIT.md` — pins prod vs bench escalation constants.
- `docs/RELEASE-SCOPE.md` — in-scope / held-for-later with unlock criteria.

## v0.3.0 — 2026-04-16

- New `recall_hybrid` path: BM25 + dense + RRF with tiered LLM escalation.
  Port of the architecture that reached **93.4% R@1 on LongMemEval_S** (up
  from the 56% mem0 baseline). Superseded by v0.3.1 (96.4%).
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
