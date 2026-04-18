# cogito-ergo — Status as of 2026-04-17

One-page crystallization. If you only read one file, read this.

## Bottom line

cogito-ergo has **two retrieval paths** living in the same server. One is the existing production path (atomic-fact recall). The other is a new session-retrieval pipeline that wins LongMemEval_S at 93.4% (projected 95-96% with verify-guard, still running). On cogito's OWN internal eval, the new path regresses 21 points vs the existing path. The benchmark win and the production workload are not the same task.

## Numbers

| System | Benchmark | R@1 | Notes |
|---|---|---|---|
| `/recall` (atomic, existing) | 31-case eval (cogito's own) | 75% | stochastic 60-90% per seed (LLM filter is non-deterministic) |
| `/recall` (atomic, existing) | LongMemEval_S | never run | workload mismatch — cogito doesn't natively handle multi-turn sessions |
| `/recall_hybrid` (session, new) | LongMemEval_S | 93.4% final, 95-96% projected with verify-guard (in-flight) | beats Mastra 94.87% if it lands |
| `/recall_hybrid` (session, new) | 31-case eval (cogito's own) | 54% | 21pt regression vs `/recall` — hit@any +11pt, latency halved |

Cost: ~$0.007/query avg, $2.33 total flagship spend on 37/470 hardset questions. ~1/10-1/50 cost of GPT-5-mini-backed leaderboard systems.

## The two architectures (why they don't interchange)

**Path A — `/recall` (atomic):**
- Stores ~50-200 char facts
- Dense (nomic) retrieval + optional LLM filter reranker
- Tuned for paraphrase queries over short facts
- Code: `src/cogito/recall.py`

**Path B — `/recall_hybrid` (session):**
- Stores 2000+ char multi-turn sessions
- BM25 + nomic dense + RRF fusion
- Turn-level chunking
- Nomic prefixes (`search_query:` / `search_document:`)
- Regex router classifies query type → llm / skip / default
- Tiered LLM: qwen-turbo cheap filter on routed cases, qwen-max flagship on hardset 8%
- Code: `src/cogito/recall_hybrid.py` (new, v0.3.0, uncommitted)
- Benchmark script: `bench/longmemeval_combined_pipeline_flagship.py`

**Why Path B regressed on cogito's eval (architect's diagnosis):** it's path-structural, not routing failure. BM25 adds noise on 50-200 char facts; the filter's 150-char snippet truncation mangles session content. Two separate bugs, fixable by separating the storage indexes (not by a smarter query router).

## What's been tried (don't redo)

| Attempt | Outcome |
|---|---|
| Combined retrieval (BM25 + dense + RRF + chunks + prefixes, zero LLM) | 83.2% R@1 on LongMemEval |
| Regex router + qwen-turbo filter | 88.9% |
| Learned router (LR + RF, 774 features) | 83% CV — worse than hand-regex (13% positive labels, ML can't learn the boundary). **Publishable negative result.** |
| Chunk-level date injection | 0pp net. Noise floor. |
| LPCI temporal scaffold in filter prompt | -1.2pp. Reverted. Scaffold format was wrong for this prompt position. Thesis not disproved. |
| Assistant-text in snippets | Regressed (500-char budget too tight). Reverted. |
| qwen-max flagship on hardset | +4.5pp → **93.4% final** |
| Verify-guard (restore S1 #1 when filter demotes it) | In flight. Projected 95-96%. Specialist agent on ScheduleWakeup. |

## New findings worth naming

1. **The demotion problem** (novel). LLM filter actively demotes gold that retrieval already ranked #1. 15/470 questions hit by this. Generalizes to any RAG + LLM-reranker system. Verify-guard fixes it.
2. **Latest-date temporal trap** (novel). 7 hardset questions have gold dated "today" — haystack_dates metadata can't help. Needs event-date extraction from text.
3. **Two-speed recall**. S1 (retrieval) and S2 (LLM filter) solve different populations. Current architecture doesn't exploit the split.
4. **Workload divergence**. Architecture that wins LongMemEval (session-oriented) regresses on atomic-memory paraphrase recall. Single-workload tuning doesn't transfer.

## Docs to read in order

1. This file
2. `docs/DISPATCHER_DESIGN.md` — Sonnet architect's plan for per-type indexes + generalizability guardrails + CI gates
3. `docs/GENERALIZABILITY_SKEPTIC.md` — Haiku skeptic's five falsifier tests + hole list

## Artifacts

- `bench/runs/` — per-question JSONs for every benchmark run
- `bench/phase-2/hardset.json` — the 37 hardest questions (frozen regression set for the 93.4% run)
- `bench/results-runB-flagship-*.json` — the 93.4% final numbers
- `bench/results-guard-*.json` — in-flight verify-guard run
- `CHANGELOG.md` — v0.3.0 entry (uncommitted)

## The fork (pick one)

**Fast track (~2 weeks):** paper on LongMemEval 93.4-96% + cost advantage + demotion-problem finding. Narrow scoped claim ("session-retrieval, LongMemEval workload"). Honest. Ships quickly. Doesn't overclaim generalizability.

**Big track (~4-8 weeks):** build the per-type index dispatcher (architect's design), run the 5 falsifier tests (skeptic's ask), add LOCOMO + MemoryBench to the suite, measure cogito's actual memory length distribution, publish generalizability scorecard. Claim "workload-aware memory retrieval" with evidence. Paper becomes canonical.

## What's flagged for follow-up (auto-memory)

- Documentation owed once specialist lands final numbers: handbook entry on filter-demotion, research-corpus experiment record, applied case study, memory/registry updates. See `~/.claude/projects/-Users-rbr-lpci/memory/project_cogito_v2_docs_owed.md`.

## Uncommitted changes in repo

v0.3.0 integration (recall_hybrid.py, 11 new tests, README, CHANGELOG, pyproject, etc.) is staged but **not committed**. Tests pass 42/42. Default `/recall` behavior untouched. Review before `git commit`.
