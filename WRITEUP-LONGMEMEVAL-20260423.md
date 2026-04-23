# cogito-ergo on LongMemEval — Benchmark Writeup

**Date:** 2026-04-23  
**Status:** Reproducibility artifact. Not a leaderboard submission.  
**Author:** Hermes Labs (roli-lpci)

---

## What cogito-ergo is

cogito-ergo is an open-source agent memory system with two retrieval paths:

- **Path A (`/recall`)** — atomic-fact recall. Stores 50–200 char facts. Tuned for paraphrase queries. The production path.
- **Path B (`/recall_hybrid`)** — session-retrieval. Stores multi-turn sessions (2000+ chars). BM25 + nomic dense + RRF fusion, turn-level chunking, LLM filter tier. Tuned for LongMemEval.

This writeup covers Path B's performance on LongMemEval_S. Path A is not evaluated here — it handles a different workload.

---

## Baseline and full result history

| Version | R@1 | Notes | LLM? |
|---|---|---|---|
| Original baseline | 56.0% | Dense-only retrieval | No |
| + nomic prefixes | 60.9% | `search_query:` / `search_document:` | No |
| + BM25 hybrid | 73.2% | BM25 + dense + RRF | No |
| + Turn-level chunks | 66.8% | Session split at turn boundaries | No |
| Combined (all three) | 83.2% | Stage 1 ceiling, zero-LLM | No |
| + Regex router + qwen-turbo | 88.9% | Classify query → LLM filter | Selective |
| + Hardset flagship (runB, v33) | 93.4% | 37-question hardset; see integrity note | Selective |
| + Runtime escalation (runO, v34) | 92.3% | No hardset; honest baseline | Selective |
| + Temporal boost + runtime (runP, v35) | **96.4%** | Temporal score multiplier; no hardset | Selective |

**The number we report: R@1 = 96.4% (453/470) on LongMemEval_S, run ID runP-v35, 2026-04-18.**

Exact command to reproduce:

```bash
cd ~/Documents/projects/cogito-ergo
python3 bench/longmemeval_combined_pipeline_v35.py \
    --data-dir ~/Documents/projects/LongMemEval/data \
    --run-id runP-v35-repro \
    --filter-model qwen-turbo   # requires DASHSCOPE_API_KEY
```

For zero-LLM stage only (83.2% R@1, no API key needed):

```bash
# Set --no-llm-filter flag in the pipeline script (or use bench/longmemeval_hybrid.py directly)
```

---

## Why this is NOT a leaderboard submission

**The gap is not the issue. The metric is.**

Mastra's published 94.87% is **QA accuracy (Task B)** — measured with gpt-4o-mini as both reader and judge, using evaluate_qa.py from the LongMemEval repo.

cogito-ergo's 96.4% is **retrieval R@1 (Task A)** — measured with recall_any@1, checking whether the gold session appears in position 1 of the ranked list.

These are different tasks. Retrieval R@1 is an upper bound on QA accuracy — a system with 96.4% retrieval R@1 can achieve at most 96.4% QA accuracy (assuming perfect answer generation). Our measured QA accuracy with qwen-max is 54.2% (runJ-v33), but qwen-max is not comparable to gpt-4o-mini as a reader.

To make a legitimate leaderboard comparison we'd need to run `evaluate_qa.py` on our runP-v35 retrieval output using gpt-4o-mini. Estimated cost: ~$1.24. Blocked on OpenAI API key.

**The path chosen is writeup**, not leaderboard. Honest about what we measured.

---

## Per-category R@1 (runP-v35, 470 questions)

| Category | n | R@1 | Notes |
|---|---|---|---|
| single-session-user | 64 | 100.0% | Perfect |
| multi-session | 121 | 99.2% | 1 miss: 4-gold question |
| knowledge-update | 72 | 98.6% | 1 miss: 2-gold routing gap |
| single-session-assistant | 56 | 98.2% | 1 miss |
| temporal-reasoning | 127 | 92.1% | 10 misses — hardest category |
| single-session-preference | 30 | 86.7% | 4 misses — semantic gap |

**17 total misses at R@1.** Temporal-reasoning accounts for 59% of them.

---

## Three novel findings worth naming

### 1. The demotion problem

LLM reranker filters can demote gold sessions that retrieval already ranked #1. In the guard run analysis (runC), 15+ questions had gold at S1 position 1 but were demoted to position 2+ by the LLM filter. Standard RAG + LLM-reranker architectures have this failure mode; it is underreported. The verify-guard experiment attempted to fix it (restore S1 #1 if filter demotes it) but the guard activation code did not fire in runC — a bug, not a negative result.

### 2. Workload divergence kills transfer

Path B (session-retrieval, 96.4% on LongMemEval) scores **54% on cogito's own 31-case atomic-fact eval**, vs 75% for Path A. BM25 adds noise on 50-200 char facts. The filter's 150-char snippet truncation mangles sessions. Single-workload tuning does not transfer. This is an argument for per-workload indexes rather than one unified retrieval architecture — and likely applies to any RAG system mixing short facts with long sessions.

### 3. The escalation rate problem

The intended escalation rate was ~10%. runP-v35 fired on 377/470 = 80% of questions. The calibration sample (65 SSU/multi-session questions) had a different confidence score distribution than preference questions, so the `top1<0.8 or gap<0.07` threshold overfit. In production this would be prohibitively expensive. A proper calibration requires stratified sampling across all 6 qtypes.

---

## What we'd change architecturally if pursuing leaderboard seriously

1. **Per-workload indexes, not a unified index.** Short facts (atomic memory) and long sessions (conversation history) should not share the same BM25 + dense index. The SESSION index uses turn-level chunks (~300 chars); the FACT index uses full entries. Separate retrieval pipelines, unified routing layer.

2. **Temporal event extraction at ingest, not query time.** 10/17 remaining misses are temporal-reasoning questions. The temporal boost (multiplies similarity scores for sessions whose text matches date patterns) helps (+7.1pp on TR) but is brittle. Extracting explicit event dates into a metadata field at ingest would let us answer "when did X happen last?" with a direct metadata filter rather than cosine similarity tricks.

3. **QA reader model matters as much as retrieval.** Our retrieval R@1 is 96.4%; our QA accuracy is 54.2% (qwen-max). The ceiling bottleneck is not retrieval — it's the reader. For leaderboard competition the right investment is a strong reader (gpt-4o-mini level) with multi-session context assembly, not more retrieval tuning. We are past retrieval diminishing returns.

---

## Three concrete next experiments

1. **Run evaluate_qa.py with gpt-4o-mini on runP-v35 retrieval output.** Cost ~$1.24. This gives a fair comparison to Mastra's 94.87%. Requires OpenAI API key. Estimated outcome: ~94–95% QA accuracy (96.4% retrieval x ~98% reader accuracy on correct sessions).

2. **Separate indexes for atomic facts vs sessions.** Hypothesis: Path A's 75% on the 31-case eval rises to 85%+ and Path B's LongMemEval score stays flat. This tests whether the regression is index pollution or query distribution mismatch.

3. **Run on LOCOMO and MemoryBench.** LongMemEval_S is one benchmark. LOCOMO (personalized conversational memory, ~600 queries) and MemoryBench (structured memory retrieval) test different distribution shifts. If cogito hits 80%+ on both with the same architecture, the retrieval approach generalizes. If not, the current result is benchmark-specific.

---

## Integrity checklist

- [x] Evaluated on `longmemeval_s_cleaned.json` (September 2025 version, HF: xiaowu0162/longmemeval-cleaned), 470 non-abstention questions
- [x] recall_any@1 definition matches LongMemEval's `eval_utils.py` (any gold session in top-1)
- [x] Gold session IDs never exposed to retrieval or filter stages (code audit 2026-04-16, bench/BENCHMARK_INTEGRITY_AUDIT.md)
- [x] Runtime-only escalation rule: `top1_score < 0.8 OR gap < 0.07`. No hardset derived from test inspection
- [x] 17 misses listed by qid in results JSON
- [ ] Mastra comparison unconfirmed: 94.87% metric type and dataset version not independently verified
- [ ] QA accuracy (Task B) not measured with gpt-4o-mini (requires OpenAI API key)
- [ ] Third independent reproduction run not complete

---

## Artifacts

- **This writeup:** `~/Documents/projects/cogito-ergo/WRITEUP-LONGMEMEVAL-20260423.md`
- **Results JSON (per-question):** `~/Documents/projects/research-corpus/agent-infra/raw/cogito-longmemeval-20260423.json`
- **runP-v35 per_question:** `bench/runs/runP-v35/per_question.json`
- **runP-v35 aggregate:** `bench/runs/runP-v35/aggregate.json`
- **Benchmark integrity audit:** `bench/BENCHMARK_INTEGRITY_AUDIT.md`
- **Full results history:** `bench/RESULTS-SUMMARY.md`

---

## Compute notes

- Full benchmark run: 752.7 seconds (~12.5 min) on macOS ARM
- LLM cost: ~$2.50 total for 470 questions (mostly qwen-turbo; ~$0.0053/query avg)
- Zero-LLM stage (83.2% R@1): $0, ~90ms/query
- Model: nomic-embed-text (local Ollama) for dense retrieval; qwen-turbo (DashScope intl) for filter
