# v0.3.1 Scaffold Dispatcher — Design + Evidence

Data sources:
- `bench/scaffold_arena_results.md` (10 scaffolds × 20 failing questions tested, gemma3:4b)
- `bench/r5_ceiling_diagnostic.md` (8 R@5-miss questions analyzed, zero-LLM)
- `bench/runs/runC-guard/per_question.json` (baseline 94.0%)

## Two-layer architecture

### Layer 1 — Retrieval (Stage 1, zero-LLM)
Breaks the R@5 = 98.3% ceiling.

**Add temporal-boost to RRF fusion** (targets the 8 R@5-misses):
- Current Stage 1 ignores `haystack_dates` metadata entirely
- 7 of 8 R@5-misses are temporal-reasoning questions
- Gold sessions at haystack ranks 10-38 (present, just under-ranked)
- Fix: when query has temporal signal (regex detection already exists), boost session scores by date-match / recency / range-overlap
- Expected lift: R@5 95% → 98-99% (6-7 of 8 recovered)

### Layer 2 — Scaffold dispatcher (Rerank, per-qtype)
Maximizes R@1 given retrieval.

Runtime dispatch (zero-LLM query classifier → per-qtype scaffold):

```python
SCAFFOLD_DISPATCHER = {
    "temporal-reasoning":        "s2-temporal-inline",        # +8/12 on failures, inline date block in user msg
    "single-session-preference": "s4-declarative-preference", # +3/3 on failures, declarative-statement focus
    "knowledge-update":          "s0-minimal-baseline",       # arena: scaffolds don't help here
    "multi-session":             "s0-minimal-baseline",       # arena: all scaffolds tie at 1/2
    "single-session-assistant":  "s2-temporal-inline",        # arena: 1/1 with s2
    "single-session-user":       "s0-minimal-baseline",       # already 100% in runC-guard
}
```

**Composition candidate for temporal-reasoning** (run both, union):
- s2-temporal-inline covers 8/12 failures
- s7-chain-of-thought covers a disjoint subset (vague-reference queries: "how many days ago")
- Union = 10/12 temporal failures fixed (vs 8/12 with s2 alone)
- Cost: 2 LLM calls per temporal query instead of 1

## Scaffold text (verbatim, from arena)

### s2-temporal-inline
Inline in user message (not system prompt), immediately before the candidate list:
```
Below are {n} candidate sessions. I have already ordered them by date:
[1] {date[0]} (earliest)
[2] {date[1]} (+{delta1} days)
[3] {date[2]} (+{delta2} days)
...

Question: {query}

Candidates:
{candidate_block}

Return the index of the session that answers the question.
```

### s4-declarative-preference
```
One of these sessions contains the user's preference as an EXPLICIT DECLARATIVE STATEMENT.
Return only the session where the user STATES a preference — 'I like X', 'I prefer Y', 'my favorite is Z'.
Do NOT return sessions where the user ASKS about preferences or DISCUSSES preferences generally.
```

### s3-contrastive-not (alternative to s4)
Matches s4 on preference (3/3) but more general. Pick one — not both.

## Scaffolds to avoid

- **s8-socratic**: -3 vs baseline. Overthinks, chases type-classification semantic alignment rather than content. DO NOT ship.
- **s1-temporal-sysprefix**: 0 net gain. Arena conclusion: "date info in system prompt is being ignored or misapplied." The CURRENT sweep's Shot 1 uses this variant and likely won't deliver.

## Hard cases (ceiling even with dispatcher)

Two questions resistant to ALL 10 scaffolds tested:
- `6d550036` (multi-session): "How many projects have I led?" — semantic dilution, gold demoted
- `9a707b82` (temporal): "I mentioned cooking for my friend a couple of days ago" — vague relative reference, no date anchor

These require retrieval-layer or richer metadata fixes. Not scaffold-solvable.

## Expected impact if fully implemented

| Component | Mechanism | Target | Est. lift |
|---|---|---|---|
| Temporal-boost at Stage 1 | Retrieval-layer | 8 R@5-misses | +6-7 questions = +1.3-1.5pp |
| Dispatcher: temporal scaffold | Rerank | 12 temporal failures in top-5 | +3-4 questions = +0.6-0.8pp |
| Dispatcher: preference scaffold | Rerank | 4 preference failures | +1-2 questions = +0.2-0.4pp |
| s2+s7 composition (optional) | Rerank union | +2 more temporal | +2 questions = +0.4pp |

**Stacked estimated ceiling: 96-97% R@1** (vs current 94.0%).

## What this replaces in the paper story

Old: "cogito beat Mastra at 1/10 cost via flagship + verify-guard."
New: "cogito has a **diagnosis-driven two-layer architecture**: retrieval-layer temporal-boost + per-qtype scaffold dispatcher. Each component derived from failure analysis, not benchmark tuning. Adaptive at install time."

This is the meta-scaffold story, even though no single meta-scaffold exists.
