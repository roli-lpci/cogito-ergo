# Threshold Constants Audit (2026-04-24)

Generated during the 2026-04-24 hardening audit. Pins every escalation /
confidence constant to its authoritative file:line so future doc-drift
flags can be adjudicated quickly.

## There are TWO escalation thresholds, not one

This is the single most common source of confusion for auditors reading
STATUS.md and `recall_hybrid.py` side by side.

### Production path — `src/cogito/recall_hybrid.py`

| Constant | Value | File:line | Role |
|---|---|---|---|
| `_GAP_THRESHOLD` | `0.1` | `src/cogito/recall_hybrid.py:66` | `top1 − top2` gap; above = confident, skip cheap filter |

Call site: `recall_hybrid.py:607` — `confident = (top1 - top2) > _GAP_THRESHOLD`.
Calibrated on LongMemEval_S score distributions (runB-flagship, 470q).
See comment at `recall_hybrid.py:64`.

### Benchmark path — `bench/longmemeval_combined_pipeline_v35.py`

| Constant | Value | File:line | Role |
|---|---|---|---|
| `FLAGSHIP_SCORE_THRESHOLD` | `0.80` | `bench/longmemeval_combined_pipeline_v35.py:78` | escalate to flagship if top-1 score below this |
| `FLAGSHIP_GAP_THRESHOLD` | `0.07` | `bench/longmemeval_combined_pipeline_v35.py:79` | escalate to flagship if (top1 − top2) below this |
| `GAP_THRESHOLD` | `0.1` | `bench/longmemeval_combined_pipeline_v35.py:131` | mirror of prod `_GAP_THRESHOLD` for the bench copy of the default route |

Call site: `bench/longmemeval_combined_pipeline_v35.py:913` —
`escalate, reason = should_escalate_to_flagship(...)`.
Fires ~80% of queries (per v35 notes line 89 and STATUS.md).

## Why "doc drift" is a misleading framing

An audit of the README or STATUS.md can easily flag that the prod path
uses `0.1` but the documented escalation rule cites `0.80 / 0.07`. These
are **different mechanisms in different codebases**:

- `_GAP_THRESHOLD` gates whether the default route uses the cheap filter
  at all. It has nothing to do with the bench's `filter → flagship`
  runtime escalation.
- `FLAGSHIP_SCORE_THRESHOLD` / `FLAGSHIP_GAP_THRESHOLD` decide when the
  bench pipeline escalates from cheap filter to flagship LLM.

Both are "escalation thresholds" by name; neither is the other. The
STATUS.md line "80.2% actual vs 10% intended" refers to the bench
rule firing, not the prod gap-threshold.

## Source-of-truth rules

1. For **production recall** behavior, `src/cogito/recall_hybrid.py`
   constants win. README and STATUS.md must quote them by file:line.
2. For **benchmark numbers**, `bench/longmemeval_combined_pipeline_v35.py`
   constants win, and the bench version's name should appear in any
   STATUS.md line citing a percentage.
3. If a new shared constant is introduced, add it here and cite back to
   this file from the README.

## Related known bugs (STATUS.md)

- **Escalation rate: 80.2% actual vs 10% intended** — refers to
  `FLAGSHIP_SCORE_THRESHOLD` + `FLAGSHIP_GAP_THRESHOLD` firing pattern
  in bench, not `_GAP_THRESHOLD` in prod.
- **Verify-guard inactive** — unrelated to either threshold;
  surfaces at `bench/longmemeval_combined_pipeline_v35.py:958`, guard
  implementation at `:456`.

Pin file: do not remove without replacement.
