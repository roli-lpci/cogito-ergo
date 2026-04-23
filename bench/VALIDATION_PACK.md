# VALIDATION_PACK — cogito-ergo LongMemEval (April 2026)

## Summary

| Metric | v33 (hardset) | runO (runtime, no hardset) |
|--------|--------------|---------------------------|
| Retrieval R@1 | 95.11% | 92.34% |
| Integrity | FAILED (test-set memorization) | PASS (no gold leakage) |
| Escalation trigger | 37 hardcoded qids (from test inspection) | Runtime score rule (no gold access) |
| Escalation count | varies (hardset size = 37) | 114/470 = 24.3% |

**QA accuracy (Task B): 54.16% on 469/470 questions** (qwen-max grader, NOT comparable to Mastra's 94.87% — different model)

---

## Task A: Runtime Escalation Pipeline (runO)

### Config
- Pipeline: `bench/longmemeval_combined_pipeline_v34.py`
- Run ID: `runO-runtime-escalate`
- Split: S (470 questions)
- Escalation rule: `top1 < 0.65 OR gap < 0.03` (gap = scores[0] - scores[1])
- No hardset, no test-set inspection at any point

### Final Results

**Overall R@1: 92.34% (434/470)**

| qtype | n | R@1 | v33 R@1 | delta |
|-------|---|-----|---------|-------|
| knowledge-update | 72 | 95.83% | 97.20% | -1.37% |
| multi-session | 121 | 95.87% | 98.30% | -2.43% |
| single-session-assistant | 56 | 100.00% | 100.00% | 0.00% |
| single-session-preference | 30 | 66.67% | 86.70% | -20.03% |
| single-session-user | 64 | 98.44% | 100.00% | -1.56% |
| temporal-reasoning | 127 | 86.61% | 88.20% | -1.59% |

### Escalation Breakdown

- Total escalations: 114/470 = 24.3%
  - `small_gap` (scores[0]-scores[1] < 0.03): 112
  - `low_top1` (scores[0] < 0.65): 2
- Target was ~10%; actual is 24.3% — calibration sample (65 SSU/multi-session questions) had different score distribution than preference questions

### Integrity Analysis

**v33 used a hardset of 37 questions identified by inspecting test results.** This is test-set memorization. Those 37 questions were routed to qwen-max flagship regardless of retrieval scores.

Hardset composition:
- single-session-preference: 10 of 30 (33%)
- temporal-reasoning: 15 of 127 (12%)
- knowledge-update: 3 of 72 (4%)
- multi-session: 6 of 121 (5%)
- other: 3

Impact on runO (questions hardset covered that runO missed):
- SSP: 8 misses → explains -20% on preference category
- temporal-reasoning: 14 misses → explains -1.6% on TR
- knowledge-update: 2 misses
- multi-session: 5 misses

**Honest R@1 without test memorization: 92.34%** vs inflated 95.11%.
Integrity tax from removing hardset: approximately -2.77 percentage points overall.

---

## Task B: QA Accuracy (qa_eval.py on runJ-v33)

### Config
- Input: `bench/runs/runJ-v33/per_question.json` (v33 retrieval results)
- QA model: qwen-max via dashscope-intl
- Multi-session questions: top-5 retrieved sessions passed as context
- Single-session questions: top-1 session only
- 469/470 questions (b6019101 failed both API retries, skipped)

### Final Results

**Overall QA accuracy: 54.16% (254/469)**

| qtype | n | QA accuracy |
|-------|---|------------|
| single-session-user | 64 | 96.88% |
| single-session-assistant | 56 | 98.21% |
| single-session-preference | 30 | 13.33% |
| knowledge-update | 71 | 64.79% |
| multi-session | 121 | 43.80% |
| temporal-reasoning | 127 | 26.77% |

### Conditional Accuracy

| Condition | n | QA accuracy |
|-----------|---|------------|
| retrieval_hit=True | 446 | 56.73% |
| retrieval_hit=False | 23 | 4.35% |

Retrieval R@1 from v33 run: 95.10%

### Comparison to Mastra

**This number is NOT directly comparable to Mastra's 94.87%.**

Mastra used gpt-4o-mini as the QA grader. We used qwen-max. Key differences:
1. **Model capability gap**: gpt-4o-mini and qwen-max differ in instruction-following for extraction tasks
2. **Preference questions at 13.33%**: qwen-max appears to over-refusal on personal preference questions (e.g., "what kind of music does the user prefer?") — returns "I cannot determine..." even when the answer is in context
3. **Temporal-reasoning at 26.77%**: Multi-hop temporal queries require chaining reasoning across sessions; single-pass QA with top-5 sessions is not sufficient
4. **No OPENAI_API_KEY available**: Cannot run directly comparable gpt-4o-mini eval without separate API access

**To get a directly comparable number**: re-run `bench/qa_eval.py` with `OPENAI_API_KEY` set (it auto-selects gpt-4o-mini when key is present).

### Why QA accuracy is low

1. **Model mismatch**: qwen-max is not gpt-4o-mini. The benchmark was designed around OpenAI models.
2. **Preference extraction**: qwen-max refuses or hedges on personal preference questions even with clear context.
3. **Multi-session aggregation**: Questions like "how many times did the user mention X across sessions?" require careful counting — model often gives partial or hedged answers.
4. **Temporal chains**: Date arithmetic + event sequencing across 20+ sessions hits instruction-following limits.

The SSU/SSA accuracy (96-98%) confirms retrieval is working correctly. The failure modes are model-specific, not retrieval-specific.

---

## Files

| File | Description |
|------|-------------|
| `bench/longmemeval_combined_pipeline_v34.py` | Task A pipeline (runtime escalation, no hardset) |
| `bench/runs/runO-runtime-escalate/per_question.json` | Task A retrieval results (470 questions) |
| `bench/qa_eval.py` | Task B QA grader |
| `bench/qa_eval_runJ-v33.json` | Task B QA results (469 questions) |
| `bench/qa_eval_runJ-v33_summary.json` | Task B summary stats |

---

## Key Findings

1. **Production-honest R@1 = 92.34%** (no test-set memorization). Previous 95.11% was inflated by hardset.
2. **Preference category is the main casualty** of removing hardset: 66.67% vs 86.70%. 8 of 10 hardset-preference questions are now misses.
3. **Runtime rule fires at 24.3%** (target was ~10%). The calibration was done on retrieval-friendly question types. Calibrate on full distribution or tighten thresholds.
4. **QA accuracy comparison to Mastra is blocked** without OpenAI API key. Qwen-max grader at 54.16% is not the same measurement.
5. **Given correct retrieval, QA accuracy = 56.73%** — half of correctly-retrieved questions are still wrong. This is a grader model problem, not a retrieval problem.
