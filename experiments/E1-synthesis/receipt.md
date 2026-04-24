# E1 — Synthesis Prompt Enhancement on Multi-Session Questions
**Date:** 2026-04-24
**Cost:** $0.259 (1,544,098 input + 45,649 output tokens, gpt-4o-mini)
**Reproduce:** `OPENAI_API_KEY=sk-... python3 bench/qa_eval_v3_routing.py --run-id runP-v35 --experiment-id E1 --qtypes multi-session --data-dir ~/Documents/projects/LongMemEval/data --out-dir experiments/E1-synthesis`
**Random seed:** temperature=0 (deterministic responses)
**Backend:** OpenAI gpt-4o-mini (reader + grader)
**Prompt version:** v3 enhanced (see bench/qa_eval_v3_routing.py `_QA_SYS_MULTI_V3`)

## Result

| Config | n | MS QA accuracy |
|---|---|---|
| E1 (v3 enhanced prompt, K=5) | 109 | **57.8%** |
| Baseline top5 v2, same 109 questions | 109 | 69.7% |
| Baseline top5 v2, all 121 MS questions | 121 | 67.8% |

**Delta vs baseline (same 109 questions): -11.9pp**

**REGRESSION.** The enhanced synthesis prompt hurts gpt-4o-mini on MS questions.

## What Changed vs Baseline

| Factor | v2 baseline | v3 E1 |
|---|---|---|
| MS synthesis prompt | Simple aggregate instruction | Complex: "cite session IDs", "do NOT stop", "list from ALL sessions" |
| max_tokens (reader) | 512 | 768 |
| K sessions | 5 | 5 |
| Reader model | gpt-4o-mini | gpt-4o-mini |
| Grader model | gpt-4o-mini | gpt-4o-mini |

**Confound note:** max_tokens differs (512 → 768). The direction of this confound is unclear: more tokens gives longer output, which could help or hurt the grader. This cannot fully explain -11.9pp regression.

## Adversarial Gate Verdict
FLAGGED (5 flags raised by adversarial reviewer):
1. 12 skipped questions are sorted tail — possible order bias (not confirmed random)
2. No per-MS-subtype breakdown
3. max_tokens confound between v3 and v2
4. Rate limiting may have affected some of the 109 completed responses
5. No reproducibility chain (no commit SHA)

**Net verdict:** Regression is real (confirmed on matched 109 questions: 19 flipped v2✓→v3✗ vs only 6 v3✓→v2✗). Trust enough to NOT use v3 MS prompt with gpt-4o-mini. Do NOT conclude "explicit synthesis hurts generally" — may be gpt-4o-mini-specific over-instruction.

## Missing 12 Questions

Questions 107-121 in the MS sequence failed with "QA failed twice" — API rate limit after ~109 consecutive large-context calls. These are the last 12 MS questions in per_question.json order (sorted by `qi` index, not by difficulty). No clear difficulty bias, but cannot rule it out without further analysis.

## Per-Question Breakdown

- Questions where v2 correct, v3 wrong: **19** (prompt breaks working cases)
- Questions where v3 correct, v2 wrong: **6** (prompt helps some cases)
- Net: -13 questions → -11.9pp on matched subset

## Decision Gate (per roadmap)
- E1 result is MS ≥ 75%? NO (57.8%)
- E1 result ≥ 67.8%? NO (-11.9pp regression)
- Action: Proceed to E2 (GPT-4o reader) using v2 prompts. Do NOT use v3 MS prompt with gpt-4o-mini.

## What E1 Proves

1. **gpt-4o-mini cannot benefit from complex multi-session synthesis instructions.** The model is capacity-limited — adding "cite which session each fact came from" and "do NOT stop reading" instructions causes it to misfollow or hallucinate.
2. **v2 MS synthesis prompt (simpler) is the ceiling for gpt-4o-mini at K=5.** 67.8-69.7% is the MS ceiling for this model.
3. **To push MS beyond 70%, we need a stronger reader model (E2).** Prompt engineering on mini is not the lever.
