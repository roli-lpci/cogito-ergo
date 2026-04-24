# E2 Adversarial Review

**Reviewed by:** fresh-context subagent (Claude Opus)
**Date:** 2026-04-24
**E2 result:** 75.5% (355/470)

---

## Flag 1 — PROMPT-MODEL CONFOUND ON TR — HIGH / LIVE

TR gained +26.8pp with BOTH a new prompt (v3_routing.py) AND a new model (gpt-4o). Cannot separate. Parallel Pref result (+26.7pp, same new prompt family, SAME model gpt-4o-mini) shows near-identical magnitude — strongly suggesting the prompt alone explains most/all of TR's gain. Without ablation (old-prompt + gpt-4o, or new-prompt + gpt-4o-mini on TR), E2 cannot claim a "GPT-4o reader improvement" on TR.

**Resolution path:** Run E2a — TR-only with gpt-4o-mini + v3_routing prompts.

---

## Flag 2 — CROSS-PROMPT BASELINE COMPARISON — HIGH / LIVE

E2 uses v3_routing.py prompts; E0 used qa_eval_v2.py prompts. The +11.5pp headline delta includes prompt improvement for all qtypes. The two qtypes where only the prompt was held more stable (SSU 0pp, SSA 0pp — both at ceiling) show zero movement. No clean evidence the reader upgrade did anything independent of the prompt change.

**Resolution path:** E2a resolves this for TR. For MS, would need an E2b (old-prompt + gpt-4o-mini on MS). Accept as noted caveat if E2a shows gpt-4o adds meaningful delta beyond the prompt.

---

## Flag 3 — GRADER FAMILY BIAS ON PREF — MEDIUM / LIVE

gpt-4o-mini grades its own outputs on Pref (both reader and grader = gpt-4o-mini). LLM-as-judge literature shows same-family preference bias. However: (1) grading here is binary factual comparison (gold vs predicted), not subjective quality rating; (2) removing all 20 Pref correct answers still gives 335/440 = 76.1% — above 75% floor. Risk bounded.

**Resolution path:** Accept as noted caveat. Pref is n=30; grader bias on binary factual questions is lower than on open-ended quality. Not blocking.

---

## Flag 4 — FRAGMENTED RUN / REPRODUCE_COMMAND WRONG — MEDIUM / LIVE

Three sessions with --resume; multiple processes raced at one point (killed, no duplicate qids confirmed). receipt.json reproduce_command shows only `--qtypes knowledge-update` (last sub-run). Not fully reproducible from the stored receipt.

**Resolution path:** Re-emit correct receipt command. Noted in receipt.md caveats. Data integrity confirmed (no duplicate qids). Not blocking for proceeding — but final E5 must be single-process.

---

## Flag 5 — UNADJUSTED CI; CEILING MASKING REGRESSIONS — LOW / LIVE

95% CI [71.4%, 79.2%] lower bound doesn't exclude <75%. Six per-qtype comparisons unadjusted. KU +2.8pp on n=72 is within noise. SSU/SSA ceiling hides any regression signal.

**Resolution path:** n=470 is the full benchmark — no alternative sample size. Accept CI as-is; note that point estimate clears floor. Not blocking.

---

## Net Verdict

**PROCEED TO E2a, then E5**

Flags 1 and 2 are real and must be addressed before the paper can claim "GPT-4o reader adds X pp over gpt-4o-mini." The 75.5% result IS real as a system-level result (v3 prompts + GPT-4o routing together hits 75.5%), but the attribution is unresolved.

E2a (TR with gpt-4o-mini + v3 prompts, ~$2) resolves the main confound. Expected outcomes:
- If E2a TR ≈ E2 TR (66.9%): prompt is doing the work; GPT-4o adds little on TR
- If E2a TR << E2 TR: GPT-4o is genuinely needed; proceed to E5
- If E2a system overall ≥ 75%: can claim 75% with gpt-4o-mini + better prompts (cheaper, publishable)

Budget remaining: $50 - $9.77 = $40.23. E2a costs ~$2.
