# E1 — Adversarial Review
**Reviewer:** Fresh-context Claude subagent (Sonnet 4.6)
**Date:** 2026-04-24
**Verdict:** FLAGGED — 5 flags, 2 resolved, 3 accepted as limitations

---

## Flag 1: Methodology Change — Unfair Comparison
**Severity: HIGH**

E0's 67.8% MS K=5 was computed post-hoc by cherry-picking the best K from two existing runs (no live API calls). E1 is a live run with a different prompt. The -10pp delta compares E1 against its own cherry-picked historical ceiling, not a matched control.

**Response:** PARTIALLY RESOLVED. The receipt's per-question analysis (19 regressed, 6 improved on the matched 109-question subset) IS a valid apples-to-apples comparison — both v2 and v3 evaluated on the same questions. The -11.9pp on matched subset is the defensible number; -10.0pp vs E0 overall is the noisier headline. Downstream reports should cite -11.9pp as the matched delta.

---

## Flag 2: Selection Bias from 12 Skipped Questions
**Severity: MEDIUM**

12/121 questions (9.9%) skipped silently. If harder questions produce longer prompts that time out first, skips are non-random. E0 baseline had no skips (post-hoc computation on complete JSON). Denominator mismatch (121 vs 109) means the populations differ.

**Response:** ACCEPTED LIMITATION. The 12 questions failed at positions [107-121] in sequential run order due to API rate limiting near the end, not inherent difficulty. Cannot fully rule out positional bias. Impact on the -10pp delta: if the 12 skipped questions performed at E0's 67.8% rate, overall would be ~58.7% vs 57.8% — immaterial. Flag stands as a caveat on the receipt.

---

## Flag 3: Same Model as Reader and Grader (Correlated Errors)
**Severity: LOW (structural)**

gpt-4o-mini is both reader and grader. The grader may be more lenient toward well-structured but wrong answers produced by the same model it's evaluating.

**Response:** ACCEPTED — but shared across E0 and E1. Since both experiments use the same grader model, the error correlation is constant and does not explain the differential. This is a known limitation of the whole evaluation framework (not E1-specific). No action needed for delta comparison.

---

## Flag 4: Reproducibility Command Missing Data Hash
**Severity: MEDIUM**

The reproduce command lacks a commit SHA. `per_question.json` is not cryptographically hashed. A re-runner cannot verify their retrieval input matches E1's input.

**Response:** ACCEPTED. Mitigation: `per_question.json` is committed to git. A re-runner can verify `git log bench/runs/runP-v35/per_question.json` to confirm the file hasn't changed. SHA256 should be computed and added. **TODO before E2: compute and embed data hash in receipts.**

---

## Flag 5: Regression is Gameable (Constructed Stepping Stone)
**Severity: HIGH — cannot fully resolve**

The v3 prompt added citation requirements known to strain small models. E2 (GPT-4o) was planned before E1 ran. The "regression" is exactly the failure mode that justifies the upgrade. This could be a constructed stepping stone.

**Response:** ACKNOWLEDGED AND DOCUMENTED. Cannot falsify this concern within E1 alone. Mitigation plan:
1. E2 will run the **same v3 prompt** with GPT-4o. If GPT-4o also regresses vs E0, the v3 prompt is genuinely bad and E2 is a dud — not a convenient win.
2. If E2 succeeds, we note the asymmetry (v3 prompt works with GPT-4o but not mini) as a real finding, not a trick.
3. A v2-prompt + GPT-4o ablation (E2b) would fully close this flag, but is outside current budget scope.

**Verdict on Flag 5: LIVE FLAG — E2 results will either corroborate or escalate this concern.**

---

## Bonus Flags

**string_containment_check inflation risk:** The grading fast-path passes if the gold answer string appears anywhere in output — verbose answers that contain the answer word without correct reasoning could inflate accuracy. This is structural to the evaluation framework (inherited from LongMemEval's own `evaluate_qa.py`) and applies equally to E0 and E1. No differential impact.

**Cost figure: CLEAN.** $(1,544,098 × 0.15/1M) + (45,649 × 0.60/1M) = $0.232 + $0.027 = $0.259. Matches receipt exactly.

**Test-set leakage: LOW RISK.** Reader system prompts contain no gold answer strings or session IDs. Grading prompts verbatim from LongMemEval's own code. Clean.

---

## Net Verdict

The regression is real on matched questions (-11.9pp, 19 regressions vs 6 improvements). The comparison against E0's 67.8% is methodologically impure but not dishonest. The gameable-regression flag (Flag 5) is the most concerning and cannot be closed without E2 results. E1 should NOT be cited as "synthesis prompts hurt" in any paper; cite only "v3 synthesis prompt hurts gpt-4o-mini" with E2 as the test of GPT-4o response.

**Cleared to proceed to E2 with caveats documented.**
