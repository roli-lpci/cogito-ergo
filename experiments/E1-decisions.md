## After E1 (2026-04-24)

Current QA: 57.8% on MS only (n=109 of 121). Delta vs E0 MS K=5 baseline: -11.9pp on matched subset. E0 overall blended optimal still stands at 64.04% (E1 only touched MS qtype, did not re-run full benchmark).

Evidence so far supports:

- **Planned next step (E2 — GPT-4o reader)?** YES, but with a warning. E1 proved v3 synthesis prompt regresses gpt-4o-mini. E2 tests whether GPT-4o handles the same prompt correctly. If GPT-4o also regresses, we must revert to v2 prompts entirely. If GPT-4o succeeds, the finding is "model capability gating on synthesis instructions" — a real result.

- **Pivot option (use v2 prompts for E2)?** CONSIDERED but deferred. Running E2 with `--v2-prompts` would break the "E2 = GPT-4o reader" hypothesis isolation. The adversarial flag (Flag 5: gameable regression) will be answered by E2 results. A v2+GPT-4o ablation (E2b) is budget-contingent after E2.

- **Stopping here?** NO. E0 blended optimal is 64.04%, 11pp short of the 75% floor. E1 is a negative result but does not change the path — it informs E2 design.

Decision: **Proceed to E2 (GPT-4o reader, v3 prompts, full 470 questions, --gpt4o-for-multi)**

Rationale: GPT-4o's stronger reasoning should handle the citation/synthesis requirements that broke gpt-4o-mini. The adversarial Flag 5 (gameable regression) will be resolved by E2 results — if GPT-4o also fails, we abandon v3 MS prompt; if it succeeds, the regression was model-capacity-limited, not manufactured. Budget allows E2 at current $0.26 spent vs $50 cap.
