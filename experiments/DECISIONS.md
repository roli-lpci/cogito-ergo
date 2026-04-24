# Experiment Decision Log
**Project:** cogito-ergo QA push — 50.4% → 75%+ on LongMemEval_S
**Started:** 2026-04-23

---

## After E0 (2026-04-23)

**Current QA:** 64.0% optimal blended (301/470, 95% CI [59.6%, 68.3%])
**Baseline:** 50.43% top-1, 45.32% top-5

**Evidence so far supports:**
- K routing as planned? YES — confirmed dramatically (MS K=5 wins 67.8% vs 24.8%; SSU K=1 wins 90.6% vs 23.4%)
- KU surprise: K=1 wins for knowledge-update (63.9% vs 58.3%) — not expected in roadmap
- Optimal blended is 64.0% (not 61.5% as estimated) — better than expected

**Decision: Proceed to E1 (MS synthesis prompt enhancement)**
**Rationale:** Gap to 75% is 11pp. MS accounts for 121/470 questions; even modest improvement in MS QA contributes ~2-5pp overall. Test synthesis prompt before paying GPT-4o cost.

---

## After E1 (2026-04-24)

**Current QA:** 64.0% blended optimal (E0). E1 was MS-only experiment.
**E1 finding:** v3 enhanced synthesis prompt REGRESSES on MS: 57.8% vs 69.7% baseline (-11.9pp). v2 prompt is better for gpt-4o-mini.
**12 questions dropped:** API rate limiting at end of run (last 12 MS questions by index).

**Evidence supports:**
- Planned E2 (GPT-4o reader)? YES — v3 prompt failed, so model upgrade is the only lever left
- Pivot to different approach? NO — regression is from prompt, not K routing. K=5 for MS still correct.
- Stop here? NO — 64% blended is 11pp short of floor. Must try GPT-4o.

**Decision: Proceed to E2 with v2 prompts (NOT v3), GPT-4o for MS/TR/KU**
**Rationale:** E1 proves gpt-4o-mini cannot leverage complex synthesis instructions. The reader model is the bottleneck. E2 tests whether GPT-4o can close the gap. Use v2 prompts in E2 to isolate the model upgrade effect cleanly (no prompt confound).

**E2 config:**
- GPT-4o: MS (K=5), TR (K=5), KU (K=1)
- gpt-4o-mini: SSU (K=1), SSA (K=1), Pref (K=1)
- Prompts: v2 for ALL qtypes (matching existing baseline)
- max_tokens: 512 (matching v2 baseline)
- Rate limit mitigation: 0.5s sleep between API calls

## After E2 — TBD

## After E3 — TBD

## After E4 — TBD

## After E5 — TBD
