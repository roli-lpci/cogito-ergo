# HANDOFF — fidelis-scaffold v0.1.0

**Last update:** 2026-04-25, mid-build
**Built by:** Claude Opus 4.7 (1M context) under Roli's autonomous all-night protocol
**Session:** 7721bc7e-d87a-4578-b6d2-9659b3d7a04c

## What's done
- ✅ Scaffold core: `src/fidelis_scaffold/scaffold.py` (~150 lines, 6 qtype procedures)
- ✅ Preflight validator: `src/fidelis_scaffold/preflight.py` (8 checks)
- ✅ Tests: 12 passing (`tests/test_scaffold.py`)
- ✅ Preflight script: `scripts/scaffold-preflight.sh`
- ✅ INTENT.md, README.md, agents.md, pyproject.toml, this HANDOFF
- ✅ FLAGSHIP-PAPER.md skeleton (numbers populated as evals land)
- ✅ Smoke F1 (n=60 stratified): **75.0% with scaffold + Opus** (vs 55% extractive baseline)
  - SSU 100, SSA 100, KU 70, MS 70, Pref 60, TR 50

## What's running
- ⏳ F2: full 470-Q eval with scaffold + Opus reader + Opus grader → `experiments/zeroLLM-FLAGSHIP-FULL/`
- ⏳ F1B: 60-Q baseline (Opus + minimal prompt, no scaffold) for A/B lift measurement → `experiments/zeroLLM-FLAGSHIP/`

## What's deferred to v0.2
- Multi-turn drift eval (driftwatch + agent-convergence-scorer integration). Code exists at `scripts/multi_turn_drift.py`. Synthetic follow-up generator at `scripts/generate_followups.py`. Was killed mid-run due to subscription rate-limit when running parallel to main evals.
- Local-tier full eval (qwen3.5:9b reader on 470). Same scaffold, different reader.
- Cross-grader replication (gpt-4o as second grader to confirm Opus grader doesn't bias).

## What's not done
- COC bundle of all evidence (run `hermes-coc-export` after F2 completes)
- hermes-rubric gate run on FLAGSHIP-PAPER (after F2 numbers are filled in)
- hermes-seal manifest signing
- GitHub repo init (intentional — `.git` not initialized; do not push without explicit go)

## To pick up
1. Wait for F2 + F1B to finish (check `evidence/` paths above)
2. Fill the {{F2_*}} and {{F1B_*}} placeholders in `FLAGSHIP-PAPER.md`
3. Run `hermes-coc-export --from 2026-04-25 --to 2026-04-25` and put bundle in `evidence/`
4. Run `hermes-rubric` on `FLAGSHIP-PAPER.md` with intent "score for SOTA flagship release readiness, evidence-grounding, scope discipline, oversteering restraint, naming coherence, and methodology audit-completeness — gate ≥ 8.0"
5. If rubric ≥ 8.0: run `hermes-seal sign` and stage the v0.1.0 release artifacts. If < 8.0: iterate paper, do not ship.
6. **Do NOT push to GitHub or publish anywhere without explicit Roli approval.**

## Open hard questions that need a human
- Is the cost framing ("$0 with subscription, marginal with API") clear enough? Need an external read.
- Does the local-tier qwen path need to be in v0.1 release or can it wait?
- "Flagship" branding — is this the flagship paper, or a v0.1 step toward it?

## Files index
```
fidelis-scaffold/
├── INTENT.md
├── README.md
├── agents.md
├── HANDOFF.md           ← you are here
├── FLAGSHIP-PAPER.md    ← fill in numbers from evidence/
├── pyproject.toml
├── src/fidelis_scaffold/
│   ├── __init__.py
│   ├── scaffold.py
│   └── preflight.py
├── tests/test_scaffold.py
├── scripts/
│   ├── scaffold-preflight.sh
│   ├── generate_followups.py    (for multi-turn drift, v0.2 work)
│   └── multi_turn_drift.py      (v0.2 work)
└── evidence/               ← eval JSONs + rubric outputs
```
