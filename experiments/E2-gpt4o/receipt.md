# E2 Receipt — GPT-4o Reader

**Experiment:** E2
**Date:** 2026-04-24
**Run ID:** runP-v35

## Config
- GPT-4o reader: MS (K=5), TR (K=5), KU (K=1)
- gpt-4o-mini reader: SSU (K=1), SSA (K=1), Pref (K=1)
- Prompts: v2 for all qtypes (v3 temporal for TR)
- Grader: gpt-4o-mini
- max_tokens: 512

## Result
- **Overall QA:** 75.5% (355/470)
- **95% CI:** [71.4%, 79.2%] (Wilson, n=470)
- **E0 optimal blended:** 64.0%
- **Delta vs E0:** +11.5%
- **Cost:** $9.774 USD
- **Tokens:** 237,397 in / 8,957 out

## Per-Qtype Breakdown

| qtype                          | n   | baseline | E2 acc | delta   | reader         | K     |
|--------------------------------|-----|----------|--------|---------|----------------|-------|
| knowledge-update               |  72 |  63.9% |  66.7% |   +2.8% | gpt-4o         | K=1 |
| multi-session                  | 121 |  67.8% |  76.0% |   +8.3% | gpt-4o         | K=5 |
| single-session-assistant       |  56 |  92.9% |  92.9% |   +0.0% | gpt-4o-mini    | K=1 |
| single-session-preference      |  30 |  40.0% |  66.7% |  +26.7% | gpt-4o-mini    | K=1 |
| single-session-user            |  64 |  90.6% |  90.6% |   -0.0% | gpt-4o-mini    | K=1 |
| temporal-reasoning             | 127 |  40.2% |  66.9% |  +26.8% | gpt-4o         | K=5 |

## Key Findings

1. **Floor met**: 75.5% clears the 75% target. E0 → E2 = +11.5pp.
2. **Pref massive gain**: +26.7pp (40% → 66.7%) driven by improved prompt in v3_routing.py, not GPT-4o (Pref uses gpt-4o-mini). Prompt confound vs E0 baseline.
3. **TR massive gain**: +26.8pp (40.2% → 66.9%) from GPT-4o + improved temporal prompt. Error analysis: ~29% of TR errors were today-reference errors (GPT-4o uses training cutoff as "today"). Remaining 71% are wrong-event identification and arithmetic errors.
4. **MS improvement**: +8.3pp (67.8% → 76.0%) from GPT-4o reader alone (same v2 prompts).
5. **KU modest gain**: +2.8pp (63.9% → 66.7%) from GPT-4o upgrade.
6. **SSA unchanged**: 92.9% (ceiling effect, gpt-4o-mini K=1 at baseline).

## Caveats

- **Run fragmentation**: E2 ran across 3 sessions with `--resume`. reproduce_command in receipt.json incorrectly shows `--qtypes knowledge-update` (last sub-run only). Full command: `--gpt4o-for-multi --data-dir ... --resume`.
- **Multiple processes race**: Multiple E2 processes ran concurrently at some point. Killed all but 64294. No duplicate qids found — data integrity confirmed.
- **Prompt confound on Pref**: E2 uses v3_routing.py prompts which are substantially better than qa_eval_v2.py prompts for Pref and SSU. The Pref gain (+26.7pp) cannot be attributed to GPT-4o — it comes from the new prompt.
- **CI note**: 95% CI lower bound is 71.4%. Point estimate (75.5%) is above floor but CI doesn't exclude < 75%.

## Decision Gate

- Target: ≥75.0%
- E2 result: 75.5% (point estimate)
- **FLOOR MET** — E2 is the stopping point. Run E5 for final sealed measurement.
- Stretch (85%): not met. E3 TR-fix could add ~3-5pp → 78-80%.
