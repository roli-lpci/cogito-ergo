# Fidelis Scaffold v0.1.0 — Publishability Plan (Today)

**Date:** 2026-04-25
**Author:** Claude Opus 4.7 under autonomous protocol
**Reviewed by:** Three sonnet subagents (academic / GTM funder / CEO perspectives — convergent diagnostic, divergent ship-decision)
**Gate:** This plan must be hermes-rubric'd at ≥ 8.0 before execution.

## Goal

A road to **publishable today** that leaves room to keep building. "Publishable" defined as: at least one shippable Hermes Labs artifact lands publicly today, with all placeholders filled, with audit receipts attached, and with v0.2 roadmap visible. Not paper-only — the *package* is also publishable on its own.

## Current state of placeholders (the hard gate)

Inventory of `{{}}` slots in `FLAGSHIP-PAPER.md`:

| Placeholder | Resolution source | Status |
|---|---|---|
| `{{F2_FULL_QA}}` (×3) | F2 full 470-Q run | ⏳ in progress |
| `{{F2_CI_LO}}`, `{{F2_CI_HI}}` | computed from F2 results | ⏳ derived after F2 |
| `{{F2_WALLCLOCK_MIN}}` | F2 timing | ⏳ derived after F2 |
| `{{F2_SSU/SSA/KU/MS/PREF/TR}}` (×6) | F2 per-qtype | ⏳ in progress |
| `{{F1B_OVERALL}}` | F1B baseline | ✅ have it: 75.6% (n=41 partial) |
| `{{F1B_SSU/SSA/KU/MS/PREF}}` (×5) | F1B per-qtype | ✅ have them |
| `{{F1B_TR}}` | F1B-TR-only | ⏳ running NOW (~10 min) |
| `{{F1_VS_F1B_LIFT}}` | derived from F1 + F1B | ✅ derivable |
| `{{COMMIT_SHA}}` | git commit | ✅ have it: 8255a0f |

**14 unique placeholders** mapping to **3 resolution sources**:
1. F2 full eval result (multiple slots)
2. F1B-TR-only result (one slot)
3. Local computation (already have)

## Plan: Two-Track Publish

### Track A — Package OSS release **TODAY** (no placeholders required)

The package itself has zero placeholder dependencies. Ship the package as v0.1.0 with smoke-only evidence. Paper follows in Track B.

**Phase A.1 — Pre-flight gate (30 min, NOW)**
- Wait for F1B-TR-only smoke to complete (~10 min wallclock)
- Fill the `{{F1B_TR}}` slot in the README's smoke table
- Run `hermes-rubric` on package README + scaffold source with intent: "score for methodology clarity, scope discipline, evidence-grounding, no fabrication, and Hermes-quality OSS standards — gate ≥ 8.0 to ship"
- **Hard gate:** rubric ≥ 8.0. If < 8.0, do NOT proceed to A.2; iterate README/scope/claims until passes.

**Phase A.2 — Push (15 min, gate-conditional, REQUIRES Roli approval)**
- Three binary decisions from Roli (already requested): push / visibility / org
- If approved + gate green: push `~/Documents/projects/fidelis-scaffold/` to chosen GitHub remote
- Tag `v0.1.0`, create release notes citing smoke-only evidence + "full 470 paper landing within 24h"
- `hermes-seal sign` the release manifest
- Run `hermes-coc-export --from 2026-04-25 --to 2026-04-25` for the audit bundle

**Phase A.3 — Verify post-push (5 min)**
- Confirm clone URL works on a fresh clone test
- Confirm pip install -e . works on fresh clone
- Confirm tests still pass
- Update HANDOFF.md to reflect v0.1.0 shipped state

### Track B — Paper publishes when F2 lands (today if possible)

**Phase B.1 — Wait + monitor F2** (background, no manual action)
- F2 in progress at ~92/470 as of plan-write
- Auto-checkpoint at n=200 (statistical floor for Wilson CI ±5pp)
- If F2 hits 470: fill all `{{F2_*}}` placeholders from real data
- If F2 stalls or wallclock blows past Roli's wake: ship paper as "n=200 partial Wilson-CI'd preliminary" or hold to next session

**Phase B.2 — Fill + rubric (45 min after F2 completes)**
- Sed all `{{F2_*}}` placeholders with real values from `evidence/F2-FULL.json`
- Compute 95% Wilson CI on overall + per-qtype
- Reframe abstract per funder feedback: lead with cost-accuracy wedge, not leaderboard
- Run `hermes-rubric` on completed paper with intent: "score for SOTA-flagship release readiness, evidence-grounding, scope discipline, oversteering restraint, and methodology audit-completeness — gate ≥ 8.0"
- **Hard gate:** rubric ≥ 8.0. If < 8.0, iterate on framing (NOT data — data is what it is).

**Phase B.3 — Push paper (15 min, gate-conditional)**
- Add `PAPER.md` to repo + push
- Optionally arxiv submission (Roli decision)
- Update README to link to filled paper
- Re-run `hermes-coc-export` for the paper-bundle

### Track C — Fallback if F2 doesn't complete today

- Track A still ships (package, no F2 dependency)
- Paper stays as draft with placeholders in HANDOFF
- Schedule auto-resume agent to fill placeholders + ship paper when F2 lands tomorrow
- Today's ship moment is "Hermes Labs ships fidelis-scaffold v0.1.0 OSS package; paper landing within 24h with full eval"

## Hard gates (non-negotiable, audit-tested)

1. **F1B-TR-only completes.** No paper or package update without TR baseline number.
2. **hermes-rubric ≥ 8.0 on package** before any push.
3. **hermes-rubric ≥ 8.0 on paper** before paper push.
4. **No `{{}}` in any pushed artifact.** Either filled or removed-with-explicit-scope-mark.
5. **Roli approval on push action.** Public-repo-gate hook enforces; this plan does not bypass.
6. **Rubric iteration ceiling.** If two rubric iterations on the same artifact still produce score < 8.0, treat as Track C extended (hold artifact, do not ship today, surface to Roli with diagnosis). Prevents infinite-loop on a stubborn gate failure.

## What this plan does NOT do

- Does not auto-push without explicit Roli go on each push action
- Does not invent numbers to fill placeholders
- Does not lower the rubric gate from 8.0
- Does not skip hermes-coc-export or hermes-seal steps
- Does not attempt to backfit a "SOTA flagship" claim if F2 lands below 70% — would reframe as cost-frontier methodology release instead
- Does not block on multi-turn drift (deferred to v0.2 explicitly)

## Reversibility

Every step is reversible:
- Push can be reverted with a force-push (with Roli explicit approval)
- Paper revision can be re-rubric'd
- Failed gates produce a HANDOFF.md update so the next session can pick up cleanly

## Time budget (relative to plan-write at 06:35 PT)

| Phase | Earliest | Latest | Blocker |
|---|---|---|---|
| F1B-TR finish | ~06:50 PT | ~07:00 PT | LLM throughput |
| Package rubric green | ~07:15 PT | ~07:45 PT | rubric runtime |
| Package push (after Roli go) | ~07:45 PT | ~08:30 PT | Roli decision |
| F2 finish | ~09:00 PT | ~13:00 PT | LLM throughput |
| Paper rubric green | ~10:00 PT | ~14:00 PT | F2 + iteration |
| Paper push | ~10:30 PT | ~14:30 PT | Roli decision |

Track A is independent of Track B's timing. Worst case Track A: shipped by 08:30 PT.

## Definition of done (today)

**Minimum publishable smack:**
- ✅ fidelis-scaffold v0.1.0 OSS pushed to GitHub
- ✅ README cites real smoke numbers (no placeholders)
- ✅ hermes-rubric receipt attached at gate ≥ 8.0
- ✅ hermes-seal manifest + hermes-coc-export bundle in `evidence/`
- ✅ HANDOFF.md updated to reflect v0.1.0 state

**Stretch publishable smack (if F2 finishes today):**
- ✅ FLAGSHIP-PAPER.md filled with real numbers
- ✅ Paper rubric green
- ✅ Optional arxiv submission

**Non-goals today:**
- v0.2 (multi-turn drift, qwen-tier full eval, cross-grader)
- Hosted demo / blog post
- Social media announcement
