# E4 — Per-Qtype Dedicated Reader Prompts

**Status:** CONTINGENCY — only run if E2 + E3 fall short of 75% floor

## Target qtypes
1. **Preference** (n=30, currently ~40%): reader must infer preferences and apply them
2. **Temporal** (n=127, currently ~30-40%): date math, ordering, duration calculation

## Preference prompt enhancement
Current issue: reader quotes preference but gives generic recommendation.
Fix: "Your recommendation must explicitly name the brand/product/style the user stated. 
Don't suggest alternatives — use what they said they use/prefer."

## Temporal prompt enhancement (if v3 temporal didn't already help in E2)
Current issue: reader finds dates but makes arithmetic errors or misorders events.
Fix: structured date extraction → chronological sort → explicit calculation steps.
Check if E2 temporal improved vs baseline first.

## Expected delta
- Preference: +10-20pp on preference (n=30) = +0.6-1.3pp overall
- Temporal: +5-10pp on temporal (n=127) = +1.3-2.7pp overall
- Combined: +2-4pp overall

## Decision Gate
- Only run if gap after E2 (±E3) is ≥ 2pp
- If gap is <1pp, finish at E5 without E4
