# E3 — Langstate Compression for Multi-Session Context

**Status:** CONTINGENCY — only run if E2 falls short of 75% floor

## Hypothesis
Use `langstate.compress(messages)` to compress each retrieved session to ~300 tokens
before feeding to the reader. Feed up to 8 compressed sessions instead of 5 raw —
same token budget, more session coverage.

**Expected delta:** +3-5pp on MS/TR/KU → +1-3pp overall

## Why it might help
- 5 raw sessions at ~2000 chars each = ~10000 chars context → diluted
- 5 compressed sessions at ~500 chars each = ~2500 chars context → denser, less noise
- More sessions visible at same token budget → better multi-session coverage

## Why it might not help
- langstate compresses OpenAI message format, not raw session text — adaptation needed
- Compression might lose the exact facts needed for QA
- Adversarial gate: "verify compression is lossless on the facts actually needed"

## Implementation Plan (if needed)
1. Check langstate availability: `python3 -c "from langstate import compress; print('ok')"`
2. Wrap langstate around session text: convert session text → message format → compress → convert back
3. Run on MS-only (n=121) at K=8 with compressed sessions
4. Verify 10 random compression cases manually: gold answer fact must survive compression
5. Compare to E2 MS accuracy

## Adversarial Gate
Run fresh-context subagent to verify compression is lossless on gold answer facts.

## Decision Gate
- If E2 ≥ 75%: skip E3
- If E2 < 75%, E3 worth testing if: |E3_expected_gain| > |cost_of_dev + error_margin|
