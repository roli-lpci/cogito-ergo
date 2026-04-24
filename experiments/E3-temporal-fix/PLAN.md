# E3 — Temporal Reasoning Prompt Fix

**Status:** READY — triggers after E2 completes
**Date planned:** 2026-04-24
**Expected cost:** ~$8-12 (127 TR questions × GPT-4o)

## Root Cause Found in E2

E2 TR analysis (56 questions done): **20/24 wrong answers (83%)** are "today reference" errors:
- Pattern A: GPT-4o uses training cutoff date (2023/10/xx) as "today" → computes nonsense elapsed time
- Pattern B: GPT-4o uses most recent session date as "today" = 0 days elapsed → wrong reference event
- Neither error is arithmetic — the date arithmetic itself is correct once dates are established
- Fix: explicit prompt instruction about what "today" means in this context

## The Fix

Current `_QA_SYS_TEMPORAL` ends with:
```
"Guidance: Dates appear in session headers as 'YYYY/MM/DD (Day)'. "
"If computing days between dates, use calendar arithmetic. "
"Off-by-one is forgiven for day counts."
```

**Add after the guidance:**
```
"CRITICAL — Reference date rule: This is a conversation history, not real-time. "
"There is NO external 'today' date. "
"For questions asking 'how many days ago', 'how long ago', or similar: "
"the reference point IS ANOTHER EVENT in the conversation, not today. "
"Find both events (the one asked about AND the reference event), "
"both must appear in the session dates. "
"Compute the difference between their session dates ONLY."
```

## Why This Will Work

- Error is systematic and GPT-4o-specific (it knows its training cutoff)
- The fix gives explicit grounding: "find BOTH events in the conversation"
- Questions like "How many days ago did X happen when I did Y?" → X and Y are both in sessions
- This is a zero-cost prompt change (no model change, no K change, no data change)

## Experiment Config

- **qtypes:** temporal-reasoning only (`--qtypes temporal-reasoning`)
- **model:** GPT-4o (`--gpt4o-for-multi` — covers TR)
- **K:** 5 (same as E2)
- **prompt:** v3 temporal with today-reference fix
- **max_tokens:** 512 (same as E2, no confound)
- **n:** 127 questions
- **resume:** no (fresh run on TR)

## Expected Outcome

- TR E2: ~57% (72/127)
- TR E3 target: 85%+ (108+/127), based on 83% of errors being fixable
- Net gain: +28 correct on TR = +5.9pp overall
- E3 blended = E2 non-TR + E3 TR:
  - E2 non-TR: SSU=58, MS=91, Pref=20, KU=??, SSA=?? (wait for E2 to finish)
  - E3 TR: ~108/127
  - Projected: ~78-82% overall depending on KU result

## Command (run after E2 finishes)

```bash
cd ~/Documents/projects/cogito-ergo
python3 bench/qa_eval_v3_routing.py \
  --run-id runP-v35 \
  --experiment-id E3 \
  --gpt4o-for-multi \
  --qtypes temporal-reasoning \
  --data-dir ~/Documents/projects/LongMemEval/data \
  --out-dir experiments/E3-temporal-fix
```

## Adversarial Gate Pre-checks

1. Is the "today" fix really the cause, or a confound?
   - Verify by checking: do CORRECT TR answers also reference "today"?
   - Check 2-3 correct answers for "today" references
2. Does improving TR cross-contaminate (share sessions with other qtypes)?
   - No — TR is a separate qtype with its own questions
3. Is the fix fabricated or evidence-based?
   - Evidence: 20/24 wrong answers explicitly show "today" or training cutoff date
4. What's the expected cost? Verify it's within budget.
   - 127 × GPT-4o at avg $0.04/call = ~$5. Well within $50 budget.
