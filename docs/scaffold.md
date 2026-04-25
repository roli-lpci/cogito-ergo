# fidelis.scaffold — the Fidelis QA scaffold technique

The QA scaffold is the v0.1.0 flagship feature. This doc captures the full contract; the README has the headline pitch and per-qtype lift table.

## Module surface

```python
from fidelis.scaffold import (
    SCAFFOLD_VERSION,        # "v0.1.0"
    SCAFFOLD_OPEN,           # "[FIDELIS-SCAFFOLD-v0.1.0]"
    SCAFFOLD_CLOSE,          # "[/FIDELIS-SCAFFOLD-v0.1.0]"
    wrap_system_prompt,      # build a scaffold for a given qtype + retrieval-confidence score
    wrap_idempotent,         # idempotent variant: wrap(wrap(x)) == wrap(x)
    is_scaffolded,           # detect any version of the Fidelis scaffold in arbitrary text
    strip_scaffold,          # remove all Fidelis scaffold markers from text
    preflight,               # 8-check static validator → PreflightReport
    preflight_or_raise,      # raises RuntimeError on failure
    PreflightReport,         # dataclass: .passed / .failures / .warnings / .metrics
)
```

## Six qtype procedures (LongMemEval-aligned)

| qtype | Procedure | Token budget |
|---|---|---|
| `single-session-user` | Quote user's exact statement, answer concisely | ~141 |
| `single-session-assistant` | Quote assistant's exact statement, answer using only the quote | ~142 |
| `single-session-preference` | Aggregate ALL stated preferences, recommend grounded | ~153 |
| `knowledge-update` | List quotes by date, MOST RECENT is the current answer | ~151 |
| `multi-session` | For each session, quote relevant; combine across sessions | ~170 |
| `temporal-reasoning` | Find session-date headers, compute calendar diff between TWO DATES | ~174 |

Plus, for every qtype:

1. **Calibrated hedge invitation:** *"If retrieval doesn't contain the answer, respond exactly: 'I cannot answer this from the retrieved memory.'"* Overrides the prior pattern of forcing the LLM to guess.
2. **Inline retrieval-confidence signal:** `[retrieval-quality: HIGH/MEDIUM/LOW]` from `top_score`.
3. **Versioned scaffold markers:** `[FIDELIS-SCAFFOLD-vX.Y.Z]…[/FIDELIS-SCAFFOLD-vX.Y.Z]` for downstream drift detection.

## Preflight (8 checks, all must pass before any release)

1. Forbidden control tokens (`<|endoftext|>`, `<|im_start|>`, etc.) — tokenizer-hijacking prevention
2. Balanced fences/brackets/parens — downstream parser-error prevention
3. Length bound (≤200 tokens, hard) — context-bloat prevention
4. ASCII-clean / UTF-8 NFC normalization — encoding-mismatch prevention
5. Idempotency — compounding-drift prevention
6. No nested scaffold markers — accidental-double-wrap prevention
7. Token-count stability — provider-tokenization-drift prevention
8. Marker integrity — open and close pair count

## Why a scaffold and not a fine-tuned reader

A fine-tuned reader specializes you for one model. The scaffold:

- Works with any LLM (Claude, GPT, qwen-local, anything that takes a system prompt)
- Costs $0 at the scaffold layer (zero training, ≤200 tokens of system prompt that can be cached)
- Stays auditable (you can read the 150 tokens; can't read a fine-tuned reader)
- Composes cleanly with prompt caching (system prompt cached → effective marginal cost ~$0)

## What the scaffold does NOT do

- Does not modify LLM weights
- Does not fine-tune anything
- Does not perform retrieval (consumes retrieval; pair with `fidelis.recall_*`)
- Does not replace the user's existing system prompt without explicit caller intent

## The drift-detection property

Versioned markers enable downstream tools to:

1. Detect a Fidelis scaffold's presence in a conversation history
2. Locate where it begins/ends
3. Strip it for clean turn-2 measurement
4. Compare arm-A (no scaffold) vs arm-B (scaffold-then-stripped) on subsequent turns

This is the load-bearing safety property for production multi-turn agents. Scaffold tokens injected at turn-1 stay in conversation history — measurable drift on later turns is the legitimate concern. Versioned markers turn that concern from theoretical to testable.

Multi-turn drift measurement infrastructure is present in v0.1.0 (markers + `is_scaffolded` / `strip_scaffold`); empirical multi-turn measurement is v0.2 work.

## Hard rule

**Do not strip the scaffold-version markers.** PRs that remove versioning will be rejected. This is the one non-negotiable rule on the package.

## Audit chain

The scaffold's release goes through Hermes Labs' eight-tool audit chain:

- `scaffold-lint` + `lintlang` — static lint of the scaffold strings
- `hermes-blind` — context-bias nullification on the eval grader
- `hermes-rubric` — evidence-first scoring on the methodology (gate ≥ 8.0)
- `hermes-coc-export` — chain-of-custody bundle for eval receipts
- `hermes-seal` — release manifest signing
- `driftwatch` (langquant-sdk) — multi-turn drift measurement (v0.2)
- `agent-convergence-scorer` — convergence steering measurement (v0.2)
- preflight — 8 static safety checks built into the package

See `experiments/zeroLLM-FLAGSHIP-evidence/` for raw smoke JSONs, the companion paper for the full methodology + receipts.
