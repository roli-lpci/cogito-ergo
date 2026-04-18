# Qwen-Native Scaffold Arena Results

**Date:** 2026-04-16  
**Baseline run:** runC-guard  
**Model:** qwen-turbo (dashscope-intl)  
**Total qwen-turbo calls:** 358  
**Parse failures:** 23 | **API errors:** 0  
**Set A size:** 20 questions (gold in top-5, wrong @1)  
**Previously tested (transfer validation):** 6 scaffolds  
**New scaffolds tested:** 15  

## 1. Full Win-Rate Table (All 21 Scaffolds, Sorted by Win Rate)

| Scaffold | Overall | knowledge-update | multi-session | single-session-assistant | single-session-preference | temporal-reasoning | Source |
|----------|---------|-----------------|---------------|--------------------------|---------------------------|--------------------|--------|
| s14-union-s8-contrastive | **15/20 (75%)** | 1/2 | 1/2 | 1/1 | 3/3 | 9/12 | new |
| s8v3-socratic-with-dates | **14/20 (70%)** | 1/2 | 1/2 | 1/1 | 3/3 | 8/12 | new |
| s13-union-s8-korean | **14/20 (70%)** | 1/2 | 1/2 | 1/1 | 3/3 | 8/12 | new |
| s8v2-socratic-structured | **12/20 (60%)** | 1/2 | 1/2 | 1/1 | 1/3 | 8/12 | new |
| s8-socratic | **12/20 (60%)** | 1/2 | 1/2 | 1/1 | 3/3 | 6/12 | prev-tested |
| hypothesis-generate-pick | **11/20 (55%)** | 1/2 | 1/2 | 1/1 | 0/3 | 8/12 | new |
| qwen-role-expert | **11/20 (55%)** | 0/2 | 1/2 | 1/1 | 3/3 | 6/12 | new |
| contrastive-explicit | **10/20 (50%)** | 1/2 | 1/2 | 1/1 | 0/3 | 7/12 | new |
| empty-format-only | **10/20 (50%)** | 1/2 | 1/2 | 0/1 | 2/3 | 6/12 | prev-tested |
| aggregation-frame | **9/20 (45%)** | 1/2 | 1/2 | 1/1 | 1/3 | 5/12 | new |
| qwen-chinese-instruction | **9/20 (45%)** | 0/2 | 1/2 | 1/1 | 1/3 | 6/12 | new |
| s15-scaffold-chain | **9/20 (45%)** | 2/2 | 1/2 | 1/1 | 3/3 | 2/12 | new |
| bayesian-prior | **8/20 (40%)** | 1/2 | 1/2 | 1/1 | 0/3 | 5/12 | new |
| s2-temporal-inline | **8/20 (40%)** | 0/2 | 1/2 | 0/1 | 1/3 | 6/12 | prev-tested |
| s4-declarative-preference | **8/20 (40%)** | 0/2 | 1/2 | 0/1 | 1/3 | 6/12 | prev-tested |
| s0-minimal-baseline | **7/20 (35%)** | 2/2 | 1/2 | 0/1 | 1/3 | 3/12 | prev-tested |
| s1-temporal-sysprefix | **7/20 (35%)** | 1/2 | 1/2 | 1/1 | 1/3 | 3/12 | prev-tested |
| multi-perspective-vote | **6/20 (30%)** | 0/2 | 0/2 | 0/1 | 2/3 | 4/12 | new |
| self-consistency | **4/20 (20%)** | 0/2 | 0/2 | 0/1 | 1/3 | 3/12 | new |
| korean-evidential-commit | **3/20 (15%)** | 0/2 | 0/2 | 0/1 | 1/3 | 2/12 | new |
| anti-scaffold-contrarian | **1/20 (5%)** | 0/2 | 0/2 | 0/1 | 0/3 | 1/12 | new |

## 2. Per-Qtype Best Scaffold

### knowledge-update (n=2 failure questions)

- s15-scaffold-chain: 2/2 (100%)
- s0-minimal-baseline: 2/2 (100%)
- s14-union-s8-contrastive: 1/2 (50%)
- s8v3-socratic-with-dates: 1/2 (50%)
- s13-union-s8-korean: 1/2 (50%)

### multi-session (n=2 failure questions)

- s14-union-s8-contrastive: 1/2 (50%)
- s8v3-socratic-with-dates: 1/2 (50%)
- s13-union-s8-korean: 1/2 (50%)
- s8v2-socratic-structured: 1/2 (50%)
- s8-socratic: 1/2 (50%)

### single-session-assistant (n=1 failure questions)

- s14-union-s8-contrastive: 1/1 (100%)
- s8v3-socratic-with-dates: 1/1 (100%)
- s13-union-s8-korean: 1/1 (100%)
- s8v2-socratic-structured: 1/1 (100%)
- s8-socratic: 1/1 (100%)

### single-session-preference (n=3 failure questions)

- s14-union-s8-contrastive: 3/3 (100%)
- s8v3-socratic-with-dates: 3/3 (100%)
- s13-union-s8-korean: 3/3 (100%)
- s8-socratic: 3/3 (100%)
- qwen-role-expert: 3/3 (100%)

### temporal-reasoning (n=12 failure questions)

- s14-union-s8-contrastive: 9/12 (75%)
- s8v3-socratic-with-dates: 8/12 (67%)
- s13-union-s8-korean: 8/12 (67%)
- s8v2-socratic-structured: 8/12 (67%)
- hypothesis-generate-pick: 8/12 (67%)

## 3. Qwen-Native SCAFFOLD_DISPATCHER Config

```python
# Qwen-turbo validated scaffold dispatcher (v0.3.1)
# Source: qwen_native_arena.py, runC-guard failure set, 2026-04-16
SCAFFOLD_DISPATCHER_QWEN = {
    "knowledge-update": "s15-scaffold-chain",  # 2/2 (100%) on failure set
    "multi-session": "s14-union-s8-contrastive",  # 1/2 (50%) on failure set
    "single-session-assistant": "s14-union-s8-contrastive",  # 1/1 (100%) on failure set
    "single-session-preference": "s14-union-s8-contrastive",  # 3/3 (100%) on failure set
    "temporal-reasoning": "s14-union-s8-contrastive",  # 9/12 (75%) on failure set
}

# Fallback (when qtype unknown)
SCAFFOLD_DISPATCHER_QWEN["_default"] = "s14-union-s8-contrastive"
```

## 4. Compositions That Compose

### s13-union-s8-korean — 14/20 (70%)

- s8-socratic alone: 12/20 (60%)
- korean-evidential-commit alone: 3/20
- union: 14/20 (70%)
- knowledge-update wins: 89941a93
- multi-session wins: c4a1ceb8
- single-session-assistant wins: 18dcd5a5
- single-session-preference wins: 06f04340, d24813b1, 1c0ddc50
- temporal-reasoning wins: gpt4_fa19884c, gpt4_b5700ca9, 4dfccbf7

### s14-union-s8-contrastive — 15/20 (75%)

- s8-socratic alone: 12/20 (60%)
- contrastive-explicit alone: 10/20
- union: 15/20 (75%)
- knowledge-update wins: 89941a93
- multi-session wins: c4a1ceb8
- single-session-assistant wins: 18dcd5a5
- single-session-preference wins: 06f04340, d24813b1, 1c0ddc50
- temporal-reasoning wins: gpt4_fa19884c, gpt4_b5700ca9, gpt4_1916e0ea

### s15-scaffold-chain — 9/20 (45%)

- scaffold-chain (socratic filter → self-consistency pick): 9/20 (45%)
- knowledge-update wins: dad224aa
- multi-session wins: c4a1ceb8
- single-session-preference wins: 06f04340, d24813b1, 1c0ddc50
- temporal-reasoning wins: gpt4_61e13b3c, gpt4_fa19884d

## 5. Qwen vs Gemma Dispatcher Comparison

| qtype | Gemma prefers | Qwen prefers | Gemma rate | Qwen rate |
|-------|--------------|--------------|------------|-----------|
| knowledge-update | s0-minimal-baseline | s15-scaffold-chain | 55% | 100% |
| multi-session | s4-declarative-preference | s14-union-s8-contrastive | 65% | 50% |
| single-session-assistant | s1-temporal-sysprefix | s14-union-s8-contrastive | 55% | 100% |
| single-session-preference | s4-declarative-preference | s14-union-s8-contrastive | 65% | 100% |
| temporal-reasoning | s2-temporal-inline | s14-union-s8-contrastive | 70% | 75% |

**Key insight:** Gemma preferred temporal/declarative scaffolds. Qwen prefers socratic/step-based scaffolds.

## 6. Meta-Scaffold Verdict

**META-SCAFFOLD(S) FOUND** — beats or matches s8-socratic (60%) on all qtypes:

- **s14-union-s8-contrastive**: 15/20 (75%)
- **s8v3-socratic-with-dates**: 14/20 (70%)
- **s13-union-s8-korean**: 14/20 (70%)

These are candidates for single-scaffold deployment (no dispatcher needed).

## 7. Surprise Findings

- **anti-scaffold-contrarian** (WORSE): 1/20 (5%, -55% vs s8-socratic 60%)
- **korean-evidential-commit** (WORSE): 3/20 (15%, -45% vs s8-socratic 60%)
- **self-consistency** (WORSE): 4/20 (20%, -40% vs s8-socratic 60%)
- **multi-perspective-vote** (WORSE): 6/20 (30%, -30% vs s8-socratic 60%)
- **bayesian-prior** (WORSE): 8/20 (40%, -20% vs s8-socratic 60%)
- **s14-union-s8-contrastive** (BETTER): 15/20 (75%, +15% vs s8-socratic 60%)
- **aggregation-frame** (WORSE): 9/20 (45%, -15% vs s8-socratic 60%)
- **qwen-chinese-instruction** (WORSE): 9/20 (45%, -15% vs s8-socratic 60%)

### Chinese instruction scaffold note
- qwen-chinese-instruction: 9/20 (45%)
  Chinese instruction does NOT outperform English scaffolds — keep English for qwen-turbo.

## Appendix: Budget

- Total qwen-turbo calls: **358**
- Estimated cost (qwen-turbo ~$0.0005/1K tokens, ~200 tokens/call avg): ~$0.036
- Parse failures: 23
- API errors: 0
