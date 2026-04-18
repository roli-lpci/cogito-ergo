# Scaffold Transfer Validation — gemma3:4b (local) → qwen-turbo (cloud)

**Date:** 2026-04-17  
**Hypothesis:** Scaffold rankings on gemma3:4b predict scaffold rankings on qwen-turbo  
**Question set:** 20 failure cases from runC-guard (s2_hit_at_1=False, s1_hit_at_5=True)  
**qwen-turbo calls:** 120  
**API errors:** 0 | **Parse failures:** 4  

## 1. Win-Rate Table — gemma3:4b vs qwen-turbo

Scaffold                       |   gemma3:4b wins |  qwen-turbo wins |  gemma rate |  qwen rate |  delta
--------------------------------------------------------------------------------------------------------
s0-minimal-baseline            |    11/20           |     7/20          |      55.0% |     35.0% | -20.0%
s2-temporal-inline             |    14/20           |     8/20          |      70.0% |     40.0% | -30.0%
s1-temporal-sysprefix          |    11/20           |     7/20          |      55.0% |     35.0% | -20.0%
s4-declarative-preference      |    13/20           |     8/20          |      65.0% |     40.0% | -25.0%
s8-socratic                    |     8/20           |    12/20          |      40.0% |     60.0% | +20.0%
empty-format-only              |    12/20           |    10/20          |      60.0% |     50.0% | -10.0%

### 1b. Win-rate per qtype (qwen-turbo)

Scaffold                     | knowledge-update         | multi-session            | single-session-assistant | single-session-preference | temporal-reasoning       | Total
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
s0-minimal-baseline          | 2/2                      | 1/2                      | 0/1                      | 1/3                      | 3/12                     | 7/20
s2-temporal-inline           | 0/2                      | 1/2                      | 0/1                      | 1/3                      | 6/12                     | 8/20
s1-temporal-sysprefix        | 1/2                      | 1/2                      | 1/1                      | 1/3                      | 3/12                     | 7/20
s4-declarative-preference    | 0/2                      | 1/2                      | 0/1                      | 1/3                      | 6/12                     | 8/20
s8-socratic                  | 1/2                      | 1/2                      | 1/1                      | 3/3                      | 6/12                     | 12/20
empty-format-only            | 1/2                      | 1/2                      | 0/1                      | 2/3                      | 6/12                     | 10/20

## 2. Pearson Correlation (gemma3:4b → qwen-turbo)

**Pearson r = -0.5886**

**Verdict:** TRANSFER FAILS (r < 0.4) — local scaffold findings do not predict cloud model behavior.

Data points:
- s0-minimal-baseline: gemma=0.55 qwen=0.35
- s2-temporal-inline: gemma=0.70 qwen=0.40
- s1-temporal-sysprefix: gemma=0.55 qwen=0.35
- s4-declarative-preference: gemma=0.65 qwen=0.40
- s8-socratic: gemma=0.40 qwen=0.60
- empty-format-only: gemma=0.60 qwen=0.50

## 3. Scaffold Ranking Agreement (Spearman correlation)

**Spearman rho = -0.0857**

### gemma3:4b ranking (by win rate):
  1. s2-temporal-inline (70%)
  2. s4-declarative-preference (65%)
  3. empty-format-only (60%)
  4. s0-minimal-baseline (55%)
  5. s1-temporal-sysprefix (55%)
  6. s8-socratic (40%)

### qwen-turbo ranking (by win rate):
  1. s8-socratic (60%)
  2. empty-format-only (50%)
  3. s2-temporal-inline (40%)
  4. s4-declarative-preference (40%)
  5. s0-minimal-baseline (35%)
  6. s1-temporal-sysprefix (35%)

### Rank position deltas:
- s0-minimal-baseline: gemma=#4 qwen=#5 (+1)
- s2-temporal-inline: gemma=#1 qwen=#3 (+2)
- s1-temporal-sysprefix: gemma=#5 qwen=#6 (+1)
- s4-declarative-preference: gemma=#2 qwen=#4 (+2)
- s8-socratic: gemma=#6 qwen=#1 (-5)
- empty-format-only: gemma=#3 qwen=#2 (-1)

## 4. Per-Scaffold Divergence Cases

For each scaffold, questions where qwen-turbo's outcome is known (win/lose).
Divergence = gemma win → qwen loss or vice versa (approximated from win totals).

| Scaffold | qid | gemma | qwen | direction |
|----------|-----|-------|------|-----------|
| s0-minimal-baseline | 06f04340 | WIN | LOSS | gemma_only |
| s0-minimal-baseline | 1c0ddc50 | LOSS | WIN | qwen_only |
| s0-minimal-baseline | d24813b1 | WIN | LOSS | gemma_only |
| s0-minimal-baseline | gpt4_59149c78 | LOSS | WIN | qwen_only |
| s0-minimal-baseline | gpt4_61e13b3c | WIN | LOSS | gemma_only |
| s0-minimal-baseline | gpt4_e061b84f | WIN | LOSS | gemma_only |
| s0-minimal-baseline | gpt4_fa19884c | WIN | LOSS | gemma_only |
| s2-temporal-inline | 06f04340 | WIN | LOSS | gemma_only |
| s2-temporal-inline | 18dcd5a5 | WIN | LOSS | gemma_only |
| s2-temporal-inline | 1c0ddc50 | LOSS | WIN | qwen_only |
| s2-temporal-inline | 4dfccbf7 | WIN | LOSS | gemma_only |
| s2-temporal-inline | 89941a93 | WIN | LOSS | gemma_only |
| s2-temporal-inline | d24813b1 | WIN | LOSS | gemma_only |
| s2-temporal-inline | dad224aa | WIN | LOSS | gemma_only |
| s2-temporal-inline | gpt4_59149c78 | LOSS | WIN | qwen_only |
| s2-temporal-inline | gpt4_e061b84f | LOSS | WIN | qwen_only |
| s1-temporal-sysprefix | 06f04340 | WIN | LOSS | gemma_only |
| s1-temporal-sysprefix | d24813b1 | WIN | LOSS | gemma_only |
| s1-temporal-sysprefix | gpt4_fa19884d | WIN | LOSS | gemma_only |
| s4-declarative-preference | 06f04340 | WIN | LOSS | gemma_only |
| s4-declarative-preference | d24813b1 | WIN | LOSS | gemma_only |
| s4-declarative-preference | dad224aa | WIN | LOSS | gemma_only |
| s4-declarative-preference | gpt4_59149c78 | LOSS | WIN | qwen_only |
| s4-declarative-preference | gpt4_61e13b3c | LOSS | WIN | qwen_only |
| s4-declarative-preference | gpt4_e061b84f | WIN | LOSS | gemma_only |
| s4-declarative-preference | gpt4_fa19884d | LOSS | WIN | qwen_only |
| s8-socratic | 06f04340 | LOSS | WIN | qwen_only |
| s8-socratic | 18dcd5a5 | LOSS | WIN | qwen_only |
| s8-socratic | 1c0ddc50 | LOSS | WIN | qwen_only |
| s8-socratic | 4dfccbf7 | LOSS | WIN | qwen_only |
| s8-socratic | gpt4_59149c78 | WIN | LOSS | gemma_only |
| s8-socratic | gpt4_61e13b3c | LOSS | WIN | qwen_only |
| s8-socratic | gpt4_b5700ca9 | LOSS | WIN | qwen_only |
| s8-socratic | gpt4_e061b84f | WIN | LOSS | gemma_only |
| empty-format-only | d24813b1 | WIN | LOSS | gemma_only |
| empty-format-only | dad224aa | WIN | LOSS | gemma_only |
| empty-format-only | gpt4_59149c78 | LOSS | WIN | qwen_only |

### Aggregate divergence by scaffold:
- **s0-minimal-baseline**: gemma=55% qwen=35% Δ=20% (gemma_higher)
- **s2-temporal-inline**: gemma=70% qwen=40% Δ=30% (gemma_higher)
- **s1-temporal-sysprefix**: gemma=55% qwen=35% Δ=20% (gemma_higher)
- **s4-declarative-preference**: gemma=65% qwen=40% Δ=25% (gemma_higher)
- **s8-socratic**: gemma=40% qwen=60% Δ=20% (qwen_higher)
- **empty-format-only**: gemma=60% qwen=50% Δ=10% (gemma_higher)

## 5. Methodology Verdict

**Transfer FAILS (r=-0.589): gemma3:4b scaffold findings do not predict qwen-turbo behavior. The two models respond to prompt scaffolds in structurally different ways.**

**Action:** All scaffold research must be run directly on the target model tier. Local model serves only as a parse/format sanity check, not a performance proxy.

## 6. Notable Surprises

- **s0-minimal-baseline**: much worse on qwen-turbo than gemma3:4b (gemma=55% → qwen=35%, Δ=-20%)
- **s2-temporal-inline**: much worse on qwen-turbo than gemma3:4b (gemma=70% → qwen=40%, Δ=-30%)
- **s1-temporal-sysprefix**: much worse on qwen-turbo than gemma3:4b (gemma=55% → qwen=35%, Δ=-20%)
- **s4-declarative-preference**: much worse on qwen-turbo than gemma3:4b (gemma=65% → qwen=40%, Δ=-25%)
- **s8-socratic**: much better on qwen-turbo than gemma3:4b (gemma=40% → qwen=60%, Δ=+20%)

## 7. Raw Correlation Data

```
Scaffold                           gemma_rate    qwen_rate
----------------------------------------------------------
s0-minimal-baseline                    0.5500       0.3500
s2-temporal-inline                     0.7000       0.4000
s1-temporal-sysprefix                  0.5500       0.3500
s4-declarative-preference              0.6500       0.4000
s8-socratic                            0.4000       0.6000
empty-format-only                      0.6000       0.5000
Pearson r                                          -0.5886
Spearman rho                                       -0.0857
```
