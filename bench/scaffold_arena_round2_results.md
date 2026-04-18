# Scaffold Arena Round 2 Results

**Date:** 2026-04-16  
**Baseline run:** runC-guard  
**Model:** gemma3:4b (local, $0)  
**Total LLM calls:** 200  
**API errors:** 0 | **Parse failures:** 3  
**Set A size:** 20 questions (gold in top-5, wrong @1)  
**Round 1 best:** s2-temporal-inline at 14/20 (+3 vs baseline)  

## 1. Win Rate Table (Round 2 vs Round 1 Best)

Scaffold                         | knowledge-update     | multi-session        | single-session-assistant | single-session-preference | temporal-reasoning   | Total | vs R1-best
-------------------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | ----- | ----------
r2s1-composition-meta            | 1/2                  | 1/2                  | 1/1                  | 3/3                  | 6/12                 | 12/20 | -2
r2s2-json-schema                 | 1/2                  | 1/2                  | 1/1                  | 3/3                  | 6/12                 | 12/20 | -2
r2s3-null-option                 | 1/2                  | 1/2                  | 1/1                  | 3/3                  | 6/12                 | 12/20 | -2
r2s4-chain-of-verification       | 2/2                  | 1/2                  | 1/1                  | 3/3                  | 5/12                 | 12/20 | -2
r2s5-contradiction-aware         | 2/2                  | 1/2                  | 1/1                  | 2/3                  | 5/12                 | 11/20 | -3
r2s6-temporal-v2                 | 2/2                  | 1/2                  | 1/1                  | 2/3                  | 7/12                 | 13/20 | -1
r2s7-s2plus7-union               | 2/2                  | 1/2                  | 1/1                  | 3/3                  | 7/12                 | 14/20 | +0
r2s8-preference-v2               | 2/2                  | 1/2                  | 0/1                  | 3/3                  | 8/12                 | 14/20 | +0
r2s9-empty-sanity                | 2/2                  | 1/2                  | 0/1                  | 3/3                  | 6/12                 | 12/20 | -2

## 2. Meta-Scaffold Verdict

**No meta-scaffold found.** Dispatcher architecture is the ceiling.

Per-scaffold category breakdown vs R1 dispatcher:
  r2s1-composition-meta: falls short on [knowledge-update: 1 vs R1=2, temporal-reasoning: 6 vs R1=8]
  r2s2-json-schema: falls short on [knowledge-update: 1 vs R1=2, temporal-reasoning: 6 vs R1=8]
  r2s3-null-option: falls short on [knowledge-update: 1 vs R1=2, temporal-reasoning: 6 vs R1=8]
  r2s4-chain-of-verification: falls short on [temporal-reasoning: 5 vs R1=8]
  r2s5-contradiction-aware: falls short on [single-session-preference: 2 vs R1=3, temporal-reasoning: 5 vs R1=8]
  r2s6-temporal-v2: falls short on [single-session-preference: 2 vs R1=3, temporal-reasoning: 7 vs R1=8]
  r2s7-s2plus7-union: falls short on [temporal-reasoning: 7 vs R1=8]
  r2s8-preference-v2: falls short on [single-session-assistant: 0 vs R1=1]
  r2s9-empty-sanity: falls short on [single-session-assistant: 0 vs R1=1, temporal-reasoning: 6 vs R1=8]

## 3. Updated Dispatcher Recommendation

No round 2 scaffold beats round 1 per-category bests. Round 1 dispatcher stands unchanged.

```python
SCAFFOLD_DISPATCHER = {
    # R1 winners unchanged — no R2 scaffold beats them per category
    "knowledge-update": "s2-temporal-inline",      # 2/2 (s2, s3, s7, s10 all tie at 2/2 — s2 chosen for temporal consistency)
    "multi-session": "s0-minimal-baseline",        # 1/2 (ceiling hit — 6d550036 is retrieval-layer failure)
    "single-session-assistant": "s1-temporal-sysprefix",  # 1/1 (s2s8-preference-v2 misses this category)
    "single-session-preference": "s1-temporal-sysprefix", # 3/3 (s2s8 also 3/3 but misses single-session-assistant)
    "temporal-reasoning": "s2-temporal-inline",    # 8/12 (R2 best: r2s8-preference-v2 at 8/12, same score)
}
```

R2 finding: r2s8-preference-v2 matches s2-temporal-inline on temporal (8/12) and improves knowledge-update (2/2 vs 1/2 for some R1 scaffolds), but drops single-session-assistant (0/1). It is NOT a drop-in replacement.

**R2 actionable upgrade:** For temporal-reasoning, r2s8-preference-v2 and r2s7-s2plus7-union both hit 7-8/12. r2s8's temporal wins differ from s2's in one case (4dfccbf7 — ukulele/guitar), making it a candidate for a 3-way union with s2+s7+s8 to potentially hit 9/12.

## 4. Hard Case Update

| qid | qtype | Round 1 result | Round 2 results |
|-----|-------|----------------|-----------------|
| 9a707b82 | temporal-reasoning | FAIL all R1 | Still fails all R2 scaffolds. Rankings: {'r2s1-composition-meta': 3, 'r2s2-json-schema': 3, 'r2s3-null-option': 3, 'r2s4-chain-of-verification': 5, 'r2s5-contradiction-aware': 3, 'r2s6-temporal-v2': 3, 'r2s7-s2plus7-union': 3, 'r2s8-preference-v2': 1, 'r2s9-empty-sanity': 3} |
| 6d550036 | multi-session | FAIL all R1 | Still fails all R2 scaffolds. Rankings: {'r2s1-composition-meta': 1, 'r2s2-json-schema': 1, 'r2s3-null-option': 1, 'r2s4-chain-of-verification': 1, 'r2s5-contradiction-aware': 1, 'r2s6-temporal-v2': 1, 'r2s7-s2plus7-union': 1, 'r2s8-preference-v2': 1, 'r2s9-empty-sanity': 1} |

## 5. Composition Validation: r2s7 (s2+s7 union) on temporal-reasoning

r2s7 union rank: **7/12** on temporal-reasoning
Round 1 analysis predicted: 10/12
Round 1 s2-alone: 8/12
**NOT CONFIRMED**: union does NOT improve on s2-alone, and falls below it.

Root cause: The 3 s7-only wins in round 1 (qids `gpt4_b5700ca9`, `gpt4_1916e0ea`, `5e1b23de`) were NOT reproduced in round 2. Those wins were stochastic, not stable behavior. The union got 7/12 vs s2's round 1 score of 8/12 — the composition prediction of 10/12 was based on non-repeatable signal.

## 6. Surprises & Notable Findings

**r2s2-json-schema**: 12/20 total wins.
  JSON schema forcing is competitive but does not dominate.

**r2s9-empty-sanity**: 12/20 total wins.
  Surprising: empty scaffold performs at or above baseline — suggests gemma3:4b has strong priors on format alone.

**r2s3-null-option**: 12/20 total wins.

## 7. Per-Scaffold Win Details

### r2s1-composition-meta — 12/20 (-2 vs R1 best)

**knowledge-update wins:**
- qid=dad224aa: gold@[2, 3] → ranked [2]
  Q: What time do I wake up on Saturday mornings?

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → ranked [4]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-assistant wins:**
- qid=18dcd5a5: gold@[1] → ranked [1]
  Q: I'm going back to our previous chat about the Lost Temple of the Djinn one-shot. Can you remind me h

**single-session-preference wins:**
- qid=06f04340: gold@[4] → ranked [4]
  Q: What should I serve for dinner this weekend with my homegrown ingredients?
- qid=d24813b1: gold@[4] → ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?
- qid=1c0ddc50: gold@[4] → ranked [4]
  Q: Can you suggest some activities I can do during my commute to work?

**temporal-reasoning wins:**
- qid=gpt4_61e13b3c: gold@[1, 2] → ranked [2]
  Q: How many weeks passed between the time I sold homemade baked goods at the Farmers' Market for the la
- qid=gpt4_e061b84f: gold@[3] → ranked [3]
  Q: What is the order of the three sports events I participated in during the past month, from earliest 
- qid=gpt4_59149c78: gold@[3] → ranked [3]
  Q: I mentioned that I participated in an art-related event two weeks ago. Where was that event held at?

### r2s2-json-schema — 12/20 (-2 vs R1 best)

**knowledge-update wins:**
- qid=dad224aa: gold@[2, 3] → ranked [2]
  Q: What time do I wake up on Saturday mornings?

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → ranked [4]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-assistant wins:**
- qid=18dcd5a5: gold@[1] → ranked [1]
  Q: I'm going back to our previous chat about the Lost Temple of the Djinn one-shot. Can you remind me h

**single-session-preference wins:**
- qid=06f04340: gold@[4] → ranked [4]
  Q: What should I serve for dinner this weekend with my homegrown ingredients?
- qid=d24813b1: gold@[4] → ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?
- qid=1c0ddc50: gold@[4] → ranked [4]
  Q: Can you suggest some activities I can do during my commute to work?

**temporal-reasoning wins:**
- qid=gpt4_fa19884c: gold@[1, 2] → ranked [1]
  Q: How many days passed between the day I started playing along to my favorite songs on my old keyboard
- qid=gpt4_61e13b3c: gold@[1, 2] → ranked [2]
  Q: How many weeks passed between the time I sold homemade baked goods at the Farmers' Market for the la
- qid=gpt4_e061b84f: gold@[3] → ranked [3]
  Q: What is the order of the three sports events I participated in during the past month, from earliest 

### r2s3-null-option — 12/20 (-2 vs R1 best)

**knowledge-update wins:**
- qid=dad224aa: gold@[2, 3] → ranked [2]
  Q: What time do I wake up on Saturday mornings?

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → ranked [2]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-assistant wins:**
- qid=18dcd5a5: gold@[1] → ranked [1]
  Q: I'm going back to our previous chat about the Lost Temple of the Djinn one-shot. Can you remind me h

**single-session-preference wins:**
- qid=06f04340: gold@[4] → ranked [4]
  Q: What should I serve for dinner this weekend with my homegrown ingredients?
- qid=d24813b1: gold@[4] → ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?
- qid=1c0ddc50: gold@[4] → ranked [4]
  Q: Can you suggest some activities I can do during my commute to work?

**temporal-reasoning wins:**
- qid=gpt4_fa19884c: gold@[1, 2] → ranked [2]
  Q: How many days passed between the day I started playing along to my favorite songs on my old keyboard
- qid=gpt4_61e13b3c: gold@[1, 2] → ranked [2]
  Q: How many weeks passed between the time I sold homemade baked goods at the Farmers' Market for the la
- qid=gpt4_e061b84f: gold@[3] → ranked [3]
  Q: What is the order of the three sports events I participated in during the past month, from earliest 

### r2s4-chain-of-verification — 12/20 (-2 vs R1 best)

**knowledge-update wins:**
- qid=89941a93: gold@[1, 2] → ranked [2]
  Q: How many bikes do I currently own?
- qid=dad224aa: gold@[2, 3] → ranked [3]
  Q: What time do I wake up on Saturday mornings?

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → ranked [4]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-assistant wins:**
- qid=18dcd5a5: gold@[1] → ranked [1]
  Q: I'm going back to our previous chat about the Lost Temple of the Djinn one-shot. Can you remind me h

**single-session-preference wins:**
- qid=06f04340: gold@[4] → ranked [4]
  Q: What should I serve for dinner this weekend with my homegrown ingredients?
- qid=d24813b1: gold@[4] → ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?
- qid=1c0ddc50: gold@[4] → ranked [4]
  Q: Can you suggest some activities I can do during my commute to work?

**temporal-reasoning wins:**
- qid=gpt4_e061b84f: gold@[3] → ranked [3]
  Q: What is the order of the three sports events I participated in during the past month, from earliest 
- qid=gpt4_59149c78: gold@[3] → ranked [3]
  Q: I mentioned that I participated in an art-related event two weeks ago. Where was that event held at?
- qid=gpt4_fa19884d: gold@[2, 4] → ranked [4]
  Q: What is the artist that I started to listen to last Friday?

### r2s5-contradiction-aware — 11/20 (-3 vs R1 best)

**knowledge-update wins:**
- qid=89941a93: gold@[1, 2] → ranked [2]
  Q: How many bikes do I currently own?
- qid=dad224aa: gold@[2, 3] → ranked [2]
  Q: What time do I wake up on Saturday mornings?

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → ranked [3]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-assistant wins:**
- qid=18dcd5a5: gold@[1] → ranked [1]
  Q: I'm going back to our previous chat about the Lost Temple of the Djinn one-shot. Can you remind me h

**single-session-preference wins:**
- qid=06f04340: gold@[4] → ranked [4]
  Q: What should I serve for dinner this weekend with my homegrown ingredients?
- qid=d24813b1: gold@[4] → ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?

**temporal-reasoning wins:**
- qid=gpt4_61e13b3c: gold@[1, 2] → ranked [2]
  Q: How many weeks passed between the time I sold homemade baked goods at the Farmers' Market for the la
- qid=gpt4_e061b84f: gold@[3] → ranked [3]
  Q: What is the order of the three sports events I participated in during the past month, from earliest 
- qid=gpt4_59149c78: gold@[3] → ranked [3]
  Q: I mentioned that I participated in an art-related event two weeks ago. Where was that event held at?

### r2s6-temporal-v2 — 13/20 (-1 vs R1 best)

**knowledge-update wins:**
- qid=89941a93: gold@[1, 2] → ranked [2]
  Q: How many bikes do I currently own?
- qid=dad224aa: gold@[2, 3] → ranked [3]
  Q: What time do I wake up on Saturday mornings?

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → ranked [1]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-assistant wins:**
- qid=18dcd5a5: gold@[1] → ranked [1]
  Q: I'm going back to our previous chat about the Lost Temple of the Djinn one-shot. Can you remind me h

**single-session-preference wins:**
- qid=06f04340: gold@[4] → ranked [4]
  Q: What should I serve for dinner this weekend with my homegrown ingredients?
- qid=d24813b1: gold@[4] → ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?

**temporal-reasoning wins:**
- qid=gpt4_fa19884c: gold@[1, 2] → ranked [1]
  Q: How many days passed between the day I started playing along to my favorite songs on my old keyboard
- qid=4dfccbf7: gold@[1, 2] → ranked [2]
  Q: How many days had passed since I started taking ukulele lessons when I decided to take my acoustic g
- qid=gpt4_61e13b3c: gold@[1, 2] → ranked [2]
  Q: How many weeks passed between the time I sold homemade baked goods at the Farmers' Market for the la

### r2s7-s2plus7-union — 14/20 (+0 vs R1 best)

**knowledge-update wins:**
- qid=89941a93: gold@[1, 2] → ranked [2]
  Q: How many bikes do I currently own?
- qid=dad224aa: gold@[2, 3] → ranked [3]
  Q: What time do I wake up on Saturday mornings?

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → ranked [3]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-assistant wins:**
- qid=18dcd5a5: gold@[1] → ranked [1]
  Q: I'm going back to our previous chat about the Lost Temple of the Djinn one-shot. Can you remind me h

**single-session-preference wins:**
- qid=06f04340: gold@[4] → ranked [4]
  Q: What should I serve for dinner this weekend with my homegrown ingredients?
- qid=d24813b1: gold@[4] → ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?
- qid=1c0ddc50: gold@[4] → ranked [4]
  Q: Can you suggest some activities I can do during my commute to work?

**temporal-reasoning wins:**
- qid=gpt4_fa19884c: gold@[1, 2] → ranked [1]
  Q: How many days passed between the day I started playing along to my favorite songs on my old keyboard
- qid=gpt4_61e13b3c: gold@[1, 2] → ranked [2]
  Q: How many weeks passed between the time I sold homemade baked goods at the Farmers' Market for the la
- qid=gpt4_e061b84f: gold@[3] → ranked [3]
  Q: What is the order of the three sports events I participated in during the past month, from earliest 

### r2s8-preference-v2 — 14/20 (+0 vs R1 best)

**knowledge-update wins:**
- qid=89941a93: gold@[1, 2] → ranked [2]
  Q: How many bikes do I currently own?
- qid=dad224aa: gold@[2, 3] → ranked [2]
  Q: What time do I wake up on Saturday mornings?

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → ranked [2]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-preference wins:**
- qid=06f04340: gold@[4] → ranked [4]
  Q: What should I serve for dinner this weekend with my homegrown ingredients?
- qid=d24813b1: gold@[4] → ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?
- qid=1c0ddc50: gold@[4] → ranked [4]
  Q: Can you suggest some activities I can do during my commute to work?

**temporal-reasoning wins:**
- qid=gpt4_fa19884c: gold@[1, 2] → ranked [1]
  Q: How many days passed between the day I started playing along to my favorite songs on my old keyboard
- qid=4dfccbf7: gold@[1, 2] → ranked [2]
  Q: How many days had passed since I started taking ukulele lessons when I decided to take my acoustic g
- qid=gpt4_61e13b3c: gold@[1, 2] → ranked [1]
  Q: How many weeks passed between the time I sold homemade baked goods at the Farmers' Market for the la

### r2s9-empty-sanity — 12/20 (-2 vs R1 best)

**knowledge-update wins:**
- qid=89941a93: gold@[1, 2] → ranked [2]
  Q: How many bikes do I currently own?
- qid=dad224aa: gold@[2, 3] → ranked [3]
  Q: What time do I wake up on Saturday mornings?

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → ranked [2]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-preference wins:**
- qid=06f04340: gold@[4] → ranked [4]
  Q: What should I serve for dinner this weekend with my homegrown ingredients?
- qid=d24813b1: gold@[4] → ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?
- qid=1c0ddc50: gold@[4] → ranked [4]
  Q: Can you suggest some activities I can do during my commute to work?

**temporal-reasoning wins:**
- qid=gpt4_fa19884c: gold@[1, 2] → ranked [1]
  Q: How many days passed between the day I started playing along to my favorite songs on my old keyboard
- qid=gpt4_61e13b3c: gold@[1, 2] → ranked [2]
  Q: How many weeks passed between the time I sold homemade baked goods at the Farmers' Market for the la
- qid=gpt4_e061b84f: gold@[3] → ranked [3]
  Q: What is the order of the three sports events I participated in during the past month, from earliest 
