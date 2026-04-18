# Scaffold Arena Results

**Date:** 2026-04-16  
**Baseline run:** runC-guard  
**Model:** gemma3:4b (local, $0)  
**Total LLM calls:** 220  
**API errors:** 0 | **Parse failures:** 1  
**Set A size:** 20 questions (gold in top-5, wrong @1)  

## 1. Win Rate Per Scaffold Per qtype

Scaffold                         | knowledge-update     | multi-session        | single-session-assistant | single-session-preference | temporal-reasoning   | Total
-------------------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -----
s0-minimal-baseline              | 2/2                  | 1/2                  | 0/1                  | 2/3                  | 6/12                 | 11/20
s1-temporal-sysprefix            | 1/2                  | 1/2                  | 1/1                  | 3/3                  | 5/12                 | 11/20
s2-temporal-inline               | 2/2                  | 1/2                  | 1/1                  | 2/3                  | 8/12                 | 14/20
s3-contrastive-not               | 2/2                  | 1/2                  | 1/1                  | 3/3                  | 6/12                 | 13/20
s4-declarative-preference        | 1/2                  | 1/2                  | 1/1                  | 3/3                  | 7/12                 | 13/20
s5-multi-session-aggregation     | 1/2                  | 1/2                  | 1/1                  | 3/3                  | 7/12                 | 13/20
s6-code-scaffold                 | 0/2                  | 1/2                  | 1/1                  | 3/3                  | 6/12                 | 11/20
s7-chain-of-thought              | 2/2                  | 1/2                  | 1/1                  | 1/3                  | 6/12                 | 11/20
s8-socratic                      | 1/2                  | 1/2                  | 0/1                  | 1/3                  | 5/12                 | 8/20
s9-hypothesis                    | 0/2                  | 1/2                  | 1/1                  | 2/3                  | 7/12                 | 11/20
s10-anti-confidence              | 2/2                  | 1/2                  | 1/1                  | 2/3                  | 6/12                 | 12/20

## 2. Specific Wins vs Baseline

**Baseline (s0-minimal) total wins: 11/20**

### s1-temporal-sysprefix — 11/20 (+0 vs baseline)

**knowledge-update wins:**
- qid=dad224aa: gold@[2, 3] → scaffold ranked [3]
  Q: What time do I wake up on Saturday mornings?

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → scaffold ranked [4]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-assistant wins:**
- qid=18dcd5a5: gold@[1] → scaffold ranked [1]
  Q: I'm going back to our previous chat about the Lost Temple of the Djinn one-shot. Can you remind me h

**single-session-preference wins:**
- qid=06f04340: gold@[4] → scaffold ranked [4]
  Q: What should I serve for dinner this weekend with my homegrown ingredients?
- qid=d24813b1: gold@[4] → scaffold ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?
- qid=1c0ddc50: gold@[4] → scaffold ranked [4]
  Q: Can you suggest some activities I can do during my commute to work?

**temporal-reasoning wins:**
- qid=gpt4_61e13b3c: gold@[1, 2] → scaffold ranked [2]
  Q: How many weeks passed between the time I sold homemade baked goods at the Farmers' Market for the la
- qid=gpt4_e061b84f: gold@[3] → scaffold ranked [3]
  Q: What is the order of the three sports events I participated in during the past month, from earliest 
- qid=gpt4_fa19884d: gold@[2, 4] → scaffold ranked [4]
  Q: What is the artist that I started to listen to last Friday?

### s2-temporal-inline — 14/20 (+3 vs baseline)

**knowledge-update wins:**
- qid=89941a93: gold@[1, 2] → scaffold ranked [2]
  Q: How many bikes do I currently own?
- qid=dad224aa: gold@[2, 3] → scaffold ranked [3]
  Q: What time do I wake up on Saturday mornings?

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → scaffold ranked [3]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-assistant wins:**
- qid=18dcd5a5: gold@[1] → scaffold ranked [1]
  Q: I'm going back to our previous chat about the Lost Temple of the Djinn one-shot. Can you remind me h

**single-session-preference wins:**
- qid=06f04340: gold@[4] → scaffold ranked [4]
  Q: What should I serve for dinner this weekend with my homegrown ingredients?
- qid=d24813b1: gold@[4] → scaffold ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?

**temporal-reasoning wins:**
- qid=gpt4_fa19884c: gold@[1, 2] → scaffold ranked [1]
  Q: How many days passed between the day I started playing along to my favorite songs on my old keyboard
- qid=4dfccbf7: gold@[1, 2] → scaffold ranked [2]
  Q: How many days had passed since I started taking ukulele lessons when I decided to take my acoustic g
- qid=gpt4_61e13b3c: gold@[1, 2] → scaffold ranked [2]
  Q: How many weeks passed between the time I sold homemade baked goods at the Farmers' Market for the la

### s3-contrastive-not — 13/20 (+2 vs baseline)

**knowledge-update wins:**
- qid=89941a93: gold@[1, 2] → scaffold ranked [2]
  Q: How many bikes do I currently own?
- qid=dad224aa: gold@[2, 3] → scaffold ranked [2]
  Q: What time do I wake up on Saturday mornings?

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → scaffold ranked [4]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-assistant wins:**
- qid=18dcd5a5: gold@[1] → scaffold ranked [1]
  Q: I'm going back to our previous chat about the Lost Temple of the Djinn one-shot. Can you remind me h

**single-session-preference wins:**
- qid=06f04340: gold@[4] → scaffold ranked [4]
  Q: What should I serve for dinner this weekend with my homegrown ingredients?
- qid=d24813b1: gold@[4] → scaffold ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?
- qid=1c0ddc50: gold@[4] → scaffold ranked [4]
  Q: Can you suggest some activities I can do during my commute to work?

**temporal-reasoning wins:**
- qid=gpt4_fa19884c: gold@[1, 2] → scaffold ranked [1]
  Q: How many days passed between the day I started playing along to my favorite songs on my old keyboard
- qid=gpt4_61e13b3c: gold@[1, 2] → scaffold ranked [2]
  Q: How many weeks passed between the time I sold homemade baked goods at the Farmers' Market for the la
- qid=gpt4_e061b84f: gold@[3] → scaffold ranked [3]
  Q: What is the order of the three sports events I participated in during the past month, from earliest 

### s4-declarative-preference — 13/20 (+2 vs baseline)

**knowledge-update wins:**
- qid=dad224aa: gold@[2, 3] → scaffold ranked [3]
  Q: What time do I wake up on Saturday mornings?

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → scaffold ranked [3]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-assistant wins:**
- qid=18dcd5a5: gold@[1] → scaffold ranked [1]
  Q: I'm going back to our previous chat about the Lost Temple of the Djinn one-shot. Can you remind me h

**single-session-preference wins:**
- qid=06f04340: gold@[4] → scaffold ranked [4]
  Q: What should I serve for dinner this weekend with my homegrown ingredients?
- qid=d24813b1: gold@[4] → scaffold ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?
- qid=1c0ddc50: gold@[4] → scaffold ranked [4]
  Q: Can you suggest some activities I can do during my commute to work?

**temporal-reasoning wins:**
- qid=gpt4_fa19884c: gold@[1, 2] → scaffold ranked [1]
  Q: How many days passed between the day I started playing along to my favorite songs on my old keyboard
- qid=4dfccbf7: gold@[1, 2] → scaffold ranked [2]
  Q: How many days had passed since I started taking ukulele lessons when I decided to take my acoustic g
- qid=gpt4_e061b84f: gold@[3] → scaffold ranked [3]
  Q: What is the order of the three sports events I participated in during the past month, from earliest 

### s5-multi-session-aggregation — 13/20 (+2 vs baseline)

**knowledge-update wins:**
- qid=89941a93: gold@[1, 2] → scaffold ranked [2]
  Q: How many bikes do I currently own?

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → scaffold ranked [3]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-assistant wins:**
- qid=18dcd5a5: gold@[1] → scaffold ranked [1]
  Q: I'm going back to our previous chat about the Lost Temple of the Djinn one-shot. Can you remind me h

**single-session-preference wins:**
- qid=06f04340: gold@[4] → scaffold ranked [4]
  Q: What should I serve for dinner this weekend with my homegrown ingredients?
- qid=d24813b1: gold@[4] → scaffold ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?
- qid=1c0ddc50: gold@[4] → scaffold ranked [4]
  Q: Can you suggest some activities I can do during my commute to work?

**temporal-reasoning wins:**
- qid=gpt4_fa19884c: gold@[1, 2] → scaffold ranked [1]
  Q: How many days passed between the day I started playing along to my favorite songs on my old keyboard
- qid=gpt4_61e13b3c: gold@[1, 2] → scaffold ranked [2]
  Q: How many weeks passed between the time I sold homemade baked goods at the Farmers' Market for the la
- qid=gpt4_e061b84f: gold@[3] → scaffold ranked [3]
  Q: What is the order of the three sports events I participated in during the past month, from earliest 

### s6-code-scaffold — 11/20 (+0 vs baseline)

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → scaffold ranked [2]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-assistant wins:**
- qid=18dcd5a5: gold@[1] → scaffold ranked [1]
  Q: I'm going back to our previous chat about the Lost Temple of the Djinn one-shot. Can you remind me h

**single-session-preference wins:**
- qid=06f04340: gold@[4] → scaffold ranked [4]
  Q: What should I serve for dinner this weekend with my homegrown ingredients?
- qid=d24813b1: gold@[4] → scaffold ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?
- qid=1c0ddc50: gold@[4] → scaffold ranked [4]
  Q: Can you suggest some activities I can do during my commute to work?

**temporal-reasoning wins:**
- qid=gpt4_fa19884c: gold@[1, 2] → scaffold ranked [1]
  Q: How many days passed between the day I started playing along to my favorite songs on my old keyboard
- qid=gpt4_e061b84f: gold@[3] → scaffold ranked [3]
  Q: What is the order of the three sports events I participated in during the past month, from earliest 
- qid=gpt4_59149c78: gold@[3] → scaffold ranked [3]
  Q: I mentioned that I participated in an art-related event two weeks ago. Where was that event held at?

### s7-chain-of-thought — 11/20 (+0 vs baseline)

**knowledge-update wins:**
- qid=89941a93: gold@[1, 2] → scaffold ranked [2]
  Q: How many bikes do I currently own?
- qid=dad224aa: gold@[2, 3] → scaffold ranked [2]
  Q: What time do I wake up on Saturday mornings?

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → scaffold ranked [1]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-assistant wins:**
- qid=18dcd5a5: gold@[1] → scaffold ranked [1]
  Q: I'm going back to our previous chat about the Lost Temple of the Djinn one-shot. Can you remind me h

**single-session-preference wins:**
- qid=d24813b1: gold@[4] → scaffold ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?

**temporal-reasoning wins:**
- qid=gpt4_fa19884c: gold@[1, 2] → scaffold ranked [1]
  Q: How many days passed between the day I started playing along to my favorite songs on my old keyboard
- qid=gpt4_b5700ca9: gold@[1] → scaffold ranked [1]
  Q: How many days ago did I attend the Maundy Thursday service at the Episcopal Church?
- qid=gpt4_1916e0ea: gold@[1] → scaffold ranked [1]
  Q: How many days passed between the day I cancelled my FarmFresh subscription and the day I did my onli

### s8-socratic — 8/20 (-3 vs baseline)

**knowledge-update wins:**
- qid=89941a93: gold@[1, 2] → scaffold ranked [1]
  Q: How many bikes do I currently own?

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → scaffold ranked [4]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-preference wins:**
- qid=d24813b1: gold@[4] → scaffold ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?

**temporal-reasoning wins:**
- qid=gpt4_fa19884c: gold@[1, 2] → scaffold ranked [1]
  Q: How many days passed between the day I started playing along to my favorite songs on my old keyboard
- qid=gpt4_e061b84f: gold@[3] → scaffold ranked [3]
  Q: What is the order of the three sports events I participated in during the past month, from earliest 
- qid=gpt4_59149c78: gold@[3] → scaffold ranked [3]
  Q: I mentioned that I participated in an art-related event two weeks ago. Where was that event held at?

### s9-hypothesis — 11/20 (+0 vs baseline)

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → scaffold ranked [2]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-assistant wins:**
- qid=18dcd5a5: gold@[1] → scaffold ranked [1]
  Q: I'm going back to our previous chat about the Lost Temple of the Djinn one-shot. Can you remind me h

**single-session-preference wins:**
- qid=06f04340: gold@[4] → scaffold ranked [4]
  Q: What should I serve for dinner this weekend with my homegrown ingredients?
- qid=d24813b1: gold@[4] → scaffold ranked [4]
  Q: I'm thinking of inviting my colleagues over for a small gathering. Any tips on what to bake?

**temporal-reasoning wins:**
- qid=gpt4_fa19884c: gold@[1, 2] → scaffold ranked [2]
  Q: How many days passed between the day I started playing along to my favorite songs on my old keyboard
- qid=gpt4_61e13b3c: gold@[1, 2] → scaffold ranked [2]
  Q: How many weeks passed between the time I sold homemade baked goods at the Farmers' Market for the la
- qid=gpt4_e061b84f: gold@[3] → scaffold ranked [3]
  Q: What is the order of the three sports events I participated in during the past month, from earliest 

### s10-anti-confidence — 12/20 (+1 vs baseline)

**knowledge-update wins:**
- qid=89941a93: gold@[1, 2] → scaffold ranked [2]
  Q: How many bikes do I currently own?
- qid=dad224aa: gold@[2, 3] → scaffold ranked [2]
  Q: What time do I wake up on Saturday mornings?

**multi-session wins:**
- qid=c4a1ceb8: gold@[1, 2, 3, 4] → scaffold ranked [3]
  Q: How many different types of citrus fruits have I used in my cocktail recipes?

**single-session-assistant wins:**
- qid=18dcd5a5: gold@[1] → scaffold ranked [1]
  Q: I'm going back to our previous chat about the Lost Temple of the Djinn one-shot. Can you remind me h

**single-session-preference wins:**
- qid=06f04340: gold@[4] → scaffold ranked [4]
  Q: What should I serve for dinner this weekend with my homegrown ingredients?
- qid=1c0ddc50: gold@[4] → scaffold ranked [4]
  Q: Can you suggest some activities I can do during my commute to work?

**temporal-reasoning wins:**
- qid=gpt4_fa19884c: gold@[1, 2] → scaffold ranked [2]
  Q: How many days passed between the day I started playing along to my favorite songs on my old keyboard
- qid=gpt4_61e13b3c: gold@[1, 2] → scaffold ranked [2]
  Q: How many weeks passed between the time I sold homemade baked goods at the Farmers' Market for the la
- qid=gpt4_e061b84f: gold@[3] → scaffold ranked [3]
  Q: What is the order of the three sports events I participated in during the past month, from earliest 

## 3. Recommended Scaffold Dispatcher

Based on win rates, recommended scaffold per qtype:

```
qtype=knowledge-update → s0-minimal-baseline (2/2 failures fixed)
```

```
qtype=multi-session → s0-minimal-baseline (1/2 failures fixed)
```

```
qtype=single-session-assistant → s1-temporal-sysprefix (1/1 failures fixed)
```

```
qtype=single-session-preference → s1-temporal-sysprefix (3/3 failures fixed)
```

```
qtype=temporal-reasoning → s2-temporal-inline (8/12 failures fixed)
```

## 4. Meta-Scaffold Candidate

**No single scaffold beats baseline on ALL qtypes.** Dispatcher approach recommended.

## 5. Composition Candidates

Scaffolds that win on DIFFERENT subsets of questions within the same qtype can be composed (try both, take max):

**knowledge-update:**
- s0-minimal-baseline: 2/2 on qids ['89941a93', 'dad224aa']
- s2-temporal-inline: 2/2 on qids ['89941a93', 'dad224aa']
- s3-contrastive-not: 2/2 on qids ['89941a93', 'dad224aa']
- s7-chain-of-thought: 2/2 on qids ['89941a93', 'dad224aa']
- s10-anti-confidence: 2/2 on qids ['89941a93', 'dad224aa']
- s1-temporal-sysprefix: 1/2 on qids ['dad224aa']
- s4-declarative-preference: 1/2 on qids ['dad224aa']
- s5-multi-session-aggregation: 1/2 on qids ['89941a93']
- s8-socratic: 1/2 on qids ['89941a93']

**multi-session:**
- s0-minimal-baseline: 1/2 on qids ['c4a1ceb8']
- s1-temporal-sysprefix: 1/2 on qids ['c4a1ceb8']
- s2-temporal-inline: 1/2 on qids ['c4a1ceb8']
- s3-contrastive-not: 1/2 on qids ['c4a1ceb8']
- s4-declarative-preference: 1/2 on qids ['c4a1ceb8']
- s5-multi-session-aggregation: 1/2 on qids ['c4a1ceb8']
- s6-code-scaffold: 1/2 on qids ['c4a1ceb8']
- s7-chain-of-thought: 1/2 on qids ['c4a1ceb8']
- s8-socratic: 1/2 on qids ['c4a1ceb8']
- s9-hypothesis: 1/2 on qids ['c4a1ceb8']
- s10-anti-confidence: 1/2 on qids ['c4a1ceb8']

**single-session-assistant:**
- s1-temporal-sysprefix: 1/1 on qids ['18dcd5a5']
- s2-temporal-inline: 1/1 on qids ['18dcd5a5']
- s3-contrastive-not: 1/1 on qids ['18dcd5a5']
- s4-declarative-preference: 1/1 on qids ['18dcd5a5']
- s5-multi-session-aggregation: 1/1 on qids ['18dcd5a5']
- s6-code-scaffold: 1/1 on qids ['18dcd5a5']
- s7-chain-of-thought: 1/1 on qids ['18dcd5a5']
- s9-hypothesis: 1/1 on qids ['18dcd5a5']
- s10-anti-confidence: 1/1 on qids ['18dcd5a5']

**single-session-preference:**
- s1-temporal-sysprefix: 3/3 on qids ['06f04340', 'd24813b1', '1c0ddc50']
- s3-contrastive-not: 3/3 on qids ['06f04340', 'd24813b1', '1c0ddc50']
- s4-declarative-preference: 3/3 on qids ['06f04340', 'd24813b1', '1c0ddc50']
- s5-multi-session-aggregation: 3/3 on qids ['06f04340', 'd24813b1', '1c0ddc50']
- s6-code-scaffold: 3/3 on qids ['06f04340', 'd24813b1', '1c0ddc50']
- s0-minimal-baseline: 2/3 on qids ['06f04340', 'd24813b1']
- s2-temporal-inline: 2/3 on qids ['06f04340', 'd24813b1']
- s9-hypothesis: 2/3 on qids ['06f04340', 'd24813b1']
- s10-anti-confidence: 2/3 on qids ['06f04340', '1c0ddc50']
- s7-chain-of-thought: 1/3 on qids ['d24813b1']
- s8-socratic: 1/3 on qids ['d24813b1']

**temporal-reasoning:**
- s2-temporal-inline: 8/12 on qids ['gpt4_fa19884c', '4dfccbf7', 'gpt4_61e13b3c']
- s4-declarative-preference: 7/12 on qids ['gpt4_fa19884c', '4dfccbf7', 'gpt4_e061b84f']
- s5-multi-session-aggregation: 7/12 on qids ['gpt4_fa19884c', 'gpt4_61e13b3c', 'gpt4_e061b84f']
- s9-hypothesis: 7/12 on qids ['gpt4_fa19884c', 'gpt4_61e13b3c', 'gpt4_e061b84f']
- s0-minimal-baseline: 6/12 on qids ['gpt4_fa19884c', 'gpt4_61e13b3c', 'gpt4_e061b84f']
- s3-contrastive-not: 6/12 on qids ['gpt4_fa19884c', 'gpt4_61e13b3c', 'gpt4_e061b84f']
- s6-code-scaffold: 6/12 on qids ['gpt4_fa19884c', 'gpt4_e061b84f', 'gpt4_59149c78']
- s7-chain-of-thought: 6/12 on qids ['gpt4_fa19884c', 'gpt4_b5700ca9', 'gpt4_1916e0ea']
- s10-anti-confidence: 6/12 on qids ['gpt4_fa19884c', 'gpt4_61e13b3c', 'gpt4_e061b84f']
- s1-temporal-sysprefix: 5/12 on qids ['gpt4_61e13b3c', 'gpt4_e061b84f', 'gpt4_fa19884d']
- s8-socratic: 5/12 on qids ['gpt4_fa19884c', 'gpt4_e061b84f', 'gpt4_59149c78']

## 6. v0.3.1 Dispatcher Config

```python
SCAFFOLD_DISPATCHER = {
    "knowledge-update": "s0-minimal-baseline",  # 2/2 failures fixed
    "multi-session": "s0-minimal-baseline",  # 1/2 failures fixed
    "single-session-assistant": "s1-temporal-sysprefix",  # 1/1 failures fixed
    "single-session-preference": "s1-temporal-sysprefix",  # 3/3 failures fixed
    "temporal-reasoning": "s2-temporal-inline",  # 8/12 failures fixed
}
```

To use: detect qtype at query-time (or use existing router), then select scaffold from dispatcher before calling LLM reranker.

## 7. Deep-Dive: Temporal Composition Analysis

Temporal-reasoning has 12 failures in Set A. Best single scaffold is s2-temporal-inline (8/12).
The 4 temporal failures that s2-inline misses:

| qid | gold_at | What s2-inline does wrong | s7-CoT? |
|-----|---------|--------------------------|---------|
| gpt4_b5700ca9 | pos 1 | ranks [3] — misses it | WIN (ranks [1]) |
| gpt4_1916e0ea | pos 1 | ranks [3] — misses it | WIN (ranks [1]) |
| 9a707b82 | pos 2 | ranks [5] — misses it | misses (ranks [1]) |
| 5e1b23de | pos 1 | ranks [4] — misses it | WIN (ranks [1]) |

**s7-CoT (chain-of-thought) fixes 3 of the 4 that s2-inline misses** (gpt4_b5700ca9, gpt4_1916e0ea, 5e1b23de). These are the "how many days ago" and "how many months ago" questions where CoT reasoning about the time gap matters more than date sorting.

**Composition result:** `s2-temporal-inline + s7-chain-of-thought` (call both, take union) would fix **10/12** temporal failures vs 8/12 for s2-inline alone.

The remaining 2 hard cases (9a707b82, multi-session `6d550036`) are resistant to all scaffolds — likely require retrieval-layer fixes, not reranking.

## 8. Key Failure Patterns (Resistant to All Scaffolds)

These qids failed under ALL 11 scaffolds:

| qid | qtype | gold_at | pattern |
|-----|-------|---------|---------|
| 6d550036 | multi-session | pos 3 | "How many projects" — needs multiple sessions counted, retrieval ranks wrong session first |
| 9a707b82 | temporal-reasoning | pos 2 | "cooking for friend a couple of days ago" — vague temporal reference, no date match |

Both are **retrieval-layer problems** where the semantic signal is too weak for any scaffold to overcome. The LLM consistently picks a surface-similar but wrong candidate regardless of scaffold framing.

## 9. Summary Statistics

| Scaffold | Total wins / 20 | vs baseline | Key strength |
|----------|----------------|-------------|--------------|
| s2-temporal-inline | **14/20** | **+3** | Best overall. Dominates temporal. |
| s3-contrastive-not | 13/20 | +2 | Good at preference + temporal balance |
| s4-declarative-preference | 13/20 | +2 | Fixes ALL 3 preference failures; good temporal |
| s5-multi-session-aggregation | 13/20 | +2 | Strong preference; mirrors s4 |
| s10-anti-confidence | 12/20 | +1 | Small lift across categories |
| s0-minimal-baseline | 11/20 | 0 | Reference |
| s1-temporal-sysprefix | 11/20 | 0 | No net gain; hurts some temporals |
| s6-code-scaffold | 11/20 | 0 | No gain vs baseline |
| s7-chain-of-thought | 11/20 | 0 | Unique fixes on hard temporals; but regresses preference |
| s9-hypothesis | 11/20 | 0 | Matches baseline; occasional temporal help |
| s8-socratic | 8/20 | **-3** | Actively harmful — do not use |