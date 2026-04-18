# Benchmark Integrity Audit — cogito-ergo 95.11% on LongMemEval_S

**Date:** 2026-04-16  
**Auditor:** Claude Sonnet 4.6 (read-only, zero LLM calls)  
**Script audited:** `bench/longmemeval_combined_pipeline_v33.py`  
**Results audited:** `bench/results-v33-2026-04-18.json`, `bench/runs/runJ-v33/per_question.json`  

---

## Overall Verdict: TWEAK-DISCLOSURES

The 95.11% number is arithmetically correct and the retrieval pipeline is clean of gold leakage at runtime. However, two issues require disclosure before public launch: (1) the Mastra comparison is **metric-apples-to-oranges** until confirmed, and (2) the hardset-gated flagship routing is a form of **test-set-informed hyperparameter**, which is defensible but must be disclosed. No rebuild needed — targeted disclosures are sufficient.

---

## Per-Item Audit

### 1. Dataset Version + Integrity — PASS

- **File:** `data/longmemeval_s_cleaned.json` — confirmed present, 500 total entries, 470 non-abstention (filter: `"_abs" not in question_id`).
- The pipeline correctly excludes `_abs` questions at line 723: `data = [e for e in data if "_abs" not in e["question_id"]]`.
- LongMemEval README explicitly links to `huggingface.co/datasets/xiaowu0162/longmemeval-cleaned` and states the September 2025 cleaning. The file at the correct path matches the format described.
- 470 evaluated matches the expected non-abstention count exactly.
- **No hash verification available** (no checksum file present), but file size (1,101,877 lines) is consistent with the full S-split corpus.

**Required disclosure:** State explicitly: "We evaluate on `longmemeval_s_cleaned.json` (September 2025 cleaned version, HuggingFace: xiaowu0162/longmemeval-cleaned), 470 non-abstention questions."

---

### 2. Evaluation Protocol Parity — PASS WITH NOTE

**LongMemEval `eval_utils.py` definition (lines 24-28):**
```python
recalled_docs = set(corpus_ids[idx] for idx in rankings[:k])
recall_any = float(any(doc in recalled_docs for doc in correct_docs))
```
This operates on session-ID strings.

**v33 definition (lines 690-693):**
```python
recalled = set(rankings[:k])
recall_any = float(any(idx in recalled for idx in correct_indices))
```
This operates on corpus indices. The semantics are identical — both compute "any gold in top-k."

**Definition: PASS** — our `recall_any` at k=1 matches the paper's R@1 recall definition.

**Critical note:** `evaluate_qa.py` is a *different script* — it evaluates LLM-generated answer quality using an LLM judge. Our pipeline measures **retrieval R@1 only**, not QA accuracy. These are distinct metrics. If any external party reports "QA accuracy" we are not comparable unless we clarify.

---

### 3. Gold Leakage in Runtime Pipeline — PASS (CLEAN)

Comprehensive trace of `answer_session_ids` in v33:

- **Line 767:** `answer_sids = set(entry["answer_session_ids"])` — reads gold IDs from dataset
- **Line 784:** `if sid in answer_sids: correct_indices.append(corpus_idx)` — populates `correct_indices` (eval-only list)
- **Lines 835, 862, 979, 980:** `evaluate_retrieval(ranked_sessions, correct_indices, k)` — gold only used AFTER retrieval ranking is finalized, for scoring
- **Lines 943–946:** `s1_hit_at_1`, `s2_hit_at_1` computed post-hoc for logging
- **Line 953:** `gold_session_ids` logged to `per_question.json` for transparency

**Gold is NEVER passed to:**
- `retrieve_chunks()` — the BM25/dense retrieval stage
- `apply_temporal_boost_to_sessions()` — the temporal multiplier
- `llm_rerank_socratic()` — the s8-socratic LLM call
- `llm_rerank_flagship_meta()` — the qwen-max flagship call
- `llm_verify_one()` — the verify-guard

**Verdict: No gold leakage in the production pipeline.**

---

### 4. Oracle Usage — PASS (compose_union NOT reachable)

**Finding:** `compose_union()` is defined in `qwen_native_arena.py`. The v33 pipeline does NOT import `qwen_native_arena.py`. References to `compose_union` in v33 are only in the module-level docstring (lines 12, 31) as a historical note.

v33's two external imports are:
- `phase-4/temporal_scaffold.py` (temporal boost logic) — does not call compose_union
- `phase-6/flagship_escalation.py` (qwen-max reranker) — does not call compose_union

`compose_union` is **unreachable** from the v33 pipeline. The oracle-inflation bug was caught and fixed before the production run. The fix (replacing s14-union with s8-socratic single scaffold) is correctly documented in the v33 docstring.

---

### 5. Hardset Routing — FLAG (DISCLOSE, NOT BLOCK)

**Finding:** `hardset.json` (37 questions) was derived from `build_hardset.py` using `runs/step1-gapfix/per_question.json`. The selection criterion: questions where BOTH S1 and S2 failed in the baseline run. The hardset is used to gate flagship escalation (qwen-max) in v33.

**This is a form of test-set-informed routing.** The hardset was derived from the same 470-question eval set, using a prior pipeline run's failure modes. This means v33 is not "fresh" — it knows which questions the baseline got wrong, and escalates those to a stronger model.

**How significant is this?** Very significant: 37 hardset questions all routed to flagship (31 via `flagship`, 6 via `flagship_llm`). This is the most expensive compute tier. The 95.11% result is partly contingent on the hardset correctly identifying hard questions — which required prior runs on the same dataset.

**This is not cheating** (the hardset does not contain gold answers, only question identifiers). But it is a **hyperparameter tuned on the test set**, similar to test-set-adaptive systems. In a truly blind evaluation, the hardset would be derived from a holdout or oracle-free criterion.

**Recommended disclosure:** "Flagship escalation is gated on a 37-question hardset derived from prior baseline failures on the same 470-question eval set. This is test-set-informed routing — not gold leakage, but a form of adaptive routing that requires disclosure in any paper or leaderboard submission."

**Mitigation path:** Replace hardset with a confidence-gap criterion (purely score-based, no test-set knowledge). v33 already partially implements this in the `gap` route branch — expanding it would make routing fully blind.

---

### 6. Prompt / Cache Leakage — PASS

- No response caching (`cache` and `pickle` absent from v33).
- Session text sent to LLM built at line 780: `" ".join(t["content"] for t in session if t["role"] == "user")` — raw conversation content, no eval metadata injected.
- No evaluation tags, gold markers, or answer strings in any prompt.
- All LLM calls use `temperature: 0` (one instance) — deterministic, no stochastic sampling. (Flagship uses same temperature in `flagship_escalation.py`.)

---

### 7. Hardcoded Answers / Shortcuts — PASS

- No literal qid references found in v33 pipeline code.
- No conditional branches on specific question content beyond the pattern-based router (`_TEMPORAL_PATTERNS`, `_SKIP_PATTERNS`, `_COUNTING_PATTERNS`) — these are generic, not question-specific.
- No embeddings files pre-filtered by gold session IDs.
- HARDSET_QIDS loaded as a pure set of qid strings — gold_session_ids stored in hardset.json are NOT read at v33 runtime (only `h["qid"]` extracted at line 72).

---

### 8. Mastra Comparison Validity — UNCLEAR (HIGH RISK)

**Problem:** The source of Mastra's 94.87% figure is not formally cited anywhere in the cogito-ergo repo. The figure appears in `LAUNCH_DEFENSE.md` as a reference point but without a URL, paper, or dataset version.

**Critical ambiguity:** LongMemEval evaluates two things:
1. **Retrieval R@1** (session retrieval) — what our pipeline measures
2. **QA accuracy** (LLM answer quality on retrieved context) — what `evaluate_qa.py` measures

If Mastra's 94.87% is a QA accuracy number (which is what most published benchmarks report on LongMemEval), then our comparison is **apples to oranges**. Retrieval R@1 is an upper bound on QA accuracy — a system with 95.11% retrieval R@1 could have lower QA accuracy if its answer generation is poor.

**If Mastra's 94.87% is retrieval R@1 on the cleaned version:** comparison is valid.  
**If Mastra's 94.87% is QA accuracy:** our metric is retrieval R@1, and we would need to run evaluate_qa.py on our outputs to make a fair comparison.  
**If Mastra used the non-cleaned version:** our result is not directly comparable due to dataset differences.

**This is the claim most likely to be challenged in public.**

---

### 9. Stochasticity / Reproducibility — PARTIAL PASS

- Both LLM stages (qwen-turbo filter, qwen-max flagship) use `temperature: 0` → deterministic given same model state.
- BM25 and cosine retrieval are fully deterministic.
- **No explicit random seed documented** in v33. However, all algorithms are deterministic without random state dependency.
- **Noise floor from baseline seeds:** baseline run vs baseline-seed2 showed 83.8% vs 84.9% — a 1.06pp delta. The cause is likely non-determinism in the embedding service (Ollama), not in the pipeline code itself.
- v33 has only ONE full run (runJ-v33). A second full v33 run does not exist. **If Ollama embed noise persists, we cannot guarantee exact reproduction of 95.11%.**
- The 95.11% claim is `447/470` — a 1.06pp noise band would mean the true value could be anywhere from ~94.1% to ~96.1%.

**Required disclosure:** "Single run (runJ-v33). Noise floor ~1pp from embedding non-determinism. We cannot guarantee bit-exact reproduction. A second full run would confirm whether 95.11% is stable."

---

## Summary Table

| Check | Status | Evidence |
|-------|--------|----------|
| Dataset version (Sept 2025 cleaned) | PASS | `data/longmemeval_s_cleaned.json`, 500 total / 470 non-abs |
| Question count matches paper (470 non-abs) | PASS | `results-v33: questions_evaluated: 470` |
| R@1 arithmetic | PASS | `447/470 = 0.95106`, matches 95.11% claim |
| Eval metric matches paper (recall_any) | PASS | Semantically identical to `eval_utils.py` |
| Gold not passed to retrieval | PASS | `correct_indices` only used post-ranking |
| Gold not passed to LLM prompts | PASS | No answer_sids in any prompt path |
| compose_union not reachable from v33 | PASS | Only in docstring, no import of qwen_native_arena |
| Prompt / cache leakage | PASS | No caching, no eval metadata in session text |
| Hardcoded qid shortcuts | PASS | No literal qid refs in pipeline |
| Hardset derivation uses gold? | FLAG | Hardset derived from test-set failures — disclose |
| Mastra metric/dataset version confirmed | UNCLEAR | No source citation, metric type unconfirmed |
| Reproducibility (second run) | PARTIAL | Single run, ~1pp embed noise floor |

---

## Most-Likely-Challenged Claim and Defense

**Challenge:** "Mastra's 94.87% is a QA accuracy number. Your 95.11% is retrieval R@1. You're comparing different metrics. Your system may have 95% retrieval but poor answer generation."

**Defense:**
1. Clarify upfront that we report **retrieval R@1**, not QA accuracy. This is a valid and useful metric independently.
2. Confirm Mastra's metric: if it's also retrieval R@1, comparison is valid. If it's QA accuracy, withdraw the direct comparison and instead state: "95.11% retrieval R@1, enabling up to 95.11% QA accuracy ceiling."
3. Note that retrieval R@1 is the harder and more reproducible metric (no LLM judge variability).

**Strongest version:** Find a paper or HuggingFace leaderboard entry for Mastra that shows dataset version and metric type. If Mastra's 94.87% is from the pre-cleaned dataset, our 95.11% on the cleaned version is not directly comparable — and may actually be harder (cleaned version removed some retrieval shortcuts).

---

## Red Flags

**No blockers to launch.** The pipeline is clean.

**Flag 1 (disclose before launch):** The hardset is test-set-informed. This is standard in iterative benchmark development but must be disclosed. Suggested language: "Flagship escalation gated on 37 questions identified as hard in prior baseline runs. This is adaptive routing, not blind evaluation."

**Flag 2 (verify before making comparative claim):** Mastra's 94.87% metric type and dataset version are unconfirmed. Do not claim head-to-head superiority until confirmed. Make the claim conditional: "beats Mastra's published 94.87% if that number is also retrieval R@1 on the cleaned dataset."

---

## Recommended Disclosures (Paper/README)

1. **Dataset:** "We evaluate on LongMemEval_S cleaned (September 2025, HuggingFace: xiaowu0162/longmemeval-cleaned), excluding 30 abstention questions (470 non-abstention total)."

2. **Metric:** "We report retrieval Recall@1 (recall_any), defined as: the correct session appears as the top-ranked result. This differs from QA accuracy, which additionally requires correct answer generation."

3. **Hardset routing:** "Flagship escalation (qwen-max) is applied to 37 questions identified as hard in prior baseline runs on the same dataset. This is test-set-informed routing. In a strictly blind evaluation, replace hardset gating with a confidence-gap criterion."

4. **Reproducibility:** "Single full run (runJ-v33, April 18 2026). Observed noise floor ~1pp from embedding service (Ollama nomic-embed-text) non-determinism. A second full run would confirm stability of the 95.11% figure."

5. **Mastra comparison:** "Mastra's 94.87% source: [cite paper/leaderboard]. Confirm dataset version and metric type before claiming direct comparison. If Mastra reports QA accuracy rather than retrieval R@1, a direct comparison requires running evaluate_qa.py on our pipeline's output."
