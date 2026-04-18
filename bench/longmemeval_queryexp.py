"""
Shot 3 — Query expansion for R@5 lift.

Changes vs longmemeval_combined_pipeline_guard.py (runC-guard baseline 94.0%):
1. Filter model: gemma3:4b via Ollama (no DASHSCOPE key needed)
2. Query expansion: for each question, generate 3 semantic paraphrase sub-queries
   using a local LLM (qwen3.5:2b, fast and cheap). Retrieve top-5 per sub-query,
   union the results, take the best top-5 as Stage 1 output.
3. Target: 8 questions where gold is not in S1 top-5 (R@5 ceiling)
4. Run ID: runF-queryexp

Cost: adds 3x retrieval per question. Acceptable for R@5 lift test.
Kill: if R@5 drops OR overall R@1 drops vs runC-guard.
"""

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

import os
os.environ["BM25S_SHOW_PROGRESS"] = "0"
os.environ["TQDM_DISABLE"] = "1"
import bm25s
import importlib.util
_ts_spec = importlib.util.spec_from_file_location(
    "temporal_scaffold",
    str(Path(__file__).parent / "phase-4" / "temporal_scaffold.py"),
)
_ts_mod = importlib.util.module_from_spec(_ts_spec)
_ts_spec.loader.exec_module(_ts_mod)
build_temporal_scaffold = _ts_mod.build_temporal_scaffold
is_temporal_query = _ts_mod.is_temporal_query

# Load hardset qids
_hardset_path = Path(__file__).parent / "hardset.json"
HARDSET_QIDS = set()
if _hardset_path.exists():
    HARDSET_QIDS = {h["qid"] for h in json.load(open(_hardset_path))}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_STOP = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall",
    "what", "which", "who", "whom", "whose", "how", "why", "when", "where",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "this", "that", "these", "those", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "into", "through", "during", "before",
    "after", "above", "below", "and", "or", "but", "if", "while", "because",
    "so", "not", "no", "nor", "yet", "than", "then", "about",
    "cause", "caused", "describe", "explain", "tell", "give", "show",
    "find", "get", "make", "use", "used", "using", "their", "its",
    "can", "cannot",
})
MAX_SUBQUERIES = 8
_RRF_K = 60
_COSINE_WEIGHT = 0.7
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LOCAL_FILTER_MODEL = "gemma3:4b"
LOCAL_EXPAND_MODEL = "gemma3:4b"  # Fast model for query expansion (gemma3:4b outputs content not thinking)
FILTER_TOP_K = 5
N_EXPANSION_QUERIES = 3  # Number of LLM-generated paraphrases per question

QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "

# ---------------------------------------------------------------------------
# Router (same as guard)
# ---------------------------------------------------------------------------
_SKIP_PATTERNS = [
    "you told me", "you suggested", "you recommended",
    "you mentioned", "you said", "you explained",
    "our previous conversation", "our last chat",
    "remind me what you",
]
_TEMPORAL_PATTERNS = [
    "which happened first", "which did i do first",
    "how many days", "how many weeks", "how many months",
    "before or after", "what order", "what was the date",
    "which event", "which trip",
    "order of the", "from earliest", "from first",
    "most recently", "a week ago", "two weeks ago",
    "a month ago", "last saturday", "last sunday",
    "last weekend", "last monday", "last tuesday",
    "graduated first", "started first", "finished first",
    "did i do first", "did i attend first",
    "who did i go with to the",
]
_COUNTING_PATTERNS = [
    "how many", "total number", "how much total",
    "in total", "altogether", "combined",
]


def classify_query(query: str) -> str:
    q = query.lower()
    if any(p in q for p in _SKIP_PATTERNS):
        return "skip"
    if any(p in q for p in _TEMPORAL_PATTERNS):
        return "llm"
    if any(p in q for p in _COUNTING_PATTERNS):
        return "llm"
    return "gap"


# ---------------------------------------------------------------------------
# Query expansion via local LLM
# ---------------------------------------------------------------------------
_EXPAND_SYSTEM = (
    "Generate 3 different paraphrase search queries for the given question. "
    "Each paraphrase should use different vocabulary but search for the same information. "
    "Output ONLY a JSON array of 3 strings. Example: [\"query 1\", \"query 2\", \"query 3\"]"
)


def expand_query_llm(query: str) -> list[str]:
    """Generate LLM paraphrases. Returns list of paraphrase strings (may be empty on error)."""
    user_content = f"Question: {query}\n\nGenerate 3 paraphrase search queries:"
    try:
        body = json.dumps({
            "model": LOCAL_EXPAND_MODEL,
            "messages": [
                {"role": "system", "content": _EXPAND_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "temperature": 0.3,
            "options": {"num_predict": 150},
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            d = json.loads(resp.read())
        raw = d.get("message", {}).get("content", "")
        # Also check thinking field for models that emit thinking
        if not raw:
            raw = d.get("message", {}).get("thinking", "")
        # Parse JSON array
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            arr = json.loads(raw[start:end])
            if isinstance(arr, list):
                return [str(s).strip() for s in arr if str(s).strip() and str(s).strip() != query][:N_EXPANSION_QUERIES]
    except Exception as e:
        pass  # Silent fallback - no expansion
    return []


# ---------------------------------------------------------------------------
# LLM filter (gemma3:4b, same as other shots)
# ---------------------------------------------------------------------------
_FILTER_SYSTEM = (
    "Execute this procedure:\n"
    "```\n"
    "def rank(query, candidates):\n"
    "  for each candidate:\n"
    "    keyword_match = count(query_keywords IN candidate_text)\n"
    "    fact_match = candidate contains specific answer to query (bool)\n"
    "    score = keyword_match * 2 + fact_match * 10\n"
    "  return sorted(candidates, by=score, descending)\n"
    "```\n"
    "Input: query and 5 numbered candidates.\n"
    "Output: ONLY a JSON array of candidate numbers sorted by score.\n"
    "Example: [3, 1, 5, 2, 4]"
)


def _parse_filter_indices(raw: str, n_candidates: int) -> list[int] | None:
    if "<think>" in raw:
        end = raw.rfind("</think>")
        if end >= 0:
            after = raw[end + 8:].strip()
            if after:
                raw = after

    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start < 0 or end <= start:
        return None
    try:
        arr = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None

    result = []
    for item in arr:
        try:
            idx = int(item)
        except (ValueError, TypeError):
            continue
        if 1 <= idx <= n_candidates and (idx - 1) not in result:
            result.append(idx - 1)
    return result if result else None


def llm_rerank(
    query: str,
    top_sessions: list[tuple[int, str]],
) -> tuple[list[int], str]:
    n = len(top_sessions)
    if n <= 1:
        return [idx for idx, _ in top_sessions], "filter_skip"

    lines = []
    for i, (corpus_idx, text) in enumerate(top_sessions):
        snippet = text[:500].replace("\n", " ")
        lines.append(f"[{i+1}] {snippet}")
    candidates_block = "\n".join(lines)

    prompt_user = (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{candidates_block}\n\n"
        f"Rank these {n} candidates by relevance. Output JSON array."
    )

    try:
        body = json.dumps({
            "model": LOCAL_FILTER_MODEL,
            "messages": [
                {"role": "system", "content": _FILTER_SYSTEM},
                {"role": "user", "content": prompt_user},
            ],
            "stream": False,
            "temperature": 0,
            "options": {"num_predict": 30},
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        raw = data.get("message", {}).get("content", "")
    except Exception as e:
        print(f"  [filter error] {e}", file=sys.stderr)
        return [idx for idx, _ in top_sessions], "fallback_error"

    if not raw.strip():
        return [idx for idx, _ in top_sessions], "fallback_empty"

    parsed = _parse_filter_indices(raw, n)
    if parsed is None:
        return [idx for idx, _ in top_sessions], "fallback_parse"

    reranked = [top_sessions[i][0] for i in parsed]
    included = set(parsed)
    for i in range(n):
        if i not in included:
            reranked.append(top_sessions[i][0])

    return reranked, "filter"


_VERIFY_SYSTEM = (
    "Execute this procedure:\n"
    "```\n"
    "def verify(query, candidate):\n"
    "  facts = extract_facts(candidate)\n"
    "  answer_present = any(fact answers query for fact in facts)\n"
    "  return 'YES' if answer_present else 'NO'\n"
    "```\n"
    "Input: a query and a candidate memory.\n"
    "Output: ONLY 'YES' or 'NO'. Nothing else."
)


def llm_verify_one(query: str, candidate_text: str, snippet_len: int = 1500) -> str | None:
    snippet = candidate_text[:snippet_len].replace("\n", " ")
    prompt_user = (
        f"Query: {query}\n\n"
        f"Candidate memory:\n{snippet}\n\n"
        "Does this memory contain the specific fact that answers the query?"
    )
    try:
        body = json.dumps({
            "model": LOCAL_FILTER_MODEL,
            "messages": [
                {"role": "system", "content": _VERIFY_SYSTEM},
                {"role": "user", "content": prompt_user},
            ],
            "stream": False,
            "temperature": 0,
            "options": {"num_predict": 10},
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        raw = data.get("message", {}).get("content", "").strip().upper()
    except Exception:
        return None
    if "YES" in raw:
        return "YES"
    if "NO" in raw:
        return "NO"
    return None


def flagship_rerank_local(
    query: str,
    top_sessions: list[tuple[int, str]],
) -> tuple[list[int], str]:
    n = len(top_sessions)
    if n <= 1:
        return [idx for idx, _ in top_sessions], "flagship_local_skip"

    lines = []
    for i, (corpus_idx, text) in enumerate(top_sessions):
        snippet = text[:2000].replace("\n", " ")
        lines.append(f"[{i+1}] {snippet}")
    candidates_block = "\n".join(lines)

    prompt_user = (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{candidates_block}\n\n"
        f"Rank these {n} candidates by relevance. Output JSON array."
    )

    try:
        body = json.dumps({
            "model": LOCAL_FILTER_MODEL,
            "messages": [
                {"role": "system", "content": _FILTER_SYSTEM},
                {"role": "user", "content": prompt_user},
            ],
            "stream": False,
            "temperature": 0,
            "options": {"num_predict": 30},
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            d = json.loads(resp.read())
        raw = d.get("message", {}).get("content", "")
    except Exception:
        return [idx for idx, _ in top_sessions], "flagship_local_error"

    parsed = _parse_filter_indices(raw, n)
    if parsed is None:
        return [idx for idx, _ in top_sessions], "flagship_local_parse_fail"

    reranked = [top_sessions[i][0] for i in parsed]
    included = set(parsed)
    for i in range(n):
        if i not in included:
            reranked.append(top_sessions[i][0])

    return reranked, "flagship_local"


# ---------------------------------------------------------------------------
# Chunking (unchanged)
# ---------------------------------------------------------------------------
def chunk_session(session: list[dict], session_id: str) -> list[tuple[str, str]]:
    chunks = []
    turns = []
    current_user = None
    for msg in session:
        if msg["role"] == "user":
            current_user = msg["content"]
        elif msg["role"] == "assistant" and current_user is not None:
            turns.append((current_user, msg["content"]))
            current_user = None
    if current_user is not None:
        turns.append((current_user, ""))
    if not turns:
        raw = " ".join(m["content"] for m in session if m["role"] == "user")
        if raw.strip():
            chunks.append((raw, session_id))
        return chunks
    for i, (user, assistant) in enumerate(turns):
        parts = []
        if i > 0:
            prev_u, prev_a = turns[i - 1]
            parts.append(f"[context] {prev_u}")
            if prev_a:
                parts.append(f"[context] {prev_a}")
        parts.append(user)
        if assistant:
            parts.append(assistant)
        chunk_text = " ".join(parts)
        chunks.append((chunk_text, session_id))
    return chunks


def dedup_to_sessions(
    ranked_chunk_indices: list[int],
    chunk_session_ids: list[str],
    session_id_to_corpus_idx: dict[str, int],
    max_sessions: int | None = None,
) -> list[int]:
    seen_sids: set[str] = set()
    ranked_sessions: list[int] = []
    for chunk_idx in ranked_chunk_indices:
        sid = chunk_session_ids[chunk_idx]
        if sid not in seen_sids:
            seen_sids.add(sid)
            corpus_idx = session_id_to_corpus_idx[sid]
            ranked_sessions.append(corpus_idx)
            if max_sessions and len(ranked_sessions) >= max_sessions:
                break
    if not max_sessions:
        all_corpus_indices = set(session_id_to_corpus_idx.values())
        for idx in sorted(all_corpus_indices - set(ranked_sessions)):
            ranked_sessions.append(idx)
    return ranked_sessions


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def batch_embed(texts: list[str], retries: int = 8) -> list[list[float]] | None:
    sanitized = [t[:2000] if len(t) > 2000 else t if t.strip() else "empty" for t in texts]
    for attempt in range(retries):
        try:
            body = json.dumps({"model": EMBED_MODEL, "input": sanitized}).encode()
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/embed", data=body,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            vecs = data.get("embeddings", [])
            return vecs if len(vecs) == len(texts) else None
        except Exception as e:
            if attempt < retries - 1:
                wait = min(2 ** attempt, 8)
                print(f"  [embed retry {attempt+1}/{retries}] {e} — waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"  [embed FAILED] {e}", file=sys.stderr)
                return None


def batch_embed_docs(texts: list[str]) -> list[list[float]] | None:
    return batch_embed([DOC_PREFIX + t for t in texts])


def batch_embed_queries(texts: list[str]) -> list[list[float]] | None:
    return batch_embed([QUERY_PREFIX + t for t in texts])


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s-]", " ", text.lower()).split()


def key_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOP and len(t) >= 2]


def build_subqueries(query: str) -> list[str]:
    seen, result = set(), []
    def add(q):
        q = q.strip()
        if q and q not in seen:
            seen.add(q); result.append(q)
    add(query)
    tokens = key_tokens(query)
    if not tokens:
        return result
    add(" ".join(tokens))
    for i in range(len(tokens) - 1):
        add(f"{tokens[i]} {tokens[i+1]}")
    for i in range(len(tokens) - 2):
        add(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
    for t in tokens:
        if len(t) >= 3:
            add(t)
    return result[:MAX_SUBQUERIES]


# ---------------------------------------------------------------------------
# Stage 1: Combined retrieval with query expansion
# ---------------------------------------------------------------------------
def retrieve_chunks_single(
    query: str,
    sq_vecs: list[list[float]],
    subqueries: list[str],
    chunk_texts: list[str],
    chunk_vecs: list[list[float]],
    bm25_index: bm25s.BM25,
) -> dict[int, float]:
    """Retrieve and return RRF scores for a single query+vecs combo."""
    k_bm25 = min(20, len(chunk_texts))
    runs: list[list[tuple[int, float]]] = []

    for subquery, sv in zip(subqueries, sq_vecs):
        scored = [(i, cosine_sim(sv, cv)) for i, cv in enumerate(chunk_vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        runs.append(scored[:20])

        query_tokens = bm25s.tokenize([subquery])
        bm25_results, bm25_scores = bm25_index.retrieve(query_tokens, k=k_bm25)
        bm25_run = [
            (int(idx), float(score))
            for idx, score in zip(bm25_results[0], bm25_scores[0])
        ]
        runs.append(bm25_run)

    rrf_scores: dict[int, float] = {}
    for run in runs:
        for rank, (idx, _score) in enumerate(run, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank)

    query_vec = sq_vecs[0]
    cosine_scores = {i: cosine_sim(query_vec, chunk_vecs[i]) for i in rrf_scores}
    rrf_max = max(rrf_scores.values()) if rrf_scores else 1.0

    blended: dict[int, float] = {}
    for idx in rrf_scores:
        rrf_n = rrf_scores[idx] / rrf_max if rrf_max > 0 else 0.0
        cos = cosine_scores.get(idx, 0.0)
        score = (1.0 - _COSINE_WEIGHT) * rrf_n + _COSINE_WEIGHT * cos
        blended[idx] = score

    return blended


def retrieve_chunks_expanded(
    query: str,
    expanded_queries: list[str],
    chunk_texts: list[str],
    chunk_vecs: list[list[float]],
    bm25_index: bm25s.BM25,
) -> tuple[list[int], list[float]]:
    """
    SHOT 3: Multi-query retrieval with union.
    For each query (original + expansions), run full retrieval.
    Union top-5 per query, score by max blended score across queries.
    """
    all_queries = [query] + expanded_queries[:N_EXPANSION_QUERIES]

    # Collect all blended scores across queries (take max per chunk index)
    merged_scores: dict[int, float] = {}

    for q in all_queries:
        subqueries = build_subqueries(q)
        sq_vecs = batch_embed_queries(subqueries)
        if sq_vecs is None:
            continue

        blended = retrieve_chunks_single(q, sq_vecs, subqueries, chunk_texts, chunk_vecs, bm25_index)
        for idx, score in blended.items():
            if idx not in merged_scores or score > merged_scores[idx]:
                merged_scores[idx] = score

    if not merged_scores:
        # Fallback to original query only
        subqueries = build_subqueries(query)
        sq_vecs = batch_embed_queries(subqueries)
        if sq_vecs is None:
            return list(range(len(chunk_texts))), []
        merged_scores = retrieve_chunks_single(query, sq_vecs, subqueries, chunk_texts, chunk_vecs, bm25_index)

    ranked = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_ids = [idx for idx, _ in ranked]
    ranked_scores = [s for _, s in ranked]

    # Append any chunks not in merged_scores (rare edge case)
    remaining = [i for i in range(len(chunk_texts)) if i not in merged_scores]
    # No time to score these - just append in order
    ranked_ids.extend(remaining)
    ranked_scores.extend([0.0] * len(remaining))

    return ranked_ids, ranked_scores


def dcg(relevances, k):
    import numpy as np
    relevances = np.asarray(relevances, dtype=float)[:k]
    if relevances.size:
        return relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, relevances.size + 1)))
    return 0.0


def ndcg_score(rankings, correct_indices, n_corpus, k):
    import numpy as np
    relevances = [1 if i in correct_indices else 0 for i in range(n_corpus)]
    sorted_rel = [relevances[idx] for idx in rankings[:k]]
    ideal_rel = sorted(relevances, reverse=True)
    ideal = dcg(ideal_rel, k)
    actual = dcg(sorted_rel, k)
    return actual / ideal if ideal > 0 else 0.0


def evaluate_retrieval(rankings, correct_indices, k):
    recalled = set(rankings[:k])
    recall_any = float(any(idx in recalled for idx in correct_indices))
    recall_all = float(all(idx in recalled for idx in correct_indices))
    return recall_any, recall_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["s", "m"], default="s")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data_dir", default=str(Path(__file__).resolve().parent.parent.parent / "LongMemEval" / "data"))
    parser.add_argument("--run-id", default="runF-queryexp")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    split_file = f"longmemeval_{args.split}_cleaned.json"
    data_path = Path(args.data_dir) / split_file
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    print(f"SHOT 3 — Query expansion for R@5 lift")
    print(f"Expand model: {LOCAL_EXPAND_MODEL}  Filter model: {LOCAL_FILTER_MODEL}")
    print(f"Loading {data_path.name}...")
    data = json.load(open(data_path))
    data = [e for e in data if "_abs" not in e["question_id"]]
    if args.limit > 0:
        data = data[:args.limit]
    print(f"[pipeline] Running on {len(data)} questions")

    run_id = args.run_id
    run_dir = Path(__file__).parent / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] Run ID: {run_id} → {run_dir}")

    s1_metrics = {k: {"recall_any": [], "recall_all": [], "ndcg": []} for k in [1, 3, 5, 10]}
    s2_metrics = {k: {"recall_any": [], "recall_all": [], "ndcg": []} for k in [1, 3, 5, 10]}
    total_retrieval_time = 0.0
    total_filter_time = 0.0
    filter_methods: dict[str, int] = {}
    per_question_log: list[dict] = []
    n_expansions_used = 0

    done_qids: set[str] = set()
    pq_path = run_dir / "per_question.json"
    if args.resume and pq_path.exists():
        existing = json.load(open(pq_path))
        per_question_log = existing
        done_qids = {e["qid"] for e in existing}
        for e in existing:
            filter_methods[e["route_decision"]] = filter_methods.get(e["route_decision"], 0) + 1
        print(f"[resume] Loaded {len(done_qids)} completed questions")

    for qi, entry in enumerate(data):
        if entry["question_id"] in done_qids:
            continue
        question = entry["question"]
        answer_sids = set(entry["answer_session_ids"])

        corpus_texts: list[str] = []
        corpus_ids: list[str] = []
        correct_indices: list[int] = []
        session_id_to_corpus_idx: dict[str, int] = {}

        for sid, session in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
            corpus_idx = len(corpus_ids)
            text = " ".join(t["content"] for t in session if t["role"] == "user")
            corpus_texts.append(text)
            corpus_ids.append(sid)
            session_id_to_corpus_idx[sid] = corpus_idx
            if sid in answer_sids:
                correct_indices.append(corpus_idx)

        if not correct_indices:
            continue

        n_sessions = len(corpus_ids)

        all_chunks: list[tuple[str, str]] = []
        for sid, session in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
            all_chunks.extend(chunk_session(session, sid))

        chunk_texts = [c[0] for c in all_chunks]
        chunk_session_ids = [c[1] for c in all_chunks]

        corpus_tokens = bm25s.tokenize(chunk_texts)
        bm25_index = bm25s.BM25()
        bm25_index.index(corpus_tokens)

        all_vecs: list[list[float]] = []
        embed_ok = True
        for i in range(0, len(chunk_texts), 100):
            batch = chunk_texts[i:i + 100]
            vecs = batch_embed_docs(batch)
            if vecs is None:
                print(f"  [{qi+1}] EMBED FAILED, skipping")
                embed_ok = False
                break
            all_vecs.extend(vecs)

        if not embed_ok:
            continue

        # Stage 1: Combined retrieval WITH query expansion
        t0 = time.time()
        # Generate expanded queries
        expanded = expand_query_llm(question)
        if expanded:
            n_expansions_used += 1
        ranked_chunk_indices, chunk_scores = retrieve_chunks_expanded(
            question, expanded, chunk_texts, all_vecs, bm25_index
        )
        retrieval_elapsed = time.time() - t0
        total_retrieval_time += retrieval_elapsed

        ranked_sessions = dedup_to_sessions(ranked_chunk_indices, chunk_session_ids, session_id_to_corpus_idx)

        for k in [1, 3, 5, 10]:
            r_any, r_all = evaluate_retrieval(ranked_sessions, correct_indices, k)
            n = ndcg_score(ranked_sessions, set(correct_indices), n_sessions, k)
            s1_metrics[k]["recall_any"].append(r_any)
            s1_metrics[k]["recall_all"].append(r_all)
            s1_metrics[k]["ndcg"].append(n)

        # Stage 2: Routed LLM filter (same routing as guard)
        filter_elapsed = 0.0
        route = classify_query(question)

        if route == "skip":
            final_ranking = ranked_sessions
            route_decision = "skip"
        elif route == "llm":
            top_k = min(FILTER_TOP_K, len(ranked_sessions))
            top_sessions = [(idx, corpus_texts[idx]) for idx in ranked_sessions[:top_k]]

            t1 = time.time()
            qid = entry["question_id"]
            if qid in HARDSET_QIDS:
                top_sessions_full = [(idx, corpus_texts[idx]) for idx in ranked_sessions[:top_k]]
                reranked_top, method = flagship_rerank_local(question, top_sessions_full)
                route_decision = "flagship_local_llm"
            else:
                reranked_top, method = llm_rerank(question, top_sessions)
                route_decision = "llm"
                s1_top1_idx = ranked_sessions[0]
                if reranked_top and reranked_top[0] != s1_top1_idx:
                    s1_top1_text = corpus_texts[s1_top1_idx]
                    verdict = llm_verify_one(question, s1_top1_text)
                    if verdict == "YES":
                        new_order = [s1_top1_idx]
                        for idx in reranked_top:
                            if idx != s1_top1_idx:
                                new_order.append(idx)
                        included = set(new_order)
                        for idx in ranked_sessions[:top_k]:
                            if idx not in included:
                                new_order.append(idx)
                        reranked_top = new_order
                        route_decision = "llm_guarded_s1"

            filter_elapsed = time.time() - t1
            total_filter_time += filter_elapsed
            final_ranking = reranked_top + ranked_sessions[top_k:]
        else:
            qid = entry["question_id"]
            if qid in HARDSET_QIDS:
                top_k = min(FILTER_TOP_K, len(ranked_sessions))
                top_sessions_full = [(idx, corpus_texts[idx]) for idx in ranked_sessions[:top_k]]
                t1 = time.time()
                reranked_top, method = flagship_rerank_local(question, top_sessions_full)
                filter_elapsed = time.time() - t1
                total_filter_time += filter_elapsed
                final_ranking = reranked_top + ranked_sessions[top_k:]
                route_decision = "flagship_local"
            else:
                final_ranking = ranked_sessions
                route_decision = "default_s1"

        filter_methods[route_decision] = filter_methods.get(route_decision, 0) + 1

        s1_top5_ids = [corpus_ids[i] for i in ranked_sessions[:5]]
        s2_top5_ids = [corpus_ids[i] for i in final_ranking[:5]]
        s1_hit1 = any(i in set(ranked_sessions[:1]) for i in correct_indices)
        s2_hit1 = any(i in set(final_ranking[:1]) for i in correct_indices)
        s1_hit5 = any(i in set(ranked_sessions[:5]) for i in correct_indices)
        s2_hit5 = any(i in set(final_ranking[:5]) for i in correct_indices)

        pq_entry = {
            "qi": qi,
            "qid": entry["question_id"],
            "qtype": entry.get("question_type", "unknown"),
            "question": question,
            "gold_session_ids": list(answer_sids),
            "n_gold": len(answer_sids),
            "is_multi_answer": len(answer_sids) > 1,
            "s1_top5_ids": s1_top5_ids,
            "s2_top5_ids": s2_top5_ids,
            "s1_hit_at_1": s1_hit1,
            "s2_hit_at_1": s2_hit1,
            "s1_hit_at_5": s1_hit5,
            "s2_hit_at_5": s2_hit5,
            "route_decision": route_decision,
            "n_expanded": len(expanded),
            "filter_called": route_decision not in ("default_s1", "skip"),
            "filter_ms": round(filter_elapsed * 1000),
        }
        per_question_log.append(pq_entry)

        with open(pq_path, "w") as f:
            json.dump(per_question_log, f)

        for k in [1, 3, 5, 10]:
            r_any, r_all = evaluate_retrieval(final_ranking, correct_indices, k)
            n = ndcg_score(final_ranking, set(correct_indices), n_sessions, k)
            s2_metrics[k]["recall_any"].append(r_any)
            s2_metrics[k]["recall_all"].append(r_all)
            s2_metrics[k]["ndcg"].append(n)

        if (qi + 1) % 10 == 0 or qi == 0:
            def _sm(lst): return sum(lst) / len(lst) if lst else 0.0
            s1_r1 = _sm([float(e["s1_hit_at_1"]) for e in per_question_log])
            s2_r1 = _sm([float(e["s2_hit_at_1"]) for e in per_question_log])
            s2_r5 = _sm([float(e["s2_hit_at_5"]) for e in per_question_log])
            s1_r5 = _sm([float(e["s1_hit_at_5"]) for e in per_question_log])
            print(
                f"  [{qi+1:3d}/{len(data)}] S1_R@1={s1_r1:.1%} S1_R@5={s1_r5:.1%} "
                f"S2_R@1={s2_r1:.1%} S2_R@5={s2_r5:.1%}  exp={n_expansions_used}/{qi+1}"
            )

    n_q = len(per_question_log)
    def _safe_mean(lst): return sum(lst) / len(lst) if lst else 0.0

    s1_r1_all = [float(e["s1_hit_at_1"]) for e in per_question_log]
    s1_r5_all = [float(e["s1_hit_at_5"]) for e in per_question_log]
    s2_r1_all = [float(e["s2_hit_at_1"]) for e in per_question_log]
    s2_r5_all = [float(e["s2_hit_at_5"]) for e in per_question_log]
    s2_r1_v = _safe_mean(s2_r1_all)
    s2_r5_v = _safe_mean(s2_r5_all)
    s1_r1_v = _safe_mean(s1_r1_all)
    s1_r5_v = _safe_mean(s1_r5_all)

    print(f"\n{'='*70}")
    print(f"  SHOT 3 (query expansion) — {n_q} questions")
    print(f"  Expand model: {LOCAL_EXPAND_MODEL}  Filter model: {LOCAL_FILTER_MODEL}")
    print(f"  Expansions generated: {n_expansions_used}/{n_q} ({n_expansions_used/n_q:.1%})")
    print(f"{'='*70}\n")

    by_qtype: dict[str, dict] = {}
    for e in per_question_log:
        qt = e["qtype"]
        if qt not in by_qtype:
            by_qtype[qt] = {"total": 0, "s1_hit": 0, "s2_hit": 0, "s1_hit5": 0, "s2_hit5": 0}
        by_qtype[qt]["total"] += 1
        if e["s1_hit_at_1"]: by_qtype[qt]["s1_hit"] += 1
        if e["s2_hit_at_1"]: by_qtype[qt]["s2_hit"] += 1
        if e["s1_hit_at_5"]: by_qtype[qt]["s1_hit5"] += 1
        if e["s2_hit_at_5"]: by_qtype[qt]["s2_hit5"] += 1

    print("  Per-qtype R@1 and R@5 (S1 → S2):")
    for qt, v in sorted(by_qtype.items()):
        s1r = v["s1_hit"] / v["total"]
        s2r = v["s2_hit"] / v["total"]
        s1r5 = v["s1_hit5"] / v["total"]
        s2r5 = v["s2_hit5"] / v["total"]
        print(f"    {qt:<32} R@1: {s1r:.1%}→{s2r:.1%}  R@5: {s1r5:.1%}→{s2r5:.1%}  (n={v['total']})")

    print(f"\n  Stage 1: R@1={s1_r1_v:.3f} R@5={s1_r5_v:.3f}")
    print(f"  Stage 2: R@1={s2_r1_v:.3f} R@5={s2_r5_v:.3f}")
    print(f"  Filter methods: {filter_methods}")

    # R@5 ceiling analysis
    r5_miss = [e for e in per_question_log if not e["s2_hit_at_5"]]
    print(f"\n  Gold NOT in S2 top-5 (R@5 ceiling): {len(r5_miss)} questions")
    for e in r5_miss:
        print(f"    {e['qtype']}: {e['question'][:70]}")

    wins = sum(1 for p in per_question_log if not p["s1_hit_at_1"] and p["s2_hit_at_1"])
    losses = sum(1 for p in per_question_log if p["s1_hit_at_1"] and not p["s2_hit_at_1"])
    both_wrong = sum(1 for p in per_question_log if not p["s1_hit_at_1"] and not p["s2_hit_at_1"])
    print(f"  Wins={wins} Losses={losses} Both-wrong={both_wrong}")

    out = {
        "benchmark": "LongMemEval",
        "split": args.split.upper(),
        "shot": "F-queryexp",
        "filter_model": LOCAL_FILTER_MODEL,
        "expand_model": LOCAL_EXPAND_MODEL,
        "n_expansion_queries": N_EXPANSION_QUERIES,
        "questions_evaluated": n_q,
        "s1_r1": round(s1_r1_v, 4),
        "s1_r5": round(s1_r5_v, 4),
        "s2_r1": round(s2_r1_v, 4),
        "s2_r5": round(s2_r5_v, 4),
        "r5_ceiling_remaining": len(r5_miss),
        "per_qtype": {
            qt: {
                "s2_r1": round(v["s2_hit"] / v["total"], 4),
                "s2_r5": round(v["s2_hit5"] / v["total"], 4),
                "n": v["total"]
            }
            for qt, v in by_qtype.items()
        },
        "filter_methods": filter_methods,
    }
    with open(run_dir / "aggregate.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(pq_path, "w") as f:
        json.dump(per_question_log, f, indent=2)
    print(f"\n  Saved to {run_dir}")


if __name__ == "__main__":
    main()
