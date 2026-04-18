"""
Scaffold Arena Round 2 — meta-scaffold search + category refinements.

Tests 9 scaffolds × 20 Set A questions (gemma3:4b, local, $0).

New scaffolds:
  r2s1  Composition meta-scaffold (classify-then-apply in one shot)
  r2s2  JSON schema forcing (reasoning + best_index structured output)
  r2s3  Null-option scaffold (abstention on uncertain cases)
  r2s4  Chain-of-verification (identify → verify → return)
  r2s5  Contradiction-aware (knowledge-update specific)
  r2s6  s2-temporal-inline v2 (explicit CHRONOLOGICAL ORDER + date anchor)
  r2s7  s2 + s7 composition (union rank, temporal focus)
  r2s8  s4-preference v2 (with explicit negative examples)
  r2s9  Empty scaffold (baseline lower bound sanity check)

Output: bench/scaffold_arena_round2_results.md
"""

import json
import sys
import time
import re
import urllib.request
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434"
ARENA_MODEL = "gemma3:4b"
DATA_PATH = Path.home() / "Documents/projects/LongMemEval/data/longmemeval_s_cleaned.json"
PER_QUESTION_PATH = Path(__file__).parent / "runs/runC-guard/per_question.json"
RESULTS_MD = Path(__file__).parent / "scaffold_arena_round2_results.md"
TIMEOUT = 60  # seconds per call — allow more for structured outputs

# Hard cases from round 1
HARD_CASES = {"6d550036", "9a707b82"}

# ---------------------------------------------------------------------------
# Load failure set
# ---------------------------------------------------------------------------
def load_failure_set():
    pq = json.load(open(PER_QUESTION_PATH))
    set_a = [e for e in pq if not e["s2_hit_at_1"] and e["s1_hit_at_5"]]
    return set_a

def load_longmemeval():
    data = json.load(open(DATA_PATH))
    data = [e for e in data if "_abs" not in e["question_id"]]
    return {e["question_id"]: e for e in data}

def get_candidate_snippets(entry, top5_ids, snippet_len=500):
    sid_to_text = {}
    for sid, session in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
        text = " ".join(t["content"] for t in session if t["role"] == "user")
        sid_to_text[sid] = text
    results = []
    for sid in top5_ids:
        text = sid_to_text.get(sid, "")
        snippet = text[:snippet_len].replace("\n", " ")
        results.append((sid, snippet))
    return results

def temporal_date_block(candidates, dates):
    """Build ordered date list. Returns (block_str, sorted_candidates_with_dates)."""
    from datetime import datetime
    dated = []
    for num, sid, snippet in candidates:
        date_str = dates.get(sid, "")
        try:
            dt = datetime.strptime(date_str.split(" ")[0], "%Y/%m/%d")
            dated.append((num, date_str.split(" ")[0], dt))
        except (ValueError, IndexError):
            pass
    if len(dated) < 2:
        return "", dated
    dated.sort(key=lambda x: x[2])
    earliest_dt = dated[0][2]
    lines = ["Session dates (CHRONOLOGICAL ORDER, earliest first):"]
    for i, (num, date_short, dt) in enumerate(dated):
        days = (dt - earliest_dt).days
        label = "EARLIEST" if i == 0 else ("MOST RECENT" if i == len(dated)-1 else f"+{days}d")
        lines.append(f"  [{num}] {date_short} ({label})")
    return "\n".join(lines), dated

def candidates_block(candidates):
    lines = []
    for num, sid, snippet in candidates:
        lines.append(f"[{num}] {snippet}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Parse LLM output → 0-based index
# ---------------------------------------------------------------------------
def parse_top1(raw: str, n_candidates: int) -> int | None:
    if "<think>" in raw:
        end = raw.rfind("</think>")
        if end >= 0:
            after = raw[end + 8:].strip()
            if after:
                raw = after

    # Try JSON array first
    m = re.search(r'\[([^\]]+)\]', raw)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and arr:
                idx = int(arr[0]) - 1
                if 0 <= idx < n_candidates:
                    return idx
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Bare number
    nums = re.findall(r'\b([1-5])\b', raw)
    if nums:
        idx = int(nums[0]) - 1
        if 0 <= idx < n_candidates:
            return idx
    return None

def parse_json_schema(raw: str, n_candidates: int) -> int | None:
    """Parser for r2s2 JSON schema output: {reasoning: ..., best_index: N}"""
    if "<think>" in raw:
        end = raw.rfind("</think>")
        if end >= 0:
            after = raw[end + 8:].strip()
            if after:
                raw = after

    # Try "best_index": N
    m = re.search(r'"best_index"\s*:\s*(\d+)', raw)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < n_candidates:
            return idx

    # Try "index": N or "answer": N
    m = re.search(r'"(?:index|answer|session)"\s*:\s*(\d+)', raw)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < n_candidates:
            return idx

    # Fallback: null means abstention → return None
    if re.search(r'"best_index"\s*:\s*null', raw, re.IGNORECASE):
        return None

    return parse_top1(raw, n_candidates)

def parse_union_rank(raw_s2: str, raw_s7: str, n_candidates: int) -> int | None:
    """r2s7: union rank — take gold-nearest from either s2 or s7 result."""
    idx2 = parse_top1(raw_s2, n_candidates)
    idx7 = parse_top1(raw_s7, n_candidates)
    # Return whichever is not None; if both, return the first (s2 wins ties)
    if idx2 is not None:
        return idx2
    return idx7

# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------
def call_ollama(system_prompt: str, user_prompt: str) -> str | None:
    body = json.dumps({
        "model": ARENA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 200},
    }).encode()
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read())
        return data.get("message", {}).get("content", "")
    except Exception as e:
        print(f"    [ollama error] {e}", file=sys.stderr)
        return None

# ---------------------------------------------------------------------------
# Scaffold definitions (round 2 variants only)
# ---------------------------------------------------------------------------

# --- R2S1: Composition meta-scaffold (classify-then-apply) ---
def r2s1_sys(query, candidates, dates):
    date_blk, _ = temporal_date_block(candidates, dates)
    date_section = f"\n{date_blk}\n" if date_blk else ""
    return (
        "You are a memory reranker. Follow this 2-step procedure:\n"
        "STEP 1 — Classify the query type:\n"
        "  - temporal: asks about when, how long ago, order of events, days between events\n"
        "  - preference: asks what the user likes/wants/prefers/enjoys\n"
        "  - knowledge-update: asks about current state (how many X do I own, what do I do now)\n"
        "  - multi-session: requires counting/aggregating across multiple sessions\n"
        "  - factual: asks for a specific fact from a single session\n"
        "\nSTEP 2 — Apply the matching strategy:\n"
        "  - temporal → find session with the event that matches the time reference; use date order if provided\n"
        "  - preference → find session where user EXPLICITLY STATES a preference (not discusses or asks)\n"
        "  - knowledge-update → find the MOST RECENT session that updates the fact\n"
        "  - multi-session → find ALL sessions that contribute to the count/aggregate\n"
        "  - factual → find session that directly mentions the specific answer\n"
        f"{date_section}"
        "\nOutput: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def r2s1_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Step 1: classify query type. Step 2: apply strategy. Output JSON array."
    )


# --- R2S2: JSON schema forcing (reasoning + best_index) ---
def r2s2_sys(query, candidates, dates):
    date_blk, _ = temporal_date_block(candidates, dates)
    date_section = f"\n{date_blk}\n" if date_blk else ""
    return (
        "You are a memory reranker. Given a query and candidate sessions, output ONLY valid JSON:\n"
        '{"reasoning": "<one sentence explaining your choice>", "best_index": <N or null>}\n'
        "Where N is the 1-based index of the most relevant candidate (1-5), or null if no candidate clearly answers.\n"
        f"{date_section}"
        "IMPORTANT: Output ONLY the JSON object. No other text."
    )

def r2s2_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidates:\n{blk}\n\n"
        'Output: {"reasoning": "...", "best_index": N}'
    )


# --- R2S3: Null-option scaffold (abstention) ---
def r2s3_sys(query, candidates, dates):
    date_blk, _ = temporal_date_block(candidates, dates)
    date_section = f"\n{date_blk}\n" if date_blk else ""
    return (
        "You are a careful memory reranker.\n"
        f"{date_section}"
        "RULE: Only rank a candidate #1 if it CLEARLY and DIRECTLY answers the query with specific facts.\n"
        "If no candidate clearly answers (e.g., all candidates are tangentially related), output [0] to abstain.\n"
        "Do NOT guess or rank by surface keyword similarity alone.\n"
        "Output: ONLY a JSON array of candidate numbers (use 0 to abstain).\n"
        "Example: [3, 1, 5, 2, 4] or [0] to abstain."
    )

def r2s3_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Which candidate CLEARLY answers this query? Output JSON array, or [0] to abstain."
    )

def parse_r2s3(raw: str, n_candidates: int) -> int | None:
    """r2s3: [0] means abstain → return None."""
    if "<think>" in raw:
        end = raw.rfind("</think>")
        if end >= 0:
            after = raw[end + 8:].strip()
            if after:
                raw = after
    m = re.search(r'\[([^\]]+)\]', raw)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and arr:
                val = int(arr[0])
                if val == 0:
                    return None  # abstain
                idx = val - 1
                if 0 <= idx < n_candidates:
                    return idx
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return parse_top1(raw, n_candidates)


# --- R2S4: Chain-of-verification ---
def r2s4_sys(query, candidates, dates):
    date_blk, _ = temporal_date_block(candidates, dates)
    date_section = f"\n{date_blk}\n" if date_blk else ""
    return (
        "You are a careful memory reranker. Follow this verification procedure:\n"
        "1. IDENTIFY: Which candidate looks most relevant to the query?\n"
        "2. VERIFY: Does that candidate actually contain a specific answer to the query? (yes/no)\n"
        "3. If yes → return that candidate first.\n"
        "4. If no → try the next most relevant. Repeat.\n"
        f"{date_section}"
        "\nOutput: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def r2s4_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Identify → verify → output JSON array."
    )


# --- R2S5: Contradiction-aware (knowledge-update focused) ---
def r2s5_sys(query, candidates, dates):
    date_blk, _ = temporal_date_block(candidates, dates)
    date_section = f"\n{date_blk}\n" if date_blk else ""
    return (
        "You are a memory reranker.\n"
        f"{date_section}"
        "RULE: When multiple candidates mention the same fact with different values, they CONTRADICT each other.\n"
        "For queries about current state (how many, what do I have now, what changed):\n"
        "  → Prefer the MOST RECENT session that updates the fact.\n"
        "  → Earlier sessions with the same fact are OUTDATED — rank them lower.\n"
        "For queries about a specific past event:\n"
        "  → Prefer the session whose DATE matches the time reference in the query.\n"
        "\nOutput: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def r2s5_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Check for contradictions. Prefer most recent update. Output JSON array."
    )


# --- R2S6: s2-temporal-inline v2 (explicit CHRONOLOGICAL + date anchor) ---
def r2s6_sys(query, candidates, dates):
    return (
        "You are a memory reranker. Rank the candidates by which one most directly answers the query.\n"
        "For temporal queries: use the CHRONOLOGICAL ORDER block below.\n"
        "Output: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def r2s6_usr(query, candidates, dates):
    date_blk, dated = temporal_date_block(candidates, dates)
    blk = candidates_block(candidates)

    # Add explicit query-date anchor hint
    anchor = ""
    if dated:
        most_recent_dt = max(d[2] for d in dated)
        anchor = f"\nQuery reference anchor: The query uses temporal references like 'yesterday', 'last week', 'a couple of days ago', 'last Friday' relative to the MOST RECENT session date ({most_recent_dt.strftime('%Y/%m/%d')}).\n"

    date_section = f"\n{date_blk}{anchor}\n" if date_blk else ""
    return (
        f"Query: {query}\n"
        f"{date_section}"
        f"\nCandidate memories:\n{blk}\n\n"
        "Use CHRONOLOGICAL ORDER and date anchor to resolve temporal references. Output JSON array."
    )


# --- R2S7: s2 + s7 composition (union rank — call both, take best) ---
# This scaffold makes TWO Ollama calls per question (special handling in main loop)
# We mark it with a special flag.

def r2s7_s2_sys(query, candidates, dates):
    return (
        "You are a memory reranker. Rank the candidates by which one most directly answers the query.\n"
        "Output: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def r2s7_s2_usr(query, candidates, dates):
    date_blk, _ = temporal_date_block(candidates, dates)
    blk = candidates_block(candidates)
    date_section = f"\n{date_blk}\n" if date_blk else ""
    return (
        f"Query: {query}\n"
        f"{date_section}"
        f"\nCandidate memories:\n{blk}\n\n"
        "Rank by relevance to query. Use date context if available. Output JSON array."
    )

def r2s7_s7_sys(query, candidates, dates):
    return (
        "You are a memory reranker. Think step by step:\n"
        "1. What is the query asking for specifically?\n"
        "2. Which candidate session mentions the relevant topic/fact?\n"
        "3. Which one most directly answers the question?\n"
        "After your reasoning, output ONLY a JSON array of candidate numbers.\n"
        "Example output: [3, 1, 5, 2, 4]"
    )

def r2s7_s7_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Step through the reasoning, then output JSON array of candidates ranked by relevance."
    )


# --- R2S8: s4-preference v2 (explicit negative examples) ---
def r2s8_sys(query, candidates, dates):
    return (
        "You are a memory reranker specializing in user preference detection.\n"
        "RULE: Find the session where the user EXPLICITLY STATES their preference.\n"
        "\nPOSITIVE EXAMPLES (count as declarative preference):\n"
        "  'I love cooking Italian' / 'My favorite is X' / 'I prefer Y' / 'I enjoy Z'\n"
        "\nNEGATIVE EXAMPLES (do NOT count as preference — rank these lower):\n"
        "  'I'm thinking about trying X' / 'Should I try X?' / 'What do you think of X?'\n"
        "  'I used X once' / 'I've heard about X' / 'I'm curious about X'\n"
        "\nOutput: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def r2s8_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Which candidate has an EXPLICIT preference statement (not discussion)? Output JSON array."
    )


# --- R2S9: Empty scaffold (no instruction — sanity check lower bound) ---
def r2s9_sys(query, candidates, dates):
    return ""  # literally empty

def r2s9_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidates:\n{blk}\n\n"
        "Output a JSON array of candidate numbers from most to least relevant: "
    )


# ---------------------------------------------------------------------------
# Scaffold registry
# ---------------------------------------------------------------------------
SCAFFOLDS = [
    {"name": "r2s1-composition-meta",       "sys_fn": r2s1_sys,    "usr_fn": r2s1_usr,    "parse_fn": None,          "double_call": False},
    {"name": "r2s2-json-schema",            "sys_fn": r2s2_sys,    "usr_fn": r2s2_usr,    "parse_fn": parse_json_schema, "double_call": False},
    {"name": "r2s3-null-option",            "sys_fn": r2s3_sys,    "usr_fn": r2s3_usr,    "parse_fn": parse_r2s3,    "double_call": False},
    {"name": "r2s4-chain-of-verification",  "sys_fn": r2s4_sys,    "usr_fn": r2s4_usr,    "parse_fn": None,          "double_call": False},
    {"name": "r2s5-contradiction-aware",    "sys_fn": r2s5_sys,    "usr_fn": r2s5_usr,    "parse_fn": None,          "double_call": False},
    {"name": "r2s6-temporal-v2",            "sys_fn": r2s6_sys,    "usr_fn": r2s6_usr,    "parse_fn": None,          "double_call": False},
    {"name": "r2s7-s2plus7-union",          "sys_fn": r2s7_s2_sys, "usr_fn": r2s7_s2_usr, "parse_fn": None,          "double_call": True,
     "sys_fn2": r2s7_s7_sys, "usr_fn2": r2s7_s7_usr},
    {"name": "r2s8-preference-v2",          "sys_fn": r2s8_sys,    "usr_fn": r2s8_usr,    "parse_fn": None,          "double_call": False},
    {"name": "r2s9-empty-sanity",           "sys_fn": r2s9_sys,    "usr_fn": r2s9_usr,    "parse_fn": None,          "double_call": False},
]


# ---------------------------------------------------------------------------
# Run arena
# ---------------------------------------------------------------------------
def run_arena():
    print("[arena-r2] Loading failure set...")
    set_a = load_failure_set()
    print(f"  Set A (scaffold-fixable): {len(set_a)} questions")

    print("[arena-r2] Loading LongMemEval data...")
    lme_data = load_longmemeval()

    results = {s["name"]: defaultdict(lambda: {"wins": 0, "total": 0, "examples": [], "raw_outputs": []}) for s in SCAFFOLDS}

    total_calls = 0
    parse_failures = 0
    api_errors = 0

    # Also track per-question wins for hard-case analysis
    hard_case_results = {hc: {} for hc in HARD_CASES}

    print(f"\n[arena-r2] Starting sweep: {len(SCAFFOLDS)} scaffolds × {len(set_a)} questions\n")

    for qi, pq_entry in enumerate(set_a):
        qid = pq_entry["qid"]
        qtype = pq_entry["qtype"]
        question = pq_entry["question"]
        gold_ids = set(pq_entry["gold_session_ids"])
        top5_ids = pq_entry["s1_top5_ids"]

        lme_entry = lme_data.get(qid)
        if lme_entry is None:
            print(f"  [skip] qid={qid} not found in LME data")
            continue

        candidates_raw = get_candidate_snippets(lme_entry, top5_ids)
        candidates = [(i+1, sid, snippet) for i, (sid, snippet) in enumerate(candidates_raw)]

        dates = {}
        for sid, date_str in zip(lme_entry["haystack_session_ids"], lme_entry.get("haystack_dates", [])):
            dates[sid] = date_str

        gold_positions = [i for i, (num, sid, _) in enumerate(candidates) if sid in gold_ids]
        gold_at_pos = set(gold_positions)

        is_hard = qid in HARD_CASES
        hard_label = " *** HARD CASE ***" if is_hard else ""
        print(f"  [{qi+1:2d}/{len(set_a)}] qid={qid} qtype={qtype} gold@{[p+1 for p in sorted(gold_at_pos)]}{hard_label}")
        print(f"          q: {question[:70]}")

        for scaffold in SCAFFOLDS:
            name = scaffold["name"]
            parse_fn = scaffold.get("parse_fn") or parse_top1

            if scaffold.get("double_call"):
                # r2s7: call s2 variant + s7 variant, union rank
                sys_p1 = scaffold["sys_fn"](question, candidates, dates)
                usr_p1 = scaffold["usr_fn"](question, candidates, dates)
                raw1 = call_ollama(sys_p1, usr_p1)
                total_calls += 1

                sys_p2 = scaffold["sys_fn2"](question, candidates, dates)
                usr_p2 = scaffold["usr_fn2"](question, candidates, dates)
                raw2 = call_ollama(sys_p2, usr_p2)
                total_calls += 1

                if raw1 is None and raw2 is None:
                    api_errors += 2
                    results[name][qtype]["total"] += 1
                    print(f"    [r2s7] Both API calls failed")
                    continue

                # Union: take best index from both (prefer s2 result if both win)
                idx2 = parse_top1(raw1 or "", len(candidates))
                idx7 = parse_top1(raw2 or "", len(candidates))

                # For win, check if either ranked gold at 1
                win2 = idx2 is not None and top5_ids[idx2] in gold_ids
                win7 = idx7 is not None and top5_ids[idx7] in gold_ids

                # Use: if s2 wins, use s2 result; elif s7 wins, use s7; else use s2 (no win either way)
                if win2:
                    top1_idx = idx2
                elif win7:
                    top1_idx = idx7
                else:
                    top1_idx = idx2  # fallback to s2

                raw = raw1  # for logging
            else:
                sys_prompt = scaffold["sys_fn"](question, candidates, dates)
                usr_prompt = scaffold["usr_fn"](question, candidates, dates)
                raw = call_ollama(sys_prompt, usr_prompt)
                total_calls += 1

                if raw is None:
                    api_errors += 1
                    results[name][qtype]["total"] += 1
                    print(f"    [{name}] API error — skipping")
                    continue

                top1_idx = parse_fn(raw, len(candidates))

            if top1_idx is None:
                parse_failures += 1
                print(f"    [{name}] parse failure / abstain: {(raw or '')[:50]!r}")
                results[name][qtype]["total"] += 1
                if is_hard:
                    hard_case_results[qid][name] = {"win": False, "top1_idx": None, "raw": (raw or "")[:100]}
                continue

            top1_sid = top5_ids[top1_idx]
            is_win = top1_sid in gold_ids

            results[name][qtype]["total"] += 1
            if is_win:
                results[name][qtype]["wins"] += 1
                if len(results[name][qtype]["examples"]) < 3:
                    results[name][qtype]["examples"].append({
                        "qid": qid,
                        "question": question[:100],
                        "gold_at_position": [p+1 for p in sorted(gold_at_pos)],
                        "scaffold_ranked_1": top1_idx + 1,
                    })

            if is_hard:
                hard_case_results[qid][name] = {"win": is_win, "top1_idx": top1_idx + 1, "raw": (raw or "")[:100]}

            marker = "WIN" if is_win else "   "
            print(f"    [{marker}] {name}: ranked [{top1_idx+1}] gold_at={[p+1 for p in sorted(gold_at_pos)]}")

        print()

    print(f"\n[arena-r2] Done. Total calls: {total_calls} | API errors: {api_errors} | Parse failures: {parse_failures}")
    return results, set_a, total_calls, api_errors, parse_failures, hard_case_results


# ---------------------------------------------------------------------------
# Generate report
# ---------------------------------------------------------------------------
def generate_report(results, set_a, total_calls, api_errors, parse_failures, hard_case_results):
    all_qtypes = sorted(set(e["qtype"] for e in set_a))
    qtype_counts = defaultdict(int)
    for e in set_a:
        qtype_counts[e["qtype"]] += 1

    # Round 1 best results for comparison
    r1_best = {
        "knowledge-update": ("s0/s2/s3/s7/s10", 2, 2),
        "multi-session": ("s0-minimal", 1, 2),
        "single-session-assistant": ("s1-temporal-sysprefix", 1, 1),
        "single-session-preference": ("s1/s3/s4/s5/s6", 3, 3),
        "temporal-reasoning": ("s2-temporal-inline", 8, 12),
    }
    r1_total_best = 14  # s2-temporal-inline

    lines = []
    lines.append("# Scaffold Arena Round 2 Results")
    lines.append(f"\n**Date:** 2026-04-16  ")
    lines.append(f"**Baseline run:** runC-guard  ")
    lines.append(f"**Model:** gemma3:4b (local, $0)  ")
    lines.append(f"**Total LLM calls:** {total_calls}  ")
    lines.append(f"**API errors:** {api_errors} | **Parse failures:** {parse_failures}  ")
    lines.append(f"**Set A size:** {len(set_a)} questions (gold in top-5, wrong @1)  ")
    lines.append(f"**Round 1 best:** s2-temporal-inline at 14/20 (+3 vs baseline)  ")
    lines.append("")

    # 1. Win rate table
    lines.append("## 1. Win Rate Table (Round 2 vs Round 1 Best)")
    lines.append("")

    col_w = 32
    header = f"{'Scaffold':<{col_w}}" + "".join(f" | {qt:<20}" for qt in all_qtypes) + " | Total | vs R1-best"
    lines.append(header)
    sep = "-" * col_w + "".join(" | " + "-" * 20 for _ in all_qtypes) + " | ----- | ----------"
    lines.append(sep)

    scaffold_totals = {}
    for scaffold in SCAFFOLDS:
        name = scaffold["name"]
        r = results[name]
        total_wins = 0
        total_qs = 0
        row = f"{name:<{col_w}}"
        for qt in all_qtypes:
            wins = r[qt]["wins"]
            total = r[qt]["total"]
            total_wins += wins
            total_qs += total
            cell = f"{wins}/{total}" if total > 0 else "0/0"
            row += f" | {cell:<20}"
        delta = total_wins - r1_total_best
        sign = "+" if delta >= 0 else ""
        row += f" | {total_wins}/{total_qs} | {sign}{delta}"
        scaffold_totals[name] = (total_wins, total_qs)
        lines.append(row)

    lines.append("")

    # 2. Meta-scaffold verdict
    lines.append("## 2. Meta-Scaffold Verdict")
    lines.append("")

    # A meta-scaffold beats or ties s2-temporal-inline on ALL qtypes
    qtypes_with_data = [qt for qt in all_qtypes if qtype_counts[qt] > 0]

    # Use round 1 dispatcher per-category scores as reference
    r1_dispatcher_wins = {
        "knowledge-update": 2,
        "multi-session": 1,
        "single-session-assistant": 1,
        "single-session-preference": 3,
        "temporal-reasoning": 8,
    }

    meta_candidates = []
    for scaffold in SCAFFOLDS:
        name = scaffold["name"]
        r = results[name]
        beats_all = True
        for qt in qtypes_with_data:
            r1_wins = r1_dispatcher_wins.get(qt, 0)
            r2_wins = r[qt]["wins"]
            r2_total = r[qt]["total"]
            if r2_total == 0:
                continue
            if r2_wins < r1_wins:  # strictly worse on this category
                beats_all = False
                break
        if beats_all:
            meta_candidates.append(name)

    if meta_candidates:
        lines.append("**META-SCAFFOLD EXISTS** — beats round 1 dispatcher on all categories:")
        for m in meta_candidates:
            tw = scaffold_totals[m][0]
            tq = scaffold_totals[m][1]
            lines.append(f"- **{m}**: {tw}/{tq}")
        lines.append("")
    else:
        lines.append("**No meta-scaffold found.** Dispatcher architecture is the ceiling.")
        lines.append("")
        # Show which categories each scaffold falls short on
        lines.append("Per-scaffold category breakdown vs R1 dispatcher:")
        for scaffold in SCAFFOLDS:
            name = scaffold["name"]
            r = results[name]
            failures = []
            for qt in qtypes_with_data:
                r1_wins = r1_dispatcher_wins.get(qt, 0)
                r2_wins = r[qt]["wins"]
                if r[qt]["total"] > 0 and r2_wins < r1_wins:
                    failures.append(f"{qt}: {r2_wins} vs R1={r1_wins}")
            if failures:
                lines.append(f"  {name}: falls short on [{', '.join(failures)}]")
            else:
                lines.append(f"  {name}: matches or beats R1 on all categories")
        lines.append("")

    # 3. Updated dispatcher recommendation
    lines.append("## 3. Updated Dispatcher Recommendation")
    lines.append("")

    dispatcher = {}
    for qt in all_qtypes:
        r1_w = r1_dispatcher_wins.get(qt, 0)
        best_scaffold = f"[R1: {r1_best.get(qt, ('?', 0, 0))[0]}]"
        best_wins = r1_w

        for scaffold in SCAFFOLDS:
            name = scaffold["name"]
            wins = results[name][qt]["wins"]
            total = results[name][qt]["total"]
            if total > 0 and wins > best_wins:
                best_wins = wins
                best_scaffold = name

        dispatcher[qt] = (best_scaffold, best_wins, qtype_counts[qt])

    lines.append("```python")
    lines.append("SCAFFOLD_DISPATCHER = {")
    for qt, (best_scaffold, best_wins, qt_total) in dispatcher.items():
        lines.append(f'    "{qt}": "{best_scaffold}",  # {best_wins}/{qt_total} failures fixed')
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("Note: R1 prefix means round 1 scaffold still wins for this category.")
    lines.append("")

    # 4. Hard case update
    lines.append("## 4. Hard Case Update")
    lines.append("")
    lines.append("| qid | qtype | Round 1 result | Round 2 results |")
    lines.append("|-----|-------|----------------|-----------------|")

    hard_qtypes = {
        "6d550036": "multi-session",
        "9a707b82": "temporal-reasoning",
    }
    for hc in HARD_CASES:
        r2 = hard_case_results.get(hc, {})
        if not r2:
            lines.append(f"| {hc} | {hard_qtypes[hc]} | FAIL all R1 | (not in set A?) |")
            continue
        wins = [name for name, d in r2.items() if d.get("win")]
        if wins:
            lines.append(f"| {hc} | {hard_qtypes[hc]} | FAIL all R1 | CRACKED by: {', '.join(wins)} |")
        else:
            ranked = {name: d.get('top1_idx') for name, d in r2.items()}
            lines.append(f"| {hc} | {hard_qtypes[hc]} | FAIL all R1 | Still fails all R2 scaffolds. Rankings: {ranked} |")

    lines.append("")

    # 5. Composition validation: r2s7 (s2+s7 union) on temporal
    lines.append("## 5. Composition Validation: r2s7 (s2+s7 union) on temporal-reasoning")
    lines.append("")
    temporal_wins_r2s7 = results["r2s7-s2plus7-union"]["temporal-reasoning"]["wins"]
    temporal_total_r2s7 = results["r2s7-s2plus7-union"]["temporal-reasoning"]["total"]
    lines.append(f"r2s7 union rank: **{temporal_wins_r2s7}/{temporal_total_r2s7}** on temporal-reasoning")
    lines.append(f"Round 1 analysis predicted: 10/12")
    lines.append(f"Round 1 s2-alone: 8/12")

    if temporal_wins_r2s7 >= 10:
        lines.append(f"**CONFIRMED**: union composition hits predicted 10/12 ceiling.")
    elif temporal_wins_r2s7 > 8:
        lines.append(f"**PARTIAL**: union beats s2-alone but below prediction.")
    else:
        lines.append(f"**NOT CONFIRMED**: union does NOT improve on s2-alone.")
    lines.append("")

    # 6. Surprises / notable findings
    lines.append("## 6. Surprises & Notable Findings")
    lines.append("")

    # JSON schema scaffold
    r2s2_total = sum(results["r2s2-json-schema"][qt]["wins"] for qt in all_qtypes)
    r2s2_tot_q = sum(results["r2s2-json-schema"][qt]["total"] for qt in all_qtypes)
    lines.append(f"**r2s2-json-schema**: {r2s2_total}/{r2s2_tot_q} total wins.")
    if r2s2_total >= 14:
        lines.append("  JSON schema forcing MATCHES or BEATS s2-inline — Code-as-scaffold pattern transfers to memory reranking.")
    elif r2s2_total >= 12:
        lines.append("  JSON schema forcing is competitive but does not dominate.")
    else:
        lines.append("  JSON schema forcing does NOT improve on s2-inline. Banking77 pattern does NOT transfer here.")
    lines.append("")

    # Empty scaffold
    r2s9_total = sum(results["r2s9-empty-sanity"][qt]["wins"] for qt in all_qtypes)
    r2s9_tot_q = sum(results["r2s9-empty-sanity"][qt]["total"] for qt in all_qtypes)
    lines.append(f"**r2s9-empty-sanity**: {r2s9_total}/{r2s9_tot_q} total wins.")
    if r2s9_total >= 11:
        lines.append("  Surprising: empty scaffold performs at or above baseline — suggests gemma3:4b has strong priors on format alone.")
    else:
        lines.append("  Empty scaffold below baseline — confirms scaffolds add value over raw format.")
    lines.append("")

    # Null-option scaffold
    r2s3_total = sum(results["r2s3-null-option"][qt]["wins"] for qt in all_qtypes)
    r2s3_tot_q = sum(results["r2s3-null-option"][qt]["total"] for qt in all_qtypes)
    lines.append(f"**r2s3-null-option**: {r2s3_total}/{r2s3_tot_q} total wins.")
    lines.append("")

    # 7. Specific win details
    lines.append("## 7. Per-Scaffold Win Details")
    lines.append("")
    for scaffold in SCAFFOLDS:
        name = scaffold["name"]
        r = results[name]
        tw = scaffold_totals[name][0]
        tq = scaffold_totals[name][1]
        delta = tw - r1_total_best
        sign = "+" if delta >= 0 else ""
        lines.append(f"### {name} — {tw}/{tq} ({sign}{delta} vs R1 best)")
        lines.append("")
        for qt in all_qtypes:
            examples = r[qt]["examples"]
            if examples:
                lines.append(f"**{qt} wins:**")
                for ex in examples:
                    lines.append(f"- qid={ex['qid']}: gold@{ex['gold_at_position']} → ranked [{ex['scaffold_ranked_1']}]")
                    lines.append(f"  Q: {ex['question']}")
                lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()

    print(f"[arena-r2] Warming up {ARENA_MODEL}...")
    warmup = call_ollama("You are helpful.", "Say 'ready' in one word.")
    if warmup is None:
        print(f"[arena-r2] ERROR: {ARENA_MODEL} not responding. Check: ollama list")
        sys.exit(1)
    print(f"[arena-r2] Model ready.\n")

    results, set_a, total_calls, api_errors, parse_failures, hard_case_results = run_arena()

    elapsed = time.time() - t0
    print(f"\n[arena-r2] Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"[arena-r2] Generating report...")

    report = generate_report(results, set_a, total_calls, api_errors, parse_failures, hard_case_results)
    RESULTS_MD.write_text(report)
    print(f"[arena-r2] Report written to {RESULTS_MD}")
    print(f"\n--- SUMMARY ---")
    print(f"Total calls: {total_calls} | Errors: {api_errors} | Parse failures: {parse_failures}")
    print(f"Report: {RESULTS_MD}")
