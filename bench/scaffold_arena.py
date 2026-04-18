"""
Scaffold Arena — failure-set-only scaffold sweep.

Target: Set A from runC-guard — questions where gold is in top-5 (s1_hit_at_5=True)
        but stage-2 ranks it wrong (s2_hit_at_1=False).

Tests 10 scaffold variants × Set A questions × 1 call each.
Uses gemma3:4b via Ollama (local, $0 cost).

Output: bench/scaffold_arena_results.md
"""

import json
import sys
import time
import urllib.request
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434"
ARENA_MODEL = "gemma3:4b"   # fast local, no DASHSCOPE needed
DATA_PATH = Path.home() / "Documents/projects/LongMemEval/data/longmemeval_s_cleaned.json"
PER_QUESTION_PATH = Path(__file__).parent / "runs/runC-guard/per_question.json"
RESULTS_MD = Path(__file__).parent / "scaffold_arena_results.md"
TIMEOUT = 45  # seconds per call

# ---------------------------------------------------------------------------
# Load failure set
# ---------------------------------------------------------------------------
def load_failure_set():
    pq = json.load(open(PER_QUESTION_PATH))
    set_a = [e for e in pq if not e["s2_hit_at_1"] and e["s1_hit_at_5"]]
    set_b = [e for e in pq if not e["s1_hit_at_5"]]
    return set_a, set_b

# ---------------------------------------------------------------------------
# Load LongMemEval data indexed by question_id
# ---------------------------------------------------------------------------
def load_longmemeval():
    data = json.load(open(DATA_PATH))
    data = [e for e in data if "_abs" not in e["question_id"]]
    return {e["question_id"]: e for e in data}

# ---------------------------------------------------------------------------
# Build session text snippets for candidates (500 chars each, same as pipeline)
# ---------------------------------------------------------------------------
def get_candidate_snippets(entry, top5_ids, snippet_len=500):
    """Returns list of (session_id, snippet_text) for top5_ids."""
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

# ---------------------------------------------------------------------------
# Parse LLM output → 0-based index of top-ranked candidate
# ---------------------------------------------------------------------------
def parse_top1(raw: str, n_candidates: int) -> int | None:
    """Return 0-based index of the LLM's top-ranked candidate. None on failure."""
    # Strip thinking tokens
    if "<think>" in raw:
        end = raw.rfind("</think>")
        if end >= 0:
            after = raw[end + 8:].strip()
            if after:
                raw = after

    import re
    # Try JSON array first: [3, 1, 5, 2, 4]
    m = re.search(r'\[([^\]]+)\]', raw)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and arr:
                idx = int(arr[0]) - 1  # 1-based → 0-based
                if 0 <= idx < n_candidates:
                    return idx
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Try bare number
    nums = re.findall(r'\b([1-5])\b', raw)
    if nums:
        idx = int(nums[0]) - 1
        if 0 <= idx < n_candidates:
            return idx

    return None

# ---------------------------------------------------------------------------
# Call local Ollama model
# ---------------------------------------------------------------------------
def call_ollama(system_prompt: str, user_prompt: str) -> str | None:
    body = json.dumps({
        "model": ARENA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 100},
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
# Scaffold definitions
# ---------------------------------------------------------------------------
# Each scaffold is (name, build_system_fn, build_user_fn)
# build_system_fn(query, candidates, dates) → system prompt string
# build_user_fn(query, candidates, dates) → user prompt string
# candidates: list of (1-based-num, session_id, snippet_text)
# dates: dict {session_id: date_string} (may be empty)

def candidates_block(candidates):
    lines = []
    for num, sid, snippet in candidates:
        lines.append(f"[{num}] {snippet}")
    return "\n".join(lines)

def temporal_date_block(candidates, dates):
    """Build ordered date list for candidates."""
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
        return ""
    dated.sort(key=lambda x: x[2])
    earliest_dt = dated[0][2]
    lines = ["Session dates (ordered chronologically):"]
    for i, (num, date_short, dt) in enumerate(dated):
        days = (dt - earliest_dt).days
        suffix = " (earliest)" if i == 0 else f" (+{days}d)" if i < len(dated)-1 else f" (+{days}d, most recent)"
        lines.append(f"  [{num}] {date_short}{suffix}")
    return "\n".join(lines)


# --- S0: Minimal baseline (current _FILTER_SYSTEM from pipeline) ---
def s0_minimal_sys(query, candidates, dates):
    return (
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

def s0_minimal_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return f"Query: {query}\n\nCandidate memories:\n{blk}\n\nRank these candidates by relevance. Output JSON array."


# --- S1: Temporal sysprefix — dates in system prompt + ordering instruction ---
def s1_temporal_sys(query, candidates, dates):
    date_block = temporal_date_block(candidates, dates)
    prefix = ""
    if date_block:
        prefix = f"\n{date_block}\n\nIMPORTANT: Use the date ordering above to resolve which session is earlier or later. The correct answer session is the one whose content matches the temporal reference in the query (e.g., 'last Friday' = most recent session dated on a Friday).\n"
    return (
        "You are a memory reranker. Given a query and candidate memory sessions, return the candidate most likely to contain the specific answer.\n"
        f"{prefix}"
        "Output: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def s1_temporal_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return f"Query: {query}\n\nCandidate memories:\n{blk}\n\nWhich candidate contains the answer? Output JSON array."


# --- S2: Temporal inline — date block injected into user message ---
def s2_temporal_inline_sys(query, candidates, dates):
    return (
        "You are a memory reranker. Rank the candidates by which one most directly answers the query.\n"
        "Output: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def s2_temporal_inline_usr(query, candidates, dates):
    date_block = temporal_date_block(candidates, dates)
    blk = candidates_block(candidates)
    date_section = f"\n{date_block}\n" if date_block else ""
    return (
        f"Query: {query}\n"
        f"{date_section}"
        f"\nCandidate memories:\n{blk}\n\n"
        "Rank by relevance to query. Use date context if available. Output JSON array."
    )


# --- S3: Contrastive-NOT temporal ("newer session is more relevant for updates") ---
def s3_contrastive_not_sys(query, candidates, dates):
    date_block = temporal_date_block(candidates, dates)
    prefix = ""
    if date_block:
        prefix = f"\n{date_block}\n"
    return (
        "You are a memory reranker.\n"
        f"{prefix}"
        "IMPORTANT DISTINCTIONS:\n"
        "- 'most recent' is NOT 'most relevant' — the most recent session is only correct if the query asks about a recent event.\n"
        "- 'older session' is NOT 'wrong session' — if the query asks about a past event, the older session may be correct.\n"
        "- 'knowledge update' queries (e.g., 'how many do I own now') need the LATEST session that changed the fact.\n"
        "- 'temporal counting' queries (e.g., 'how many days between X and Y') need ALL sessions that mention the events.\n"
        "\nOutput: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def s3_contrastive_not_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return f"Query: {query}\n\nCandidate memories:\n{blk}\n\nRank by relevance. Output JSON array."


# --- S4: Declarative preference scaffold ---
def s4_pref_declarative_sys(query, candidates, dates):
    return (
        "You are a memory reranker specializing in user preference detection.\n"
        "RULE: One of these sessions contains a DECLARATIVE PREFERENCE statement — where the user explicitly states what they like, want, or prefer.\n"
        "Declarative preferences look like: 'I like X', 'I prefer Y', 'my favorite is Z', 'I enjoy X', 'I love X'.\n"
        "Do NOT rank sessions where the user merely discusses, asks about, or considers a topic — only rank highest the session where they STATE their preference.\n"
        "Output: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def s4_pref_declarative_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Which candidate contains the user's EXPLICIT preference statement? Output JSON array."
    )


# --- S5: Multi-session aggregation ---
def s5_multi_session_sys(query, candidates, dates):
    return (
        "You are a memory reranker for multi-session queries.\n"
        "The user's question may require aggregating information from MULTIPLE sessions.\n"
        "RULE: Return the session(s) most likely to contain the specific facts that answer this query.\n"
        "For counting queries (e.g., 'how many X have I done'), ALL sessions that mention X are relevant.\n"
        "Rank the session with the MOST relevant facts first.\n"
        "Output: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def s5_multi_session_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Which session(s) contain facts that answer this query? Rank all, most relevant first. Output JSON array."
    )


# --- S6: Code-as-scaffold (JSON schema output, which session_id answers) ---
def s6_code_scaffold_sys(query, candidates, dates):
    return (
        "You are a memory reranker. You will receive a query and numbered candidate sessions.\n"
        "Determine which candidate most directly answers the query.\n"
        "Think step-by-step silently, then output ONLY this JSON:\n"
        '{"top": [N1, N2, N3, N4, N5]}\n'
        "Where N1 is the most relevant candidate number (1-5), N5 is least relevant."
    )

def s6_code_scaffold_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidates:\n{blk}\n\n"
        'Output: {"top": [N1, N2, N3, N4, N5]}'
    )

def parse_top1_code_scaffold(raw: str, n_candidates: int) -> int | None:
    """Special parser for S6 JSON schema output."""
    import re
    # Try {"top": [...]}
    m = re.search(r'"top"\s*:\s*\[([^\]]+)\]', raw)
    if m:
        nums = re.findall(r'\d+', m.group(1))
        if nums:
            idx = int(nums[0]) - 1
            if 0 <= idx < n_candidates:
                return idx
    return parse_top1(raw, n_candidates)


# --- S7: Chain-of-thought ---
def s7_cot_sys(query, candidates, dates):
    return (
        "You are a memory reranker. Think step by step:\n"
        "1. What is the query asking for specifically?\n"
        "2. Which candidate session mentions the relevant topic/fact?\n"
        "3. Which one most directly answers the question?\n"
        "After your reasoning, output ONLY a JSON array of candidate numbers.\n"
        "Example output: [3, 1, 5, 2, 4]"
    )

def s7_cot_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Step through the reasoning, then output JSON array of candidates ranked by relevance."
    )


# --- S8: Socratic (identify type first, then match) ---
def s8_socratic_sys(query, candidates, dates):
    return (
        "You are a memory reranker. Follow this procedure:\n"
        "Step 1: Identify the query type — is it temporal (when/how long ago), preference (what I like), knowledge-update (current state), or factual?\n"
        "Step 2: Based on type, identify which candidate session contains the matching fact.\n"
        "  - temporal → find session with the date/event that matches the time reference\n"
        "  - preference → find session where user STATES a preference\n"
        "  - knowledge-update → find the MOST RECENT session that changed the fact\n"
        "  - factual → find session that directly mentions the fact\n"
        "Step 3: Output ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def s8_socratic_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Identify type, then rank. Output JSON array."
    )


# --- S9: Hypothesis scaffold (generate 3 hypotheses, pick best) ---
def s9_hypothesis_sys(query, candidates, dates):
    return (
        "You are a memory reranker.\n"
        "Generate 3 hypotheses about which candidate answers the query, then pick the best.\n"
        "Format:\n"
        "H1: [candidate N] because [reason]\n"
        "H2: [candidate N] because [reason]\n"
        "H3: [candidate N] because [reason]\n"
        "Best: [N]\n"
        "Final ranking: [N1, N2, N3, N4, N5]\n"
        "Output ONLY the final JSON array at the end."
    )

def s9_hypothesis_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Generate hypotheses, pick best, then output JSON array ranking."
    )


# --- S10: Anti-confidence (top result may be wrong, reconsider) ---
def s10_anti_confidence_sys(query, candidates, dates):
    date_block = temporal_date_block(candidates, dates)
    prefix = f"\n{date_block}\n" if date_block else ""
    return (
        "You are a careful memory reranker.\n"
        f"{prefix}"
        "WARNING: The first candidate presented is often retrieved by keyword match alone and may NOT contain the actual answer.\n"
        "Do NOT default to candidate [1]. Carefully check which candidate actually answers the query.\n"
        "Output: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def s10_anti_confidence_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Do NOT default to [1]. Which candidate actually answers the query? Output JSON array."
    )


# ---------------------------------------------------------------------------
# Scaffold registry
# ---------------------------------------------------------------------------
SCAFFOLDS = [
    {
        "name": "s0-minimal-baseline",
        "sys_fn": s0_minimal_sys,
        "usr_fn": s0_minimal_usr,
        "parse_fn": None,
    },
    {
        "name": "s1-temporal-sysprefix",
        "sys_fn": s1_temporal_sys,
        "usr_fn": s1_temporal_usr,
        "parse_fn": None,
    },
    {
        "name": "s2-temporal-inline",
        "sys_fn": s2_temporal_inline_sys,
        "usr_fn": s2_temporal_inline_usr,
        "parse_fn": None,
    },
    {
        "name": "s3-contrastive-not",
        "sys_fn": s3_contrastive_not_sys,
        "usr_fn": s3_contrastive_not_usr,
        "parse_fn": None,
    },
    {
        "name": "s4-declarative-preference",
        "sys_fn": s4_pref_declarative_sys,
        "usr_fn": s4_pref_declarative_usr,
        "parse_fn": None,
    },
    {
        "name": "s5-multi-session-aggregation",
        "sys_fn": s5_multi_session_sys,
        "usr_fn": s5_multi_session_usr,
        "parse_fn": None,
    },
    {
        "name": "s6-code-scaffold",
        "sys_fn": s6_code_scaffold_sys,
        "usr_fn": s6_code_scaffold_usr,
        "parse_fn": parse_top1_code_scaffold,
    },
    {
        "name": "s7-chain-of-thought",
        "sys_fn": s7_cot_sys,
        "usr_fn": s7_cot_usr,
        "parse_fn": None,
    },
    {
        "name": "s8-socratic",
        "sys_fn": s8_socratic_sys,
        "usr_fn": s8_socratic_usr,
        "parse_fn": None,
    },
    {
        "name": "s9-hypothesis",
        "sys_fn": s9_hypothesis_sys,
        "usr_fn": s9_hypothesis_usr,
        "parse_fn": None,
    },
    {
        "name": "s10-anti-confidence",
        "sys_fn": s10_anti_confidence_sys,
        "usr_fn": s10_anti_confidence_usr,
        "parse_fn": None,
    },
]


# ---------------------------------------------------------------------------
# Run arena
# ---------------------------------------------------------------------------
def run_arena():
    print("[arena] Loading failure set...")
    set_a, set_b = load_failure_set()
    print(f"  Set A (scaffold-fixable): {len(set_a)} questions")
    print(f"  Set B (retrieval failures, skip): {len(set_b)} questions")

    print("[arena] Loading LongMemEval data...")
    lme_data = load_longmemeval()

    # Results: {scaffold_name: {qtype: {"wins": 0, "total": 0, "examples": []}}}
    results = {s["name"]: defaultdict(lambda: {"wins": 0, "total": 0, "examples": []}) for s in SCAFFOLDS}

    total_calls = 0
    parse_failures = 0
    api_errors = 0

    print(f"\n[arena] Starting sweep: {len(SCAFFOLDS)} scaffolds × {len(set_a)} questions\n")

    for qi, pq_entry in enumerate(set_a):
        qid = pq_entry["qid"]
        qtype = pq_entry["qtype"]
        question = pq_entry["question"]
        gold_ids = set(pq_entry["gold_session_ids"])
        top5_ids = pq_entry["s1_top5_ids"]

        # Get LME entry for this question
        lme_entry = lme_data.get(qid)
        if lme_entry is None:
            print(f"  [skip] qid={qid} not found in LME data")
            continue

        # Build candidate snippets
        candidates_raw = get_candidate_snippets(lme_entry, top5_ids)
        candidates = [(i+1, sid, snippet) for i, (sid, snippet) in enumerate(candidates_raw)]

        # Build date lookup
        dates = {}
        for sid, date_str in zip(lme_entry["haystack_session_ids"], lme_entry.get("haystack_dates", [])):
            dates[sid] = date_str

        # Is gold at position? (0-indexed)
        gold_positions = [i for i, (num, sid, _) in enumerate(candidates) if sid in gold_ids]
        gold_at_pos = set(gold_positions)

        print(f"  [{qi+1:2d}/{len(set_a)}] qid={qid} qtype={qtype} gold@{[p+1 for p in sorted(gold_at_pos)]}")
        print(f"          q: {question[:70]}")

        for scaffold in SCAFFOLDS:
            name = scaffold["name"]
            sys_prompt = scaffold["sys_fn"](question, candidates, dates)
            usr_prompt = scaffold["usr_fn"](question, candidates, dates)
            parse_fn = scaffold["parse_fn"] or parse_top1

            raw = call_ollama(sys_prompt, usr_prompt)
            total_calls += 1

            if raw is None:
                api_errors += 1
                print(f"    [{name}] API error — skipping")
                results[name][qtype]["total"] += 1
                continue

            top1_idx = parse_fn(raw, len(candidates))

            if top1_idx is None:
                parse_failures += 1
                print(f"    [{name}] parse failure: {raw[:50]!r}")
                results[name][qtype]["total"] += 1
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

            marker = "WIN" if is_win else "   "
            print(f"    [{marker}] {name}: ranked [{top1_idx+1}] gold_at={[p+1 for p in sorted(gold_at_pos)]}")

        print()

    print(f"\n[arena] Done. Total calls: {total_calls} | API errors: {api_errors} | Parse failures: {parse_failures}")
    return results, set_a, total_calls, api_errors, parse_failures


# ---------------------------------------------------------------------------
# Generate markdown report
# ---------------------------------------------------------------------------
def generate_report(results, set_a, total_calls, api_errors, parse_failures):
    # Collect all qtypes
    all_qtypes = sorted(set(e["qtype"] for e in set_a))
    # Counts per qtype
    qtype_counts = defaultdict(int)
    for e in set_a:
        qtype_counts[e["qtype"]] += 1

    lines = []
    lines.append("# Scaffold Arena Results")
    lines.append(f"\n**Date:** 2026-04-16  ")
    lines.append(f"**Baseline run:** runC-guard  ")
    lines.append(f"**Model:** gemma3:4b (local, $0)  ")
    lines.append(f"**Total LLM calls:** {total_calls}  ")
    lines.append(f"**API errors:** {api_errors} | **Parse failures:** {parse_failures}  ")
    lines.append(f"**Set A size:** {len(set_a)} questions (gold in top-5, wrong @1)  ")
    lines.append("")

    # 1. Win rate table
    lines.append("## 1. Win Rate Per Scaffold Per qtype")
    lines.append("")

    # Header
    header_qtypes = all_qtypes
    col_w = 32
    header = f"{'Scaffold':<{col_w}}" + "".join(f" | {qt:<20}" for qt in header_qtypes) + " | Total"
    lines.append(header)
    sep = "-" * col_w + "".join(" | " + "-" * 20 for _ in header_qtypes) + " | -----"
    lines.append(sep)

    # Baseline wins for comparison
    baseline_results = results["s0-minimal-baseline"]

    scaffold_totals = {}
    for scaffold in SCAFFOLDS:
        name = scaffold["name"]
        r = results[name]
        total_wins = 0
        total_qs = 0
        row = f"{name:<{col_w}}"
        for qt in header_qtypes:
            wins = r[qt]["wins"]
            total = r[qt]["total"]
            total_wins += wins
            total_qs += total
            cell = f"{wins}/{total}" if total > 0 else "0/0"
            row += f" | {cell:<20}"
        row += f" | {total_wins}/{total_qs}"
        scaffold_totals[name] = (total_wins, total_qs)
        lines.append(row)

    lines.append("")

    # 2. Specific wins
    lines.append("## 2. Specific Wins vs Baseline")
    lines.append("")

    baseline_total_wins = sum(results["s0-minimal-baseline"][qt]["wins"] for qt in all_qtypes)
    lines.append(f"**Baseline (s0-minimal) total wins: {baseline_total_wins}/{len(set_a)}**")
    lines.append("")

    for scaffold in SCAFFOLDS:
        name = scaffold["name"]
        if name == "s0-minimal-baseline":
            continue
        r = results[name]
        total_wins = scaffold_totals[name][0]
        total_qs = scaffold_totals[name][1]
        delta = total_wins - baseline_total_wins
        sign = "+" if delta >= 0 else ""
        lines.append(f"### {name} — {total_wins}/{total_qs} ({sign}{delta} vs baseline)")
        lines.append("")

        # Show examples where this scaffold wins
        for qt in all_qtypes:
            examples = r[qt]["examples"]
            if examples:
                lines.append(f"**{qt} wins:**")
                for ex in examples:
                    lines.append(f"- qid={ex['qid']}: gold@{ex['gold_at_position']} → scaffold ranked [{ex['scaffold_ranked_1']}]")
                    lines.append(f"  Q: {ex['question']}")
                lines.append("")

    # 3. Dispatcher recommendation
    lines.append("## 3. Recommended Scaffold Dispatcher")
    lines.append("")
    lines.append("Based on win rates, recommended scaffold per qtype:")
    lines.append("")

    dispatcher = {}
    for qt in all_qtypes:
        best_scaffold = "s0-minimal-baseline"
        best_wins = results["s0-minimal-baseline"][qt]["wins"]
        best_total = results["s0-minimal-baseline"][qt]["total"]

        for scaffold in SCAFFOLDS:
            name = scaffold["name"]
            if name == "s0-minimal-baseline":
                continue
            wins = results[name][qt]["wins"]
            total = results[name][qt]["total"]
            if total > 0 and wins > best_wins:
                best_wins = wins
                best_total = total
                best_scaffold = name

        dispatcher[qt] = (best_scaffold, best_wins, best_total)
        lines.append(f"```")
        lines.append(f"qtype={qt} → {best_scaffold} ({best_wins}/{best_total} failures fixed)")
        lines.append(f"```")
        lines.append("")

    # 4. Meta-scaffold check
    lines.append("## 4. Meta-Scaffold Candidate")
    lines.append("")

    # A meta-scaffold wins across ALL qtypes (at least +1 vs baseline on each qtype with data)
    qtypes_with_data = [qt for qt in all_qtypes if qtype_counts[qt] > 0]
    meta_candidates = []

    for scaffold in SCAFFOLDS:
        name = scaffold["name"]
        if name == "s0-minimal-baseline":
            continue
        beats_all = True
        for qt in qtypes_with_data:
            baseline_wins = results["s0-minimal-baseline"][qt]["wins"]
            scaffold_wins = results[name][qt]["wins"]
            scaffold_total = results[name][qt]["total"]
            if scaffold_total == 0:
                continue  # no data
            if scaffold_wins <= baseline_wins:
                beats_all = False
                break
        if beats_all:
            meta_candidates.append(name)

    if meta_candidates:
        lines.append(f"**META-SCAFFOLD FOUND (beats baseline on all qtypes):**")
        for m in meta_candidates:
            total_wins = scaffold_totals[m][0]
            total_qs = scaffold_totals[m][1]
            lines.append(f"- **{m}** ({total_wins}/{total_qs})")
        lines.append("")
        lines.append("This is a candidate for the unified meta-scaffold in v0.3.1.")
    else:
        lines.append("**No single scaffold beats baseline on ALL qtypes.** Dispatcher approach recommended.")
    lines.append("")

    # 5. Composition candidates
    lines.append("## 5. Composition Candidates")
    lines.append("")
    lines.append("Scaffolds that win on DIFFERENT subsets of questions within the same qtype can be composed (try both, take max):")
    lines.append("")

    for qt in all_qtypes:
        qt_total = qtype_counts[qt]
        if qt_total == 0:
            continue
        winners_for_qt = []
        for scaffold in SCAFFOLDS:
            name = scaffold["name"]
            wins = results[name][qt]["wins"]
            total = results[name][qt]["total"]
            if wins > 0 and total > 0:
                winners_for_qt.append((name, wins, total, results[name][qt]["examples"]))

        if len(winners_for_qt) >= 2:
            lines.append(f"**{qt}:**")
            for (name, wins, total, examples) in sorted(winners_for_qt, key=lambda x: -x[1]):
                qids = [ex["qid"] for ex in examples]
                lines.append(f"- {name}: {wins}/{total} on qids {qids}")
            lines.append("")

    # 6. v0.3.1 dispatcher config
    lines.append("## 6. v0.3.1 Dispatcher Config")
    lines.append("")
    lines.append("```python")
    lines.append("SCAFFOLD_DISPATCHER = {")
    for qt, (best_scaffold, best_wins, best_total) in dispatcher.items():
        lines.append(f"    \"{qt}\": \"{best_scaffold}\",  # {best_wins}/{best_total} failures fixed")
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("To use: detect qtype at query-time (or use existing router), then select scaffold from dispatcher before calling LLM reranker.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()

    # Warm up ollama
    print(f"[arena] Warming up {ARENA_MODEL}...")
    warmup = call_ollama("You are helpful.", "Say 'ready' in one word.")
    if warmup is None:
        print(f"[arena] ERROR: {ARENA_MODEL} not responding. Check: ollama list")
        sys.exit(1)
    print(f"[arena] Model ready. Starting arena.\n")

    results, set_a, total_calls, api_errors, parse_failures = run_arena()

    elapsed = time.time() - t0
    print(f"\n[arena] Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"[arena] Generating report...")

    report = generate_report(results, set_a, total_calls, api_errors, parse_failures)

    RESULTS_MD.write_text(report)
    print(f"[arena] Report written to {RESULTS_MD}")
    print(f"\n--- SUMMARY ---")
    print(f"Total calls: {total_calls} | Errors: {api_errors} | Parse failures: {parse_failures}")
    print(f"Report: {RESULTS_MD}")
