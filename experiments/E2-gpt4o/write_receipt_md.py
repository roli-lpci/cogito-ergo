"""Generate receipt.md from E2 summary + receipt JSONs. Run after E2 completes."""
import json
import math
from pathlib import Path

here = Path(__file__).parent
summary = json.load(open(here / "qa_eval_v3_E2_runP-v35_summary.json"))
receipt = json.load(open(here / "qa_eval_v3_E2_runP-v35_receipt.json"))

E0_BASELINES = {
    "knowledge-update":           {"k1": 0.6389, "k5": 0.5833, "n": 72},
    "multi-session":              {"k1": 0.2479, "k5": 0.6777, "n": 121},
    "single-session-assistant":   {"k1": 0.9286, "k5": 0.4107, "n": 56},
    "single-session-preference":  {"k1": 0.4000, "k5": 0.0000, "n": 30},
    "single-session-user":        {"k1": 0.9063, "k5": 0.2344, "n": 64},
    "temporal-reasoning":         {"k1": 0.3071, "k5": 0.4016, "n": 127},
}
E0_OPTIMAL_BLENDED = 0.6404


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0, center - margin), min(1, center + margin))


n_total = summary["n_questions"]
overall = summary["overall_qa_accuracy"]
n_correct = round(overall * n_total)
lo, hi = wilson_ci(n_correct, n_total)
delta_e0 = overall - E0_OPTIMAL_BLENDED

cost = summary.get("cost", {})
cost_usd = cost.get("total_usd", receipt.get("cost_usd_total", "N/A"))
tok_in = cost.get("total_input_tokens", receipt.get("tokens_in", "N/A"))
tok_out = cost.get("total_output_tokens", receipt.get("tokens_out", "N/A"))

qtype_rows = []
for qt, v in sorted(summary["qtype_qa_accuracy"].items()):
    base = E0_BASELINES.get(qt, {})
    baseline_k = v["k"]
    baseline_acc = base.get(f"k{baseline_k}", base.get("k1", 0))
    delta = v["qa_accuracy"] - baseline_acc
    qtype_rows.append(
        f"| {qt:<30} | {v['n']:>3} | {baseline_acc:>6.1%} | {v['qa_accuracy']:>6.1%} | "
        f"{delta:>+7.1%} | {v['reader_model']:<14} | K={v['k']} |"
    )

lines = [
    "# E2 Receipt — GPT-4o Reader",
    "",
    "**Experiment:** E2",
    "**Date:** 2026-04-24",
    "**Run ID:** runP-v35",
    "",
    "## Config",
    "- GPT-4o reader: MS (K=5), TR (K=5), KU (K=1)",
    "- gpt-4o-mini reader: SSU (K=1), SSA (K=1), Pref (K=1)",
    "- Prompts: v2 for all qtypes (v3 temporal for TR)",
    "- Grader: gpt-4o-mini",
    "- max_tokens: 512",
    "",
    "## Result",
    f"- **Overall QA:** {overall:.1%} ({n_correct}/{n_total})",
    f"- **95% CI:** [{lo:.1%}, {hi:.1%}] (Wilson, n={n_total})",
    f"- **E0 optimal blended:** {E0_OPTIMAL_BLENDED:.1%}",
    f"- **Delta vs E0:** {delta_e0:+.1%}",
    f"- **Cost:** ${cost_usd} USD",
    f"- **Tokens:** {tok_in:,} in / {tok_out:,} out" if isinstance(tok_in, int) else f"- **Tokens:** {tok_in} in / {tok_out} out",
    "",
    "## Per-Qtype Breakdown",
    "",
    "| qtype                          | n   | baseline | E2 acc | delta   | reader         | K     |",
    "|--------------------------------|-----|----------|--------|---------|----------------|-------|",
] + qtype_rows + [
    "",
    "## Key Findings",
    "",
    "*(Fill in after reviewing)*",
    "",
    "## Decision Gate",
    "",
    "- Target: ≥75.0%",
    f"- E2 result: {overall:.1%}",
    f"- {'FLOOR MET — consider stopping or running E5 final measurement' if overall >= 0.75 else 'BELOW FLOOR — proceed to E2b or E3'}",
]

out = here / "receipt.md"
out.write_text("\n".join(lines) + "\n")
print(f"Written: {out}")
print("\nE2 Summary:")
print(f"  Overall: {overall:.1%} ({n_correct}/{n_total})")
print(f"  95% CI: [{lo:.1%}, {hi:.1%}]")
print(f"  Delta vs E0: {delta_e0:+.1%}")
print(f"  Cost: ${cost_usd}")
print()
for row in qtype_rows:
    print(row)
