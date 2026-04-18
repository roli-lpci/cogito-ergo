"""Threshold calibration analysis for runtime escalation rule."""
import json

data = json.load(open('bench/runs/runO-runtime-escalate/per_question.json'))
print(f"Questions analyzed: {len(data)}")
all_scores = []
for e in data:
    scores = e.get('s1_top5_scores', [])
    if not scores:
        continue
    top1 = scores[0]
    gap = scores[0] - scores[1] if len(scores) >= 2 else 0
    all_scores.append((top1, gap, e['s1_hit_at_1']))

n = len(all_scores)
print(f"With s1 scores logged: {n}")
print(f"top1: min={min(t[0] for t in all_scores):.3f} mean={sum(t[0] for t in all_scores)/n:.3f}")
print(f"gap:  min={min(t[1] for t in all_scores):.3f} mean={sum(t[1] for t in all_scores)/n:.3f}")

print("\nThreshold options (target: ~8% fire rate = 37-50/470):")
thresholds = [
    (0.75, 0.05),  # current (too loose)
    (0.70, 0.04),
    (0.65, 0.03),
    (0.65, 0.02),
    (0.60, 0.03),
    (0.60, 0.02),
    (0.55, 0.02),
]
for t_t, g_t in thresholds:
    count = sum(1 for t, g, h in all_scores if t < t_t or g < g_t)
    rate = count/n
    proj_470 = int(rate * 470)
    # Check if escalated questions actually had s1 misses
    esc_s1miss = sum(1 for t, g, h in all_scores if (t < t_t or g < g_t) and not h)
    print(f"  top1<{t_t} OR gap<{g_t}: {count}/{n}={rate:.1%} -> proj ~{proj_470}/470  (s1_miss in escalated: {esc_s1miss})")
