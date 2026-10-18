"""Median of several bench runs, row by row: python scripts/aggregate.py out.csv run1.csv run2.csv ...

Recall is deterministic; latency/build time vary run to run, so we publish the median run.
"""
import csv
import statistics
import sys

out, runs = sys.argv[1], [list(csv.DictReader(open(p, newline=""))) for p in sys.argv[2:]]
keys = list(runs[0][0].keys())
text_cols = {"algo"}
int_cols = {"M", "ef_construction", "ef_search"}  # configuration, identical across runs
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(keys)
    for rows in zip(*runs):
        merged = []
        # per-insert cost must agree with build_s, so derive it from the median build time
        # instead of taking an independent median (which gave 188 vs 189 for M=16).
        ratio = float(rows[0]["build_us_per_insert"]) / float(rows[0]["build_s"]) if float(rows[0]["build_s"]) else 0
        build_med = statistics.median(float(r["build_s"]) for r in rows)
        for k in keys:
            if k == "build_us_per_insert":
                merged.append(f"{build_med * ratio:.0f}" if ratio else "0")
                continue
            if k in text_cols or k in int_cols:
                merged.append(rows[0][k])
            else:
                vals = [float(r[k]) for r in rows]
                v = statistics.median(vals)
                merged.append(f"{v:.4f}" if k.startswith("recall") else f"{v:.2f}" if v else "0")
        w.writerow(merged)
print("wrote", out, "from", len(runs), "runs")
