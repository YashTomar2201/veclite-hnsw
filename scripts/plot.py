"""Recall@10 (x) vs median query latency (y, log scale), one line per M.

Usage: python scripts/plot.py [results.csv] [out.png]
Output is 1800x1200 px with large text so it stays readable on a phone.
"""
import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "recall_vs_latency.png"

HIGHLIGHT_M, HIGHLIGHT_EF = 16, 40

series, brute = defaultdict(list), None
with open(src, newline="") as f:
    for row in csv.DictReader(f):
        point = (float(row["recall@10"]), float(row["median_us"]), int(row["ef_search"]))
        if row["algo"] == "brute_force":
            brute = point
        else:
            series[int(row["M"])].append(point)

plt.rcParams.update({"font.size": 15, "axes.titlesize": 18, "axes.labelsize": 17,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15})
colors = {8: "#1f77b4", 16: "#d95f02", 32: "#2ca02c"}
offsets = {8: (-34, 6), 16: (12, -22), 32: (12, 8)}  # label nudges so overlapping points stay legible

fig, ax = plt.subplots(figsize=(12, 8))
for M, pts in sorted(series.items()):
    pts.sort(key=lambda p: p[2])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=colors.get(M),
            label=f"HNSW M={M}", lw=2.6, ms=8)
    seen_saturated = False
    for x, y, ef in pts:
        if x >= 0.9995:  # recall saturated: label only the cheapest such point
            if seen_saturated:
                continue
            seen_saturated = True
        ax.annotate(str(ef), (x, y), textcoords="offset points", xytext=offsets.get(M, (6, 6)),
                    fontsize=13, color=colors.get(M), fontweight="bold")

if brute:
    ax.plot(*brute[:2], marker="*", ms=26, color="black", ls="none",
            label=f"Brute force (exact): {brute[1]:.0f} µs")

# Circle + callout on the headline configuration (numbers come from the CSV, not hard-coded).
hl = next(p for p in series[HIGHLIGHT_M] if p[2] == HIGHLIGHT_EF)
if brute:
    callout = f"{brute[1] / hl[1]:.1f}x faster, {hl[0] * 100:.1f}% recall"
    ax.scatter([hl[0]], [hl[1]], s=900, facecolors="none", edgecolors="crimson", linewidths=3, zorder=5)
    ax.annotate(callout, xy=(hl[0], hl[1]), xytext=(0.815, 135), fontsize=18, fontweight="bold",
                color="crimson", arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.5,
                                                 connectionstyle="arc3,rad=-0.12", shrinkB=18))

ax.set_yscale("log")
ticks = [10, 20, 50, 100, 200, 500, 1000]
ax.set_yticks(ticks)
ax.set_yticklabels([str(t) for t in ticks])
ax.minorticks_off()
ax.set_xlim(0.78, 1.012)
ax.set_xlabel("recall@10")
ax.set_ylabel("median query latency (µs, log scale)")
ax.set_title("HNSW from scratch (C++17) on SIFT10K\nnumbers on points = ef_search; lower-right is better")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(dst, dpi=150)
print("wrote", dst)
