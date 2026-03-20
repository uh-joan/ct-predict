#!/usr/bin/env python3
"""Generate progress.png from results.tsv — styled like karpathy/autoresearch."""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Parse results.tsv
experiments = []
with open("results.tsv") as f:
    header = f.readline()
    for i, line in enumerate(f):
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        commit, auc, nfeat, status, desc = parts[0], float(parts[1]), int(parts[2]), parts[3], parts[4]
        experiments.append({
            "idx": i,
            "commit": commit,
            "auc": auc,
            "n_features": nfeat,
            "status": status,
            "desc": desc,
        })

# Separate kept vs discarded/crash
kept = [e for e in experiments if e["status"] == "keep"]
discarded = [e for e in experiments if e["status"] in ("discard", "crash")]

# Build running best line
running_best = []
best_so_far = 0
for e in experiments:
    if e["status"] == "keep" and e["auc"] > best_so_far:
        best_so_far = e["auc"]
    running_best.append(best_so_far)

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Discarded experiments (gray, small)
ax.scatter(
    [e["idx"] for e in discarded],
    [e["auc"] for e in discarded],
    color="#cccccc", s=18, zorder=2, label="Discarded", alpha=0.6,
)

# Running best step line
ax.step(
    range(len(experiments)),
    running_best,
    where="post", color="#2ca02c", linewidth=1.5, zorder=3, label="Running best",
)

# Kept experiments (green, bigger)
ax.scatter(
    [e["idx"] for e in kept],
    [e["auc"] for e in kept],
    color="#2ca02c", s=50, zorder=4, label="Kept", edgecolors="white", linewidth=0.5,
)

# Annotate kept experiments
for e in kept:
    # Short label
    label = e["desc"][:45]
    ax.annotate(
        label,
        xy=(e["idx"], e["auc"]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=6.5,
        color="#2ca02c",
        rotation=30,
        ha="left", va="bottom",
        alpha=0.85,
    )

# Labels and title
n_kept = len(kept)
n_total = len(experiments)
ax.set_title(
    f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Experiment #", fontsize=11)
ax.set_ylabel("Validation AUC-ROC (higher is better)", fontsize=11)

# Legend
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

# Grid
ax.grid(True, alpha=0.2)
ax.set_xlim(-1, n_total + 1)

# Y-axis range
all_aucs = [e["auc"] for e in experiments if e["auc"] > 0]
ymin = min(all_aucs) - 0.01
ymax = max(all_aucs) + 0.01
ax.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig("progress.png", dpi=180, bbox_inches="tight")
print(f"Saved progress.png ({n_total} experiments, {n_kept} kept)")
