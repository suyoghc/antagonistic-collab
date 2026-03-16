"""Visualize M13 debate ablation results (3×2 design)."""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- Load data ---
with open("runs/debate_ablation/summary.json") as f:
    main = json.load(f)
with open("runs/debate_no_arbiter/summary.json") as f:
    no_arb = json.load(f)

# --- Organize into structured data ---
ground_truths = ["GCM", "RULEX", "SUSTAIN"]
debate_levels = ["No Debate", "Debate (no arbiter)", "Debate + Arbiter"]
strategies = ["thompson", "greedy"]

# Map condition keys to debate level
def get_entry(data, prefix, gt):
    key = f"{prefix}_{gt}"
    return data.get(key, {})

# Build arrays: [debate_level][strategy][ground_truth]
rmse = np.zeros((3, 2, 3))
gap = np.zeros((3, 2, 3))

for gi, gt in enumerate(ground_truths):
    for si, strat in enumerate(strategies):
        # No debate
        e = get_entry(main, f"{strat}_no_debate", gt)
        rmse[0, si, gi] = e.get("winner_rmse", np.nan)
        gap[0, si, gi] = e.get("gap_pct", np.nan)

        # Debate no arbiter
        e = get_entry(no_arb, f"{strat}_debate_no_arbiter", gt)
        rmse[1, si, gi] = e.get("winner_rmse", np.nan)
        gap[1, si, gi] = e.get("gap_pct", np.nan)

        # Debate + arbiter
        e = get_entry(main, f"{strat}_debate", gt)
        rmse[2, si, gi] = e.get("winner_rmse", np.nan)
        gap[2, si, gi] = e.get("gap_pct", np.nan)

# --- Colors ---
colors = {
    "No Debate": "#2ecc71",
    "Debate (no arbiter)": "#e74c3c",
    "Debate + Arbiter": "#3498db",
}

# --- Figure: 2 panels ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [2, 1]})
fig.suptitle(
    "M13 Debate Ablation: arbiter-v0.1 (3×2 design, 18/18 correct)",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

# --- Panel A: RMSE by condition ---
ax = axes[0]
x = np.arange(len(ground_truths))
bar_width = 0.12
offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

condition_labels = []
for di, dlevel in enumerate(debate_levels):
    for si, strat in enumerate(strategies):
        idx = di * 2 + si
        vals = rmse[di, si, :]
        label = f"{dlevel}\n({strat})" if si == 0 else f"({strat})"
        bar_label = f"{dlevel} ({strat})"
        condition_labels.append(bar_label)
        bars = ax.bar(
            x + offsets[idx] * bar_width,
            vals,
            bar_width * 0.9,
            label=bar_label,
            color=colors[dlevel],
            alpha=0.9 if si == 0 else 0.6,
            edgecolor="white",
            linewidth=0.5,
        )
        # Add value labels
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=45,
                )

ax.set_xticks(x)
ax.set_xticklabels(
    ["GCM\n(→ Exemplar)", "RULEX\n(→ Rule)", "SUSTAIN\n(→ Clustering)"],
    fontsize=10,
)
ax.set_ylabel("Winner RMSE (lower is better)", fontsize=11)
ax.set_title("A. RMSE by Condition × Ground Truth", fontsize=12, pad=10)
ax.set_ylim(0, 0.21)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, linestyle="--")

# Legend
handles = [
    plt.Rectangle((0, 0), 1, 1, fc=colors[dl], alpha=0.9)
    for dl in debate_levels
]
strat_handles = [
    plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.9),
    plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.5),
]
ax.legend(
    handles + strat_handles,
    debate_levels + ["Thompson", "Greedy"],
    loc="upper right",
    fontsize=8,
    ncol=2,
    framealpha=0.9,
)

# --- Panel B: Summary by debate level ---
ax2 = axes[1]

# Compute averages across strategies and ground truths
avg_rmse = np.nanmean(rmse, axis=(1, 2))
avg_gap = np.nanmean(gap, axis=(1, 2))

y_pos = np.arange(len(debate_levels))
bars = ax2.barh(
    y_pos,
    avg_gap,
    color=[colors[dl] for dl in debate_levels],
    edgecolor="white",
    linewidth=0.5,
    height=0.6,
)

for bar, g, r in zip(bars, avg_gap, avg_rmse):
    ax2.text(
        bar.get_width() - 1.5,
        bar.get_y() + bar.get_height() / 2,
        f"Gap: {g:.1f}%\nRMSE: {r:.3f}",
        ha="right",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
    )

ax2.set_yticks(y_pos)
ax2.set_yticklabels(debate_levels, fontsize=10)
ax2.set_xlabel("Average Gap % (higher is better)", fontsize=11)
ax2.set_title("B. Summary by Debate Level", fontsize=12, pad=10)
ax2.set_xlim(0, 100)
ax2.invert_yaxis()
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="x", alpha=0.3, linestyle="--")

# Add time annotation
times = ["~6 min", "~22 min", "~18 min"]
for i, t in enumerate(times):
    ax2.text(2, i, t, ha="left", va="center", fontsize=8, color="gray", style="italic")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Notes/figures/m13_ablation.png", dpi=200, bbox_inches="tight")
plt.savefig("Notes/figures/m13_ablation.pdf", bbox_inches="tight")
print("Saved to Notes/figures/m13_ablation.png and .pdf")
