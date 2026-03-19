"""Generate publication-quality figures for M16 open design space results.

M16 tests whether agents proposing all experiment structures via debate
(open design space) improves or degrades model identification compared to
a curated registry, and whether the arbiter's crux machinery helps focus
agent proposals. The 2x2+1 factorial design crosses registry type
(closed/open) with debate/arbiter conditions.

Figures produced:
  1. m16_gap_by_condition  — Grouped bar chart of Gap% by condition and GT
  2. m16_gap_heatmap       — Heatmap of Gap% across conditions and GTs
  3. m16_gap_advantage     — Bar chart of pp advantage over baseline
  4. m16_arbiter_recovery  — Arbiter recovery of open-design losses
  5. m16_m15_arbiter_comparison — Cross-milestone arbiter comparison

Usage:
    python scripts/figures/m16_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    # Fallback for older matplotlib
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)

# ---------------------------------------------------------------------------
# Colorblind-friendly palette (Okabe-Ito)
# ---------------------------------------------------------------------------
OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_GREEN = "#009E73"
OI_VERMILION = "#D55E00"
OI_PURPLE = "#CC79A7"
OI_SKY = "#56B4E9"
OI_YELLOW = "#F0E442"

CONDITION_COLORS = {
    "closed_no_debate": OI_BLUE,
    "closed_debate": OI_ORANGE,
    "closed_arbiter": OI_GREEN,
    "open_debate": OI_VERMILION,
    "open_arbiter": OI_PURPLE,
}

CONDITION_LABELS = {
    "closed_no_debate": "Closed\nNo Debate",
    "closed_debate": "Closed\nDebate",
    "closed_arbiter": "Closed\nArbiter",
    "open_debate": "Open\nDebate",
    "open_arbiter": "Open\nArbiter",
}

CONDITION_LABELS_SHORT = {
    "closed_no_debate": "Closed No-Deb",
    "closed_debate": "Closed Debate",
    "closed_arbiter": "Closed Arbiter",
    "open_debate": "Open Debate",
    "open_arbiter": "Open Arbiter",
}

# ---------------------------------------------------------------------------
# M16 data (15 conditions: 3 ground truths x 5 conditions)
# ---------------------------------------------------------------------------
GROUND_TRUTHS = ["GCM", "SUSTAIN", "RULEX"]
CONDITIONS = [
    "closed_no_debate",
    "closed_debate",
    "closed_arbiter",
    "open_debate",
    "open_arbiter",
]

# Gap% values indexed as M16_GAP[gt][condition]
M16_GAP = {
    "GCM": {
        "closed_no_debate": 76.8,
        "closed_debate": 81.0,
        "closed_arbiter": 79.2,
        "open_debate": 71.6,
        "open_arbiter": 76.9,
    },
    "SUSTAIN": {
        "closed_no_debate": 87.7,
        "closed_debate": 88.6,
        "closed_arbiter": 96.0,
        "open_debate": 64.1,
        "open_arbiter": 70.7,
    },
    "RULEX": {
        "closed_no_debate": 86.1,
        "closed_debate": 58.6,
        "closed_arbiter": 63.9,
        "open_debate": 82.7,
        "open_arbiter": 82.0,
    },
}

M16_RMSE = {
    "GCM": {
        "closed_no_debate": 0.088,
        "closed_debate": 0.067,
        "closed_arbiter": 0.073,
        "open_debate": 0.084,
        "open_arbiter": 0.074,
    },
    "SUSTAIN": {
        "closed_no_debate": 0.057,
        "closed_debate": 0.053,
        "closed_arbiter": 0.020,
        "open_debate": 0.108,
        "open_arbiter": 0.100,
    },
    "RULEX": {
        "closed_no_debate": 0.053,
        "closed_debate": 0.168,
        "closed_arbiter": 0.140,
        "open_debate": 0.055,
        "open_arbiter": 0.061,
    },
}

# M15 data (misspecification) — Gap% only
M15_GAP = {
    "GCM": {"no_debate": 74.4, "debate": 77.9, "arbiter": 79.3},
    "SUSTAIN": {"no_debate": 87.7, "debate": 85.8, "arbiter": 76.1},
    "RULEX": {"no_debate": 58.0, "debate": 80.4, "arbiter": 3.2},
}


def _save(fig, name):
    """Save figure as PDF and PNG, print paths."""
    pdf_path = OUT_DIR / f"{name}.pdf"
    png_path = OUT_DIR / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"  {pdf_path}")
    print(f"  {png_path}")
    return pdf_path, png_path


# ===================================================================
# Figure 1: Gap% by condition, grouped by ground truth
# ===================================================================
def fig1_gap_by_condition():
    """Grouped bar chart of Gap% for each condition, grouped by GT model.

    A horizontal dashed line marks the closed_no_debate baseline for each
    group. Value labels appear on each bar.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_gt = len(GROUND_TRUTHS)
    n_cond = len(CONDITIONS)
    bar_width = 0.14
    group_width = n_cond * bar_width
    x_centers = np.arange(n_gt) * (group_width + 0.35)

    for ci, cond in enumerate(CONDITIONS):
        offsets = x_centers + (ci - (n_cond - 1) / 2) * bar_width
        vals = [M16_GAP[gt][cond] for gt in GROUND_TRUTHS]
        bars = ax.bar(
            offsets,
            vals,
            bar_width * 0.88,
            label=CONDITION_LABELS_SHORT[cond],
            color=CONDITION_COLORS[cond],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )

    # Baseline dashed lines
    for gi, gt in enumerate(GROUND_TRUTHS):
        baseline = M16_GAP[gt]["closed_no_debate"]
        left = x_centers[gi] - group_width / 2 - 0.04
        right = x_centers[gi] + group_width / 2 + 0.04
        ax.hlines(
            baseline,
            left,
            right,
            colors="black",
            linestyles="dashed",
            linewidth=1.2,
            zorder=4,
            alpha=0.7,
        )
        ax.text(
            right + 0.02,
            baseline,
            f"baseline {baseline:.1f}%",
            va="center",
            ha="left",
            fontsize=7,
            color="black",
            alpha=0.7,
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(GROUND_TRUTHS, fontsize=12, fontweight="bold")
    ax.set_ylabel("Discrimination Gap (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Model Identification Gap by Condition and Ground Truth (M16)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=5,
        frameon=True,
        framealpha=0.9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

    fig.tight_layout()
    return _save(fig, "m16_gap_by_condition")


# ===================================================================
# Figure 2: Heatmap
# ===================================================================
def fig2_gap_heatmap():
    """Heatmap of Gap% with conditions on y-axis, GTs on x-axis.

    Uses a diverging colormap centered around the mean gap, with
    cell annotations showing the exact values.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Build matrix: rows = conditions, cols = ground truths
    data = np.array(
        [[M16_GAP[gt][cond] for gt in GROUND_TRUTHS] for cond in CONDITIONS]
    )

    mean_gap = np.mean(data)
    vmin = max(0, mean_gap - 30)
    vmax = min(100, mean_gap + 30)

    im = ax.imshow(
        data,
        cmap="RdYlGn",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    # Annotate cells
    for i in range(len(CONDITIONS)):
        for j in range(len(GROUND_TRUTHS)):
            val = data[i, j]
            # Choose text color based on brightness
            text_color = (
                "white" if val < (mean_gap - 15) or val > (mean_gap + 25) else "black"
            )
            ax.text(
                j,
                i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=text_color,
            )

    ax.set_xticks(np.arange(len(GROUND_TRUTHS)))
    ax.set_xticklabels(GROUND_TRUTHS, fontsize=12, fontweight="bold")
    ax.set_yticks(np.arange(len(CONDITIONS)))
    ax.set_yticklabels(
        [CONDITION_LABELS_SHORT[c] for c in CONDITIONS],
        fontsize=10,
    )
    ax.set_xlabel("Ground Truth Model", fontsize=12)
    ax.set_ylabel("Condition", fontsize=12)
    ax.set_title(
        "Discrimination Gap Across Design Space \u00d7 Model Type",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Gap %", fontsize=11)

    fig.tight_layout()
    return _save(fig, "m16_gap_heatmap")


# ===================================================================
# Figure 3: Gap advantage over baseline
# ===================================================================
def fig3_gap_advantage():
    """Grouped bar chart of pp advantage over closed_no_debate baseline.

    Positive values (green) = better than baseline.
    Negative values (red) = worse than baseline.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    comparison_conds = [c for c in CONDITIONS if c != "closed_no_debate"]
    n_gt = len(GROUND_TRUTHS)
    n_cond = len(comparison_conds)
    bar_width = 0.17
    group_width = n_cond * bar_width
    x_centers = np.arange(n_gt) * (group_width + 0.35)

    for ci, cond in enumerate(comparison_conds):
        offsets = x_centers + (ci - (n_cond - 1) / 2) * bar_width
        vals = []
        for gt in GROUND_TRUTHS:
            baseline = M16_GAP[gt]["closed_no_debate"]
            advantage = M16_GAP[gt][cond] - baseline
            vals.append(advantage)

        # Color each bar individually based on sign
        for oi, (offset, v) in enumerate(zip(offsets, vals)):
            color = OI_GREEN if v >= 0 else OI_VERMILION
            ax.bar(
                offset,
                v,
                bar_width * 0.88,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            # Value label
            va = "bottom" if v >= 0 else "top"
            y_off = 0.4 if v >= 0 else -0.4
            ax.text(
                offset,
                v + y_off,
                f"{v:+.1f}",
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold",
            )

    # Zero line
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(GROUND_TRUTHS, fontsize=12, fontweight="bold")
    ax.set_ylabel("Gap Advantage (percentage points)")
    ax.set_title(
        "Gap Advantage Over Computational Baseline (pp)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    # Legend for conditions
    legend_handles = [
        mpatches.Patch(color=CONDITION_COLORS[c], label=CONDITION_LABELS_SHORT[c])
        for c in comparison_conds
    ]
    # Add sign legend
    legend_handles.append(mpatches.Patch(color=OI_GREEN, label="Better than baseline"))
    legend_handles.append(
        mpatches.Patch(color=OI_VERMILION, label="Worse than baseline")
    )

    # Custom legend: use condition colors as border patches
    # Since bars are colored by sign, show condition identity via position
    # We'll use a table-like annotation instead
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=True,
        framealpha=0.9,
    )

    # Add condition labels along the top
    for ci, cond in enumerate(comparison_conds):
        for gi in range(n_gt):
            offset = x_centers[gi] + (ci - (n_cond - 1) / 2) * bar_width
            ax.text(
                offset,
                ax.get_ylim()[1] - 0.5,
                CONDITION_LABELS_SHORT[cond]
                .replace("Closed ", "C-")
                .replace("Open ", "O-"),
                ha="center",
                va="top",
                fontsize=5.5,
                rotation=45,
                alpha=0.6,
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

    fig.tight_layout()
    return _save(fig, "m16_gap_advantage")


# ===================================================================
# Figure 4: Arbiter recovery of open-design losses
# ===================================================================
def fig4_arbiter_recovery():
    """For each GT, show baseline, open_debate, and open_arbiter bars.

    Annotations highlight the 'recovery' from open_debate to open_arbiter
    and the remaining gap to baseline.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    recovery_conds = ["closed_no_debate", "open_debate", "open_arbiter"]
    recovery_colors = [OI_BLUE, OI_VERMILION, OI_PURPLE]
    recovery_labels = ["Closed No-Debate\n(baseline)", "Open Debate", "Open Arbiter"]

    n_gt = len(GROUND_TRUTHS)
    n_cond = len(recovery_conds)
    bar_width = 0.22
    group_width = n_cond * bar_width
    x_centers = np.arange(n_gt) * (group_width + 0.4)

    for ci, (cond, color, label) in enumerate(
        zip(recovery_conds, recovery_colors, recovery_labels)
    ):
        offsets = x_centers + (ci - (n_cond - 1) / 2) * bar_width
        vals = [M16_GAP[gt][cond] for gt in GROUND_TRUTHS]
        bars = ax.bar(
            offsets,
            vals,
            bar_width * 0.88,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Draw recovery arrows/annotations
    for gi, gt in enumerate(GROUND_TRUTHS):
        open_deb = M16_GAP[gt]["open_debate"]
        open_arb = M16_GAP[gt]["open_arbiter"]
        recovery = open_arb - open_deb

        # Arrow from open_debate bar to open_arbiter bar
        x_deb = x_centers[gi] + (1 - (n_cond - 1) / 2) * bar_width
        x_arb = x_centers[gi] + (2 - (n_cond - 1) / 2) * bar_width
        mid_x = (x_deb + x_arb) / 2
        mid_y = max(open_deb, open_arb) + 5

        # Curved arrow annotation
        ax.annotate(
            f"+{recovery:.1f}pp",
            xy=(x_arb, open_arb + 2.5),
            xytext=(mid_x, mid_y + 3),
            fontsize=8,
            fontweight="bold",
            color=OI_GREEN if recovery > 0 else OI_VERMILION,
            ha="center",
            arrowprops=dict(
                arrowstyle="->",
                color=OI_GREEN if recovery > 0 else OI_VERMILION,
                lw=1.5,
                connectionstyle="arc3,rad=-0.2",
            ),
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(GROUND_TRUTHS, fontsize=12, fontweight="bold")
    ax.set_ylabel("Discrimination Gap (%)")
    ax.set_ylim(0, 108)
    ax.set_title(
        "Arbiter Recovery of Open-Design Losses",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=True,
        framealpha=0.9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

    fig.tight_layout()
    return _save(fig, "m16_arbiter_recovery")


# ===================================================================
# Figure 5: Cross-milestone arbiter comparison (M15 vs M16)
# ===================================================================
def fig5_m15_arbiter_comparison():
    """Side-by-side comparison of arbiter effects across M15 and M16.

    Shows the pp change from baseline (no-debate) for each model type
    across three regimes: M15 misspecification, M16 closed, M16 open.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute pp changes from baseline for each regime
    # M15: arbiter gap - no_debate gap
    m15_arbiter_pp = {
        gt: M15_GAP[gt]["arbiter"] - M15_GAP[gt]["no_debate"] for gt in GROUND_TRUTHS
    }
    # M16 closed: closed_arbiter - closed_no_debate
    m16_closed_pp = {
        gt: M16_GAP[gt]["closed_arbiter"] - M16_GAP[gt]["closed_no_debate"]
        for gt in GROUND_TRUTHS
    }
    # M16 open: open_arbiter - open_debate (arbiter effect within open design)
    m16_open_pp = {
        gt: M16_GAP[gt]["open_arbiter"] - M16_GAP[gt]["open_debate"]
        for gt in GROUND_TRUTHS
    }

    regimes = [
        "M15\nMisspecification",
        "M16 Closed\n(correct spec)",
        "M16 Open\n(correct spec)",
    ]
    regime_data = [m15_arbiter_pp, m16_closed_pp, m16_open_pp]
    regime_colors = [OI_SKY, OI_BLUE, OI_PURPLE]

    n_gt = len(GROUND_TRUTHS)
    n_regime = len(regimes)
    bar_width = 0.22
    group_width = n_regime * bar_width
    x_centers = np.arange(n_gt) * (group_width + 0.35)

    for ri, (regime, data, color) in enumerate(
        zip(regimes, regime_data, regime_colors)
    ):
        offsets = x_centers + (ri - (n_regime - 1) / 2) * bar_width
        vals = [data[gt] for gt in GROUND_TRUTHS]
        bars = ax.bar(
            offsets,
            vals,
            bar_width * 0.88,
            label=regime,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            va = "bottom" if v >= 0 else "top"
            y_off = 0.5 if v >= 0 else -0.5
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + y_off,
                f"{v:+.1f}",
                ha="center",
                va=va,
                fontsize=9,
                fontweight="bold",
            )

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(GROUND_TRUTHS, fontsize=12, fontweight="bold")
    ax.set_ylabel("Arbiter Effect (pp change from baseline)")
    ax.set_title(
        "Arbiter Effects Across Experimental Regimes",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=True,
        framealpha=0.9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

    fig.tight_layout()
    return _save(fig, "m16_m15_arbiter_comparison")


# ===================================================================
# Main
# ===================================================================
def main():
    print(f"Generating M16 figures in {OUT_DIR}/\n")

    all_paths = []

    print("Figure 1: Gap by condition...")
    all_paths.extend(fig1_gap_by_condition())

    print("\nFigure 2: Gap heatmap...")
    all_paths.extend(fig2_gap_heatmap())

    print("\nFigure 3: Gap advantage over baseline...")
    all_paths.extend(fig3_gap_advantage())

    print("\nFigure 4: Arbiter recovery...")
    all_paths.extend(fig4_arbiter_recovery())

    print("\nFigure 5: M15 vs M16 arbiter comparison...")
    all_paths.extend(fig5_m15_arbiter_comparison())

    print(f"\nDone. Generated {len(all_paths)} files:")
    for p in all_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
