"""
plot_analysis.py
----------------
Generates a multi-panel analysis of MPC experiment data,
focusing on the relationship of K with consensus_time and max_mpc_time.

Usage:
    python plot_analysis.py                         # reads output.csv in CWD
    python plot_analysis.py /path/to/output.csv     # custom path
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── 1. Load data ──────────────────────────────────────────────────────────────
csv_path = sys.argv[1] if len(sys.argv) > 1 else "output.csv"
df = pd.read_csv(csv_path)
df["max_mpc_ms"] = df["max_mpc_time"] * 1000   # seconds → ms for readability
df["mean_mpc_ms"] = df["mean_mpc_time"] * 1000

# ── 2. Style ──────────────────────────────────────────────────────────────────
PALETTE = {
    (3, "mixed"):    "#378ADD",
    (3, "circular"): "#1D9E75",
    (4, "mixed"):    "#D85A30",
    (4, "circular"): "#9F59C4",
}
MARKERS = {
    (3, "mixed"):    "o",
    (3, "circular"): "s",
    (4, "mixed"):    "^",
    (4, "circular"): "D",
}
LABELS = {
    (3, "mixed"):    "3 agents · mixed",
    (3, "circular"): "3 agents · circular",
    (4, "mixed"):    "4 agents · mixed",
    (4, "circular"): "4 agents · circular",
}
T_LS = {1200: "-", 200: "--"}   # T=1200 solid, T=200 dashed

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F9F9F7",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    1.1,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom":False,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "medium",
    "axes.labelsize":    11,
    "xtick.color":       "#888",
    "ytick.color":       "#888",
    "axes.labelcolor":   "#555",
})

KS = sorted(df["K"].unique())
SERIES = [(n, traj) for n in [3, 4] for traj in ["mixed", "circular"]]
TS = sorted(df["T"].unique())


def draw_line(ax, key, T, ycol, avg_across_T=False):
    """Plot one series line on ax."""
    n, traj = key
    mask = (df["num_agents"] == n) & (df["trajectory_type"] == traj)
    if not avg_across_T:
        mask &= (df["T"] == T)
    sub = df[mask].groupby("K")[ycol].mean().reindex(KS)
    ax.plot(
        KS, sub.values,
        color=PALETTE[key],
        marker=MARKERS[key],
        linestyle=T_LS.get(T, "-"),
        linewidth=2,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.8,
        label=LABELS[key] + (f" (T={T})" if not avg_across_T else ""),
        zorder=3,
    )


def style_ax(ax, title, ylabel, logy=False):
    ax.set_title(title, pad=10)
    ax.set_xlabel("K  (prediction horizon)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(KS)
    if logy:
        ax.set_yscale("log")
    ax.tick_params(axis="both", length=0)


def add_legend(ax, handles=None, loc="upper right", ncol=1):
    if handles:
        ax.legend(handles=handles, fontsize=9, framealpha=0.9,
                  edgecolor="#ddd", loc=loc, ncol=ncol)
    else:
        ax.legend(fontsize=9, framealpha=0.9, edgecolor="#ddd",
                  loc=loc, ncol=ncol)


# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 18))
fig.suptitle("MPC Experiment Analysis  ·  K vs Consensus & Solve Time",
             fontsize=15, fontweight="medium", y=0.98, color="#222")

gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                      left=0.07, right=0.97, top=0.94, bottom=0.05)


# ── Plot 1: Consensus time vs K, split by T ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for T in TS:
    for key in SERIES:
        draw_line(ax1, key, T, "consensus_time")
style_ax(ax1, "Consensus time vs K  (by T)", "consensus time (s)")

# build a clean two-level legend
leg_series = [Line2D([0],[0], color=PALETTE[k], marker=MARKERS[k],
                     markerfacecolor="white", markeredgewidth=1.5,
                     linewidth=1.8, label=LABELS[k]) for k in SERIES]
leg_T = [Line2D([0],[0], color="#888", linestyle=T_LS[t], linewidth=1.8,
                label=f"T = {t}") for t in TS]
ax1.legend(handles=leg_series + leg_T, fontsize=8.5, framealpha=0.9,
           edgecolor="#ddd", loc="upper right", ncol=1)


# ── Plot 2: Max MPC time vs K, split by T ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for T in TS:
    for key in SERIES:
        draw_line(ax2, key, T, "max_mpc_ms")
style_ax(ax2, "Max MPC solve time vs K  (by T)", "max MPC time (ms)")
ax2.legend(handles=leg_series + leg_T, fontsize=8.5, framealpha=0.9,
           edgecolor="#ddd", loc="upper left", ncol=1)


# ── Plot 3: Tradeoff scatter — consensus vs max MPC time, coloured by K ───────
ax3 = fig.add_subplot(gs[1, 0])
cmap = plt.cm.plasma
norm = plt.Normalize(df["K"].min(), df["K"].max())
sc = ax3.scatter(
    df["max_mpc_ms"], df["consensus_time"],
    c=df["K"], cmap=cmap, norm=norm,
    s=70, alpha=0.82, edgecolors="white", linewidths=0.6, zorder=3,
)
fig.colorbar(sc, ax=ax3, label="K", pad=0.02)
ax3.set_title("Tradeoff: consensus time vs max MPC time", pad=10)
ax3.set_xlabel("max MPC solve time (ms)")
ax3.set_ylabel("consensus time (s)")
ax3.tick_params(axis="both", length=0)

# annotate K=1 and K=25 extremes
for _, row in df[df["K"].isin([1, 25])].groupby(["K","num_agents","trajectory_type"]).mean(numeric_only=True).reset_index().iterrows():
    ax3.annotate(f"K={int(row.K)}", (row.max_mpc_ms, row.consensus_time),
                 fontsize=7.5, color="#555",
                 xytext=(4, 4), textcoords="offset points")


# ── Plot 4: Mean vs Max MPC time — how much variance in solve time ─────────────
ax4 = fig.add_subplot(gs[1, 1])
for key in SERIES:
    n, traj = key
    sub = df[(df["num_agents"]==n) & (df["trajectory_type"]==traj)].groupby("K")[["mean_mpc_ms","max_mpc_ms"]].mean()
    ax4.fill_between(sub.index, sub["mean_mpc_ms"], sub["max_mpc_ms"],
                     alpha=0.15, color=PALETTE[key])
    ax4.plot(sub.index.to_numpy(), sub["mean_mpc_ms"].to_numpy(), color=PALETTE[key],
             linewidth=1.5, linestyle="--", zorder=3)
    ax4.plot(sub.index.to_numpy(), sub["max_mpc_ms"].to_numpy(), color=PALETTE[key],
             marker=MARKERS[key], linewidth=2, markerfacecolor="white",
             markeredgewidth=1.8, markersize=6, label=LABELS[key], zorder=3)
style_ax(ax4, "Mean vs Max MPC solve time  (shaded = variance band)", "MPC time (ms)")
ax4.legend(handles=leg_series + [
    Line2D([0],[0], color="#888", linewidth=2, label="max"),
    Line2D([0],[0], color="#888", linewidth=1.5, linestyle="--", label="mean"),
], fontsize=8.5, framealpha=0.9, edgecolor="#ddd", loc="upper left")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = "mpc_analysis.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved → {out_path}")
plt.close()