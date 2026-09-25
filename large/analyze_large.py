"""
Summaries + charts for the large-space round.
  results/synthetic_summary.md, results/synthetic.png
  results/real_large_summary.md, results/real_large.png   (if real_large.csv exists)
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(__file__), "results")
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [("ags_no_early_stop", "AGS (no early stop)", "#2a78d6"),
          ("optuna_tpe", "Optuna TPE", "#eb6834"),
          ("random", "Random search", "#1baf7a")]


def style(ax):
    ax.grid(True, color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=INK2, labelsize=8)


# ------------------------------------------------------------- synthetic ---
syn = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, "synthetic_*.csv")))])
syn["gridlabel"] = syn.levels.astype(str) + "^" + syn.dims.astype(str)
BUD = [50, 100, 200]
final_default = syn[syn.method == "ags_default"].sort_values("budget").groupby(
    ["func", "gridlabel", "seed"]).tail(1)

lines = ["# Large-space synthetic benchmark", "",
         "Median over 10 seeds of the **rank percentile** of each method's pick: the fraction "
         "of the whole grid that is strictly better (0 = global optimum; 1e-3 = top 0.1%). "
         "Lower is better. Fold noise sigma = 0.05 grid-std.", ""]
for gl, g in syn.groupby("gridlabel", sort=False):
    size = int(g.grid.iloc[0])
    lines += [f"## Grid {gl} = {size:,} points", "",
              "| landscape | AGS default (evals used) | AGS @50 | AGS @100 | AGS @200 "
              "| Optuna @50 | Optuna @100 | Optuna @200 | Random @200 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for f, gf in g.groupby("func"):
        med = lambda m, b: gf[(gf.method == m) & (gf.budget == b)].rank_pct.median()
        fd = final_default[(final_default.func == f) & (final_default.gridlabel == gl)]
        lines.append(
            f"| {f} | {fd.rank_pct.median():.1e} ({fd.n_evals.median():.0f}) "
            f"| {med('ags_no_early_stop', 50):.1e} | {med('ags_no_early_stop', 100):.1e} "
            f"| {med('ags_no_early_stop', 200):.1e} | {med('optuna_tpe', 50):.1e} "
            f"| {med('optuna_tpe', 100):.1e} | {med('optuna_tpe', 200):.1e} "
            f"| {med('random', 200):.1e} |")
    lines.append("")

# paired AGS(no stop) vs Optuna / Random per budget, all landscapes & grids pooled
lines += ["## Paired by seed, all landscapes and grids pooled", "",
          "W = AGS (no early stop) pick strictly better, T = equal, L = worse.", "",
          "| budget | vs Optuna W/T/L | vs Random W/T/L |", "|---|---|---|"]
key = ["func", "gridlabel", "seed", "budget"]
A = syn[syn.method == "ags_no_early_stop"].set_index(key).rank_pct
for b in BUD:
    cells = []
    for m in ["optuna_tpe", "random"]:
        B = syn[syn.method == m].set_index(key).rank_pct
        d = (B - A).xs(b, level="budget").dropna()
        cells.append(f"{(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}")
    lines.append(f"| {b} | {cells[0]} | {cells[1]} |")
open(os.path.join(R, "synthetic_summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

funcs = sorted(syn.func.unique())
grids = list(syn.gridlabel.unique())
fig, axes = plt.subplots(len(grids), len(funcs), figsize=(3.4 * len(funcs), 2.7 * len(grids)),
                         dpi=150, sharex=True)
floor = 1e-7
for r, gl in enumerate(grids):
    for c, f in enumerate(funcs):
        ax = axes[r, c]
        g = syn[(syn.func == f) & (syn.gridlabel == gl)]
        for key_, label, col in SERIES:
            m = g[(g.method == key_) & g.budget.isin(BUD)].groupby("budget").rank_pct.median()
            ax.plot(m.index, np.maximum(m.values, floor), color=col, lw=2, marker="o", ms=4,
                    label=label)
        fd = final_default[(final_default.func == f) & (final_default.gridlabel == gl)]
        ax.plot(fd.n_evals.median(), max(fd.rank_pct.median(), floor), "o", ms=8,
                color="#2a78d6", mec="white", mew=2, label="AGS default (stops early)")
        ax.set_yscale("log")
        ax.set_ylim(floor * 0.5, 1)
        style(ax)
        if r == 0:
            ax.set_title(f, color=INK, fontsize=10, loc="left")
        if c == 0:
            ax.set_ylabel(f"grid {gl}\nrank pct (log)", color=INK2, fontsize=8)
        if r == len(grids) - 1:
            ax.set_xlabel("evaluations", color=INK2, fontsize=8)
axes[0, 0].legend(frameon=False, fontsize=7, labelcolor=INK, loc="lower left")
fig.suptitle("Fraction of grid better than the pick (median, 10 seeds; lower is better; "
             f"0 plotted at {floor:g}; lines at the floor overlap = found the optimum)", color=INK, fontsize=10, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(os.path.join(R, "synthetic.png"), facecolor="white")

# ------------------------------------------------------------------ real ---
p = os.path.join(R, "real_large.csv")
if os.path.exists(p):
    df = pd.read_csv(p)
    CP = [25, 50, 100, 150]
    fdft = df[df.method == "ags_default"].sort_values("budget").groupby("seed").tail(1)
    out = ["# Large real space: HistGradientBoosting on Covertype (103,680 configs)", "",
           "Held-out ROC AUC (10,000 unseen rows) of each method's pick. Mean ± sd over seeds.", "",
           "| method | @25 | @50 | @100 | @150 | folds @150 | wall s (full run) |",
           "|---|---|---|---|---|---|---|"]
    for key_, label, _ in SERIES:
        g = df[df.method == key_]
        cells = []
        for b in CP:
            x = g[g.budget == b].test
            cells.append(f"{x.mean():.4f} ± {x.std():.4f}" if len(x) else "–")
        last = g[g.budget == 150]
        out.append(f"| {label} | " + " | ".join(cells) +
                   f" | {last.folds.mean():.0f} | {g.wall_total.mean():.0f} |")
    out.append(f"| AGS default (stopped at {fdft.n_evals.mean():.0f} evals avg) "
               f"| final: {fdft.test.mean():.4f} ± {fdft.test.std():.4f} | | | "
               f"| {fdft.folds.mean():.0f} | {fdft.wall_total.mean():.0f} |")
    out += ["", "Picks at 150 evals:", ""]
    for key_, label, _ in SERIES:
        for _, row in df[(df.method == key_) & (df.budget == 150)].iterrows():
            out.append(f"- {label}, seed {row.seed}: test {row.test:.4f} — `{row.pick}`")
    open(os.path.join(R, "real_large_summary.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))

    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=150)
    for key_, label, col in SERIES:
        m = df[(df.method == key_) & df.budget.isin(CP)].groupby("budget").test.mean()
        ax.plot(m.index, m.values, color=col, lw=2, marker="o", ms=5, label=label)
    ax.plot(fdft.n_evals.mean(), fdft.test.mean(), "o", ms=9, color="#2a78d6", mec="white",
            mew=2, label="AGS default (stops early)")
    style(ax)
    ax.set_xlabel("evaluations", color=INK2, fontsize=9)
    ax.set_ylabel("held-out ROC AUC of pick\n(higher is better)", color=INK2, fontsize=9)
    ax.set_title("HGB on Covertype, 103,680-config grid (mean of seeds)", color=INK,
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "real_large.png"), facecolor="white")
