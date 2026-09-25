"""
Turn results/runs.csv + results/curves.csv into:
  results/summary.md       per-task table (mean ± sd over seeds)
  results/pairwise.md      AGS vs each rival: per-seed wins/ties/losses on test score
  results/anytime.png      regret vs evaluations, small multiples per task
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(__file__), "..", "results")
runs = pd.read_csv(os.path.join(R, "runs.csv"))
curves = pd.read_csv(os.path.join(R, "curves.csv"))

ORDER = ["grid_search", "ags_default", "ags_no_early_stop", "ags_no_prune_no_stop",
         "ags_gp_no_stop", "random", "optuna_tpe"]


def fmt(m, s, d=4):
    return f"{m:.{d}f}" if np.isnan(s) or s == 0 else f"{m:.{d}f} ± {s:.{d}f}"


lines = ["# Benchmark summary", "",
         "Mean ± sd over seeds. `regret` = grid-optimum reference-CV score minus the "
         "reference-CV score of the config each method picked (0 = found grid's best). "
         "`test` = held-out score of the picked config. `folds` = CV fits actually run. "
         "`wall` = seconds, single core.", ""]
for task, g in runs.groupby("task", sort=False):
    lines += [f"## {task}", "",
              "| method | regret | hit optimum | test | evals | folds | wall s |",
              "|---|---|---|---|---|---|---|"]
    for m in ORDER:
        x = g[g.method == m]
        if x.empty:
            continue
        lines.append(
            f"| {m} | {fmt(x.regret.mean(), x.regret.std())} "
            f"| {x.hit_optimum.sum()}/{len(x)} "
            f"| {fmt(x.test.mean(), x.test.std())} "
            f"| {x.n_evals.mean():.1f} | {x.folds.mean():.0f} | {x.wall.mean():.1f} |")
    lines.append("")
open(os.path.join(R, "summary.md"), "w").write("\n".join(lines))

# ---- paired per-seed comparison on regret (lower better) -------------------
pw = ["# AGS vs rivals, paired by seed", "",
      "Per seed, compare the reference-CV regret of the picked config. "
      "W = AGS better, T = tie (same score), L = AGS worse.", "",
      "| task | AGS variant | vs | W | T | L |", "|---|---|---|---|---|---|"]
for task, g in runs.groupby("task", sort=False):
    for a in ["ags_default", "ags_no_early_stop"]:
        for b in ["random", "optuna_tpe"]:
            A = g[g.method == a].set_index("seed").regret
            B = g[g.method == b].set_index("seed").regret
            d = (B - A).dropna()
            w, t, l = (d > 1e-12).sum(), (d.abs() <= 1e-12).sum(), (d < -1e-12).sum()
            pw.append(f"| {task} | {a} | {b} | {w} | {t} | {l} |")
open(os.path.join(R, "pairwise.md"), "w").write("\n".join(pw) + "\n")

# ---- anytime regret curves -------------------------------------------------
SERIES = [("ags_no_early_stop", "AGS (no early stop)", "#2a78d6"),
          ("optuna_tpe", "Optuna TPE", "#eb6834"),
          ("random", "Random search", "#1baf7a")]
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"

tasks = list(runs.task.unique())
fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6), dpi=150)
axes = np.atleast_1d(axes)
for ax, task in zip(axes, tasks):
    c = curves[curves.task == task]
    for key, label, col in SERIES:
        m = c[c.method == key].groupby("t").regret.mean()
        ax.plot(m.index, m.values, color=col, lw=2, label=label)
    d = runs[(runs.task == task) & (runs.method == "ags_default")]
    ax.plot(d.n_evals.mean(), d.regret.mean(), "o", ms=8, color="#2a78d6",
            mec="white", mew=2, label="AGS default (stops early)")
    ax.set_title(task, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("unique configs evaluated", color=INK2, fontsize=9)
    ax.grid(True, color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=INK2, labelsize=8)
    ymax = c[c.t >= 8].groupby(["method", "t"]).regret.mean().max()
    ax.set_ylim(0, ymax * 1.1 if ymax > 0 else 1)
axes[0].set_ylabel("mean regret vs grid optimum\n(lower is better)", color=INK2, fontsize=9)
axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
fig.tight_layout()
fig.savefig(os.path.join(R, "anytime.png"), facecolor="white")
print(open(os.path.join(R, "summary.md")).read())
print(open(os.path.join(R, "pairwise.md")).read())
