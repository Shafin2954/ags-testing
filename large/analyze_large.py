"""
Summaries + charts for the large-space round.
  results/v<version>/synthetic_summary.md, synthetic.png
  results/v<version>/real_large_summary.md, real_large.png   (if real_large.csv exists)

Usage: python analyze_large.py [version]   (default: installed ags version)
For v3+, AGS 2.0.0 results (results/v2.0.0) are drawn as a dashed reference.
"""

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ags

VERSION = sys.argv[1] if len(sys.argv) > 1 else ags.__version__
V3 = int(VERSION.split(".")[0]) >= 3
BASE = os.path.join(os.path.dirname(__file__), "results")
R = os.path.join(BASE, f"v{VERSION}")
REF = os.path.join(BASE, "v2.0.0")

INK, INK2, GRIDC, REFC = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
AG = "ags_default" if V3 else "ags_no_early_stop"   # main AGS series
AG_LABEL = f"AGS {VERSION} (3 climbers)" if V3 else "AGS (no early stop)"
SERIES = [(AG, AG_LABEL, "#2a78d6"),
          ("optuna_tpe", "Optuna TPE", "#eb6834"),
          ("random", "Random search", "#1baf7a")]
REF_KEY, REF_LABEL = "ags_no_early_stop", "AGS 2.0.0 (no early stop)"


def style(ax):
    ax.grid(True, color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=INK2, labelsize=8)


def load_syn(d):
    files = sorted(glob.glob(os.path.join(d, "synthetic_*.csv")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files])
    df["gridlabel"] = df.levels.astype(str) + "^" + df.dims.astype(str)
    return df


# ------------------------------------------------------------- synthetic ---
syn = load_syn(R)
ref = load_syn(REF) if V3 else None
BUD = [50, 100, 200]
key = ["func", "gridlabel", "seed", "budget"]

if syn is not None:
    final_default = syn[syn.method == "ags_default"].sort_values("budget").groupby(
        ["func", "gridlabel", "seed"]).tail(1)
    lines = [f"# Large-space synthetic benchmark (AGS {VERSION})", "",
             "Median over 10 seeds of the **rank percentile** of each method's pick: the "
             "fraction of the whole grid that is strictly better (0 = global optimum; 1e-3 = "
             "top 0.1%). Lower is better. Fold noise sigma = 0.05 grid-std.", ""]
    for gl, g in syn.groupby("gridlabel", sort=False):
        size = int(g.grid.iloc[0])
        lines += [f"## Grid {gl} = {size:,} points", ""]
        if V3:
            lines += ["| landscape | AGS @50 | AGS @100 | AGS @200 | AGS 1-climber @200 "
                      "| AGS 2.0.0 @200 | Optuna @50 | Optuna @100 | Optuna @200 | Random @200 |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
        else:
            lines += ["| landscape | AGS default (evals used) | AGS @50 | AGS @100 | AGS @200 "
                      "| Optuna @50 | Optuna @100 | Optuna @200 | Random @200 |",
                      "|---|---|---|---|---|---|---|---|---|"]
        for f, gf in g.groupby("func"):
            med = lambda m, b, src=gf: src[(src.method == m) & (src.budget == b)].rank_pct.median()
            if V3:
                rf = ref[(ref.func == f) & (ref.gridlabel == gl)] if ref is not None else gf[:0]
                lines.append(
                    f"| {f} | {med(AG, 50):.1e} | {med(AG, 100):.1e} | {med(AG, 200):.1e} "
                    f"| {med('ags_1_climber', 200):.1e} | {med(REF_KEY, 200, rf):.1e} "
                    f"| {med('optuna_tpe', 50):.1e} | {med('optuna_tpe', 100):.1e} "
                    f"| {med('optuna_tpe', 200):.1e} | {med('random', 200):.1e} |")
            else:
                fd = final_default[(final_default.func == f) & (final_default.gridlabel == gl)]
                lines.append(
                    f"| {f} | {fd.rank_pct.median():.1e} ({fd.n_evals.median():.0f}) "
                    f"| {med(AG, 50):.1e} | {med(AG, 100):.1e} | {med(AG, 200):.1e} "
                    f"| {med('optuna_tpe', 50):.1e} | {med('optuna_tpe', 100):.1e} "
                    f"| {med('optuna_tpe', 200):.1e} | {med('random', 200):.1e} |")
        lines.append("")

    rivals = [("optuna_tpe", "Optuna", syn), ("random", "Random", syn)]
    if V3 and ref is not None:
        rivals.append((REF_KEY, "AGS 2.0.0", ref))
    lines += ["## Paired by seed, all landscapes and grids pooled", "",
              f"W = {AG_LABEL} pick strictly better, T = equal, L = worse.", "",
              "| budget | " + " | ".join(f"vs {n} W/T/L" for _, n, _ in rivals) + " |",
              "|---|" + "---|" * len(rivals)]
    A = syn[syn.method == AG].set_index(key).rank_pct
    for b in BUD:
        cells = []
        for m, _, src in rivals:
            B = src[src.method == m].set_index(key).rank_pct
            d = (B - A).xs(b, level="budget").dropna()
            cells.append(f"{(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}")
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    open(os.path.join(R, "synthetic_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    funcs = sorted(syn.func.unique())
    grids = list(syn.gridlabel.unique())
    fig, axes = plt.subplots(len(grids), len(funcs),
                             figsize=(3.4 * len(funcs), 2.7 * len(grids)), dpi=150, sharex=True)
    floor = 1e-7
    for r, gl in enumerate(grids):
        for c, f in enumerate(funcs):
            ax = axes[r, c]
            g = syn[(syn.func == f) & (syn.gridlabel == gl)]
            if V3 and ref is not None:
                rg = ref[(ref.func == f) & (ref.gridlabel == gl) & (ref.method == REF_KEY)]
                m = rg[rg.budget.isin(BUD)].groupby("budget").rank_pct.median()
                ax.plot(m.index, np.maximum(m.values, floor), color=REFC, lw=1.5, ls="--",
                        marker="o", ms=3, label=REF_LABEL)
            for key_, label, col in SERIES:
                m = g[(g.method == key_) & g.budget.isin(BUD)].groupby("budget").rank_pct.median()
                ax.plot(m.index, np.maximum(m.values, floor), color=col, lw=2, marker="o",
                        ms=4, label=label)
            if not V3:
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
    fig.suptitle(f"AGS {VERSION}: fraction of grid better than the pick (median, 10 seeds; "
                 f"lower is better; 0 plotted at {floor:g}; lines at the floor overlap = "
                 "found the optimum)", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(R, "synthetic.png"), facecolor="white")

# ------------------------------------------------------------------ real ---
p = os.path.join(R, "real_large.csv")
if os.path.exists(p):
    df = pd.read_csv(p)
    CP = [25, 50, 100, 150]
    rows = list(SERIES)
    if V3:
        rows.insert(1, ("ags_1_climber", f"AGS {VERSION} (1 climber)", None))
    refdf = None
    rp = os.path.join(REF, "real_large.csv")
    if V3 and os.path.exists(rp):
        refdf = pd.read_csv(rp)
        refdf = refdf[refdf.method == REF_KEY]
    out = [f"# Large real space: HistGradientBoosting on Covertype (103,680 configs), "
           f"AGS {VERSION}", "",
           "Held-out ROC AUC (10,000 unseen rows) of each method's pick. Mean ± sd over seeds.",
           ""]
    if "reused_from" in df.columns and df.reused_from.notna().any():
        reused = sorted(df[df.reused_from.notna()].method.unique())
        out += [f"Rows for {', '.join(reused)} are reused from the "
                f"v{df.reused_from.dropna().iloc[0]} run: those methods don't use AGS and are "
                "seeded, so re-running them gives the same picks.", ""]
    out += ["| method | @25 | @50 | @100 | @150 | folds @150 | wall s (full run) |",
            "|---|---|---|---|---|---|---|"]

    def row(g, label):
        cells = []
        for b in CP:
            x = g[g.budget == b].test
            cells.append(f"{x.mean():.4f} ± {x.std():.4f}" if len(x) else "–")
        last = g[g.budget == 150]
        return (f"| {label} | " + " | ".join(cells) +
                f" | {last.folds.mean():.0f} | {g.wall_total.mean():.0f} |")

    for key_, label, _ in rows:
        out.append(row(df[df.method == key_], label))
    if refdf is not None:
        out.append(row(refdf, REF_LABEL + " — reference"))
    if not V3:
        fdft = df[df.method == "ags_default"].sort_values("budget").groupby("seed").tail(1)
        out.append(f"| AGS default (stopped at {fdft.n_evals.mean():.0f} evals avg) "
                   f"| final: {fdft.test.mean():.4f} ± {fdft.test.std():.4f} | | | "
                   f"| {fdft.folds.mean():.0f} | {fdft.wall_total.mean():.0f} |")
    out += ["", "Picks at 150 evals:", ""]
    for key_, label, _ in rows:
        for _, r_ in df[(df.method == key_) & (df.budget == 150)].iterrows():
            out.append(f"- {label}, seed {r_.seed}: test {r_.test:.4f} — `{r_.pick}`")
    open(os.path.join(R, "real_large_summary.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))

    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=150)
    if refdf is not None:
        m = refdf[refdf.budget.isin(CP)].groupby("budget").test.mean()
        ax.plot(m.index, m.values, color=REFC, lw=1.5, ls="--", marker="o", ms=4,
                label=REF_LABEL)
    for key_, label, col in SERIES:
        m = df[(df.method == key_) & df.budget.isin(CP)].groupby("budget").test.mean()
        ax.plot(m.index, m.values, color=col, lw=2, marker="o", ms=5, label=label)
    if not V3:
        ax.plot(fdft.n_evals.mean(), fdft.test.mean(), "o", ms=9, color="#2a78d6",
                mec="white", mew=2, label="AGS default (stops early)")
    style(ax)
    ax.set_xlabel("evaluations", color=INK2, fontsize=9)
    ax.set_ylabel("held-out ROC AUC of pick\n(higher is better)", color=INK2, fontsize=9)
    ax.set_title(f"HGB on Covertype, 103,680-config grid (mean of seeds), AGS {VERSION}",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "real_large.png"), facecolor="white")
