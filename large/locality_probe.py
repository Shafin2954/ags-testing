"""
Locality probe: how far from the current best does AGS look?

For each evaluation after the random init, record the grid (L1 / step)
distance between the evaluated point and the incumbent best at that moment.
Also prints how many grid points sit at each exact distance in d dimensions,
which is why the "up to 4 rings" plateau escape rarely gets past ring 2.

Writes results/locality.md
"""

import os
import warnings
from collections import Counter
from math import comb

import numpy as np
from joblib import Parallel, delayed

import synthetic_bench as b

OUT = os.path.join(os.path.dirname(__file__), "results")
GRIDS = [(4, 16), (8, 8)]
SEEDS = 10
BUDGET = 200
CAP = 6  # distances >= CAP are pooled


def ring_size(d, r):
    """Number of integer points at exact L1 distance r from an interior point in Z^d."""
    return sum(comb(d, k) * comb(r - 1, k - 1) * 2 ** k for k in range(1, min(d, r) + 1))


def one(func, d, L, seed):
    warnings.filterwarnings("ignore")
    F = b.landscape(func, d, L)
    trace = b.run_ags(F, seed, BUDGET, early_stopping_patience=None)
    dists, best, best_score = [], None, -np.inf
    for i, (s, score, _) in enumerate(trace):
        if i >= 8:  # skip AGS's 8 random initial points
            dists.append(min(sum(abs(x - y) for x, y in zip(s, best)), CAP))
        if score > best_score:
            best_score, best = score, s
    return func, d, L, dists


def main():
    jobs = [(f, d, L, s) for d, L in GRIDS for f in b.FUNCS for s in range(SEEDS)]
    res = Parallel(n_jobs=2)(delayed(one)(*j) for j in jobs)
    agg = {}
    for func, d, L, dists in res:
        agg.setdefault((d, L, func), Counter()).update(dists)

    cols = list(range(1, CAP + 1))
    lines = ["# Locality probe", "",
             f"AGS with early stopping off, {BUDGET} evaluations, {SEEDS} seeds per row. "
             "Share of evaluations (after the 8 random starts) by grid distance from the "
             "current best point.", "",
             "| grid | landscape | " + " | ".join(
                 f"{c}{'+' if c == CAP else ''} step{'s' if c > 1 else ''}" for c in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for (d, L, func), cnt in agg.items():
        tot = sum(cnt.values())
        lines.append(f"| {L}^{d} | {func} | " +
                     " | ".join(f"{100 * cnt.get(c, 0) / tot:.0f}%" for c in cols) + " |")
    lines += ["", "Grid points at exact distance r from one interior point:", "",
              "| parameters | r=1 | r=2 | r=3 | r=4 |", "|---|---|---|---|---|"]
    for d in (2, 4, 7, 8):
        lines.append(f"| {d} | " + " | ".join(f"{ring_size(d, r):,}" for r in range(1, 5)) + " |")
    text = "\n".join(lines) + "\n"
    open(os.path.join(OUT, "locality.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
