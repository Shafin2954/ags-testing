"""
Diagnosis of the v3.0.0 synthetic-grid regression.

Variants, all with n_climbers=3, n_jobs=1, 200 evaluations:
  default                 v3.0.0 as shipped
  patience_8              climber_patience=8 (public setting)
  uphill_only             PATCHED: a climber only moves when the new point beats its
                          personal best (otherwise it stays and tries another neighbour)
  uphill_only_patience_8  both
  all_three               uphill_only + patience 8 + PATCHED respawn: new climbers start at
                          the unvisited point with the best UCB (mean + k*std) instead of the
                          highest std, so they restart near promising regions
Writes results/v<version>/diagnose_v3.md
"""

import os
import warnings

import numpy as np
from joblib import Parallel, delayed

import synthetic_bench as b

SEEDS = 10
BUDGET = 200
FUNCS = ["sphere", "rosenbrock", "rastrigin", "low_eff_dim"]
GRID = (4, 16)


def uphill_only_update(self, results):
    """Patched Climber update: move only on improvement."""
    stagnated = []
    for climber_id, (state, score) in results.items():
        c = self._climber_by_id[climber_id]
        c.steps_taken += 1
        if score > c.current_score:
            c.current_score, c.current_state, c.stagnation_count = score, state, 0
        else:
            c.stagnation_count += 1
        if c.stagnation_count >= self.climber_patience:
            self._retire(c, "stagnated")
            stagnated.append(c)
    return stagnated


def ucb_spawn(self, exclude):
    """Patched respawn: best UCB instead of highest uncertainty."""
    from ags.core import Climber
    remaining = [s for s in self.all_states if s not in self.evaluated and s not in exclude]
    if not remaining:
        return None
    if len(self.evaluated) >= 3:
        mean, std = self.predict_uncertainty(remaining)
        state = remaining[int(np.argmax(mean + self.exploration_weight * std))]
    else:
        state = remaining[int(self.rng.integers(len(remaining)))]
    c = Climber(id=f"climber_{self._next_climber_idx}", current_state=state)
    self._next_climber_idx += 1
    self.climbers_spawned += 1
    self._climber_by_id[c.id] = c
    return c


def one(variant, func, seed):
    warnings.filterwarnings("ignore")
    import ags.core as core
    if variant.startswith("uphill_only") or variant == "all_three":
        core.AdaptiveGreedySearch._update_climbers = uphill_only_update
    if variant == "all_three":
        core.AdaptiveGreedySearch._spawn_climber = ucb_spawn
    kw = {"climber_patience": 8} if variant.endswith("patience_8") or variant == "all_three" else {}
    d, L = GRID
    F = b.landscape(func, d, L)
    Fs = np.sort(F.ravel())
    tr = b.run_ags(F, seed, BUDGET, **kw)
    pick = max(tr, key=lambda t: t[1])[0]
    return variant, func, seed, np.searchsorted(Fs, F[pick]) / F.size


def main():
    variants = ["default", "patience_8", "uphill_only", "uphill_only_patience_8", "all_three"]
    res = Parallel(n_jobs=os.cpu_count())(
        delayed(one)(v, f, s) for v in variants for f in FUNCS for s in range(SEEDS))
    import pandas as pd
    df = pd.DataFrame(res, columns=["variant", "func", "seed", "rank_pct"])
    t = df.pivot_table(index="func", columns="variant", values="rank_pct", aggfunc="median")
    t = t[variants]
    lines = [f"# Diagnosing the v{b.AGS_VERSION} synthetic regression", "",
             f"Grid {GRID[1]}^{GRID[0]}, {BUDGET} evaluations, {SEEDS} seeds, n_climbers=3. "
             "Median rank percentile of the pick (fraction of grid strictly better; lower is "
             "better). `uphill_only` patches `_update_climbers` so a climber only moves when "
             "it improves. `all_three` also restarts new climbers at the best-UCB unvisited "
             "point instead of the most uncertain one. For reference, AGS 2.0.0 (no early "
             "stop) medians on this grid: sphere 5.3e-05, rosenbrock 6.1e-05, rastrigin "
             "1.2e-03, low_eff_dim 7.8e-03.", "",
             "| landscape | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
    for f, r in t.iterrows():
        lines.append(f"| {f} | " + " | ".join(f"{v:.1e}" for v in r.values) + " |")
    text = "\n".join(lines) + "\n"
    out = os.path.join(b.OUT, "diagnose_v3.md")
    open(out, "w").write(text)
    print(text)


if __name__ == "__main__" and not os.environ.get("SEEDS_ONLY"):
    main()


def seed_positions():
    """Where Max-Min seeding puts the 3 climbers, vs random starting points."""
    from ags import AdaptiveGreedySearch
    d, L = GRID
    grid = {f"p{i}": list(range(L)) for i in range(d)}
    F = b.landscape("sphere", d, L)
    opt = np.unravel_index(F.argmin(), F.shape)
    Fs = np.sort(F.ravel())
    edge, dist, rank, rdist, rrank = [], [], [], [], []
    for seed in range(SEEDS):
        from sklearn.dummy import DummyRegressor
        s = AdaptiveGreedySearch(DummyRegressor(), grid, n_climbers=3, n_jobs=1,
                                 random_state=seed)
        for st in s._get_max_min_seeds(3):
            edge.append(sum(v in (0, L - 1) for v in st) / d)
            dist.append(sum(abs(a - o) for a, o in zip(st, opt)))
            rank.append(np.searchsorted(Fs, F[st]) / F.size)
        rng = np.random.default_rng(seed)
        for _ in range(3):
            st = tuple(int(v) for v in rng.integers(0, L, d))
            rdist.append(sum(abs(a - o) for a, o in zip(st, opt)))
            rrank.append(np.searchsorted(Fs, F[st]) / F.size)
    return (f"Seed positions on sphere {L}^{d} ({SEEDS} seeds x 3 climbers): "
            f"{100 * np.mean(edge):.0f}% of seed coordinates sit on the grid edge (0 or {L - 1}); "
            f"median distance to the optimum {np.median(dist):.0f} steps vs {np.median(rdist):.0f} "
            f"for random points; median rank of the seed point {np.median(rank):.2f} vs "
            f"{np.median(rrank):.2f} for random points (0 = best, 1 = worst).")


if __name__ == "__main__" and os.environ.get("SEEDS_ONLY"):
    line = seed_positions()
    out = os.path.join(b.OUT, "diagnose_v3.md")
    open(out, "a").write("\n" + line + "\n")
    print(line)
