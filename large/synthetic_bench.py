"""
Large-space benchmark on synthetic landscapes with EXACT ground truth.

Why synthetic: real models on 10^5-10^7-point grids can't be exhaustively
scored, so "how close to the optimum" is unknowable. Here the objective is a
known function on a discrete grid, so every pick gets an exact rank.

Each grid point's "CV fold score" = -f(x) + Gaussian noise (sigma in units of
the grid-wide std of f), mimicking CV noise. AGS runs through its normal
fit() path via a zero-cost fake estimator + callable scorer; random and
Optuna TPE use the identical noisy fold scores.

Landscapes (f is minimised; score = -f normalised):
  sphere       smooth, unimodal, off-centre optimum
  rosenbrock   curved narrow valley (log1p-scaled)
  rastrigin    rugged / multimodal on the grid
  low_eff_dim  only 2 of d parameters matter (common in real HPO)

Metric: rank percentile of the pick = fraction of the WHOLE grid that is
strictly better (0 = found the global optimum; 0.001 = top 0.1%).
"""

import argparse
import os
import time
import warnings
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

OUT = os.path.join(os.path.dirname(__file__), "results")
SIGMA = 0.05
GRIDS = [(4, 16), (6, 10), (8, 8)]          # (dims, levels): 65k, 1M, 16.7M points
FUNCS = ["sphere", "rosenbrock", "rastrigin", "low_eff_dim"]
BUDGETS = [50, 100, 200]


# ------------------------------------------------------------- landscapes ---

def landscape(func, d, L):
    """Return F: ndarray of shape (L,)*d with the true objective (to minimise)."""
    rng = np.random.default_rng(zlib.crc32(f"{func}{d}{L}".encode()))
    shape1 = lambda i: [L if j == i else 1 for j in range(d)]
    F = np.zeros((L,) * d)
    if func in ("sphere", "low_eff_dim", "rastrigin"):
        lo, hi = (-5.12, 5.12)
        x = np.linspace(lo, hi, L)
        c = rng.uniform(-3, 3, d)
        active = range(2) if func == "low_eff_dim" else range(d)
        for i in active:
            z = x - c[i]
            term = z ** 2 if func != "rastrigin" else z ** 2 - 10 * np.cos(2 * np.pi * z) + 10
            F = F + term.reshape(shape1(i))
    elif func == "rosenbrock":
        x = np.linspace(-2, 2, L)
        for i in range(d - 1):
            a = x.reshape(shape1(i))
            b = x.reshape(shape1(i + 1))
            F = F + 100 * (b - a ** 2) ** 2 + (1 - a) ** 2
        F = np.log1p(F)
    return (F - F.min()) / F.std()


def fold_score(F, state, fold, seed):
    h = zlib.crc32(np.asarray((seed, fold) + tuple(state), dtype=np.int64).tobytes())
    return -F[state] + SIGMA * np.random.default_rng(h).standard_normal()


def cv_mean(F, state, seed, k=5):
    return float(np.mean([fold_score(F, state, f, seed) for f in range(k)]))


# -------------------------------------------------------------- searchers ---

def run_ags(F, seed, budget, **kw):
    from sklearn.base import BaseEstimator, RegressorMixin
    from ags import AdaptiveGreedySearch

    d, L = F.ndim, F.shape[0]

    class Fake(BaseEstimator, RegressorMixin):
        def __init__(self, **p):
            self.__dict__.update(p)

        def get_params(self, deep=True):
            return dict(self.__dict__)

        def set_params(self, **p):
            self.__dict__.update(p)
            return self

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    def scorer(est, X, y):
        state = tuple(getattr(est, f"p{i}") for i in range(d))
        fold = int(X[:, 0].min())           # rows are 0..4, one per fold
        return fold_score(F, state, fold, seed)

    grid = {f"p{i}": list(range(L)) for i in range(d)}
    X, y = np.arange(5, dtype=float).reshape(-1, 1), np.zeros(5)
    s = AdaptiveGreedySearch(Fake(**{k: 0 for k in grid}), grid, cv=5, scoring=scorer,
                             max_evaluations=budget, random_state=seed, **kw)
    # Force fold i to be exactly row i so the scorer knows the fold index.
    from sklearn.model_selection import KFold
    s._cv_splitter = KFold(5)
    s.fit(X, y)
    return [(h["state"], h["score"], h["n_folds_used"]) for h in s.history]


def run_random(F, seed, budget):
    rng = np.random.default_rng(seed)
    flat = rng.choice(F.size, size=budget, replace=False)
    states = [tuple(int(v) for v in np.unravel_index(i, F.shape)) for i in flat]
    return [(s, cv_mean(F, s, seed), 5) for s in states]


def run_optuna(F, seed, budget):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    d, L = F.ndim, F.shape[0]
    cache, trace = {}, []

    def obj(trial):
        s = tuple(trial.suggest_int(f"p{i}", 0, L - 1) for i in range(d))
        if s not in cache:
            cache[s] = cv_mean(F, s, seed)
            trace.append((s, cache[s], 5))
        return cache[s]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    for _ in range(budget * 10):
        if len(cache) >= budget:
            break
        study.optimize(obj, n_trials=1)
    return trace


METHODS = {
    "ags_default": lambda F, s, b: run_ags(F, s, b),
    "ags_no_early_stop": lambda F, s, b: run_ags(F, s, b, early_stopping_patience=None),
    "random": run_random,
    "optuna_tpe": run_optuna,
}


def job(func, d, L, method, seed):
    warnings.filterwarnings("ignore")
    F = landscape(func, d, L)
    Fs = np.sort(F.ravel())
    t0 = time.perf_counter()
    trace = METHODS[method](F, seed, max(BUDGETS))
    wall = time.perf_counter() - t0
    rows, best_own, pick, folds = [], -np.inf, None, 0
    for t, (s, own, nf) in enumerate(trace, 1):
        folds += nf
        if own > best_own:
            best_own, pick = own, s
        if t in BUDGETS or t == len(trace):
            pct = np.searchsorted(Fs, F[pick], side="left") / F.size
            rows.append(dict(func=func, dims=d, levels=L, grid=F.size, method=method,
                             seed=seed, budget=t, n_evals=len(trace), folds=folds,
                             rank_pct=pct, found_opt=bool(F[pick] == 0.0), wall=wall))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for d, L in GRIDS:
        t0 = time.time()
        jobs = [(f, d, L, m, s) for f in FUNCS for m in METHODS for s in range(args.seeds)]
        # 16.7M-point grids: AGS holds ~1.8 GB each, so fewer workers.
        n = args.n_jobs if L ** d < 5_000_000 else 2
        res = Parallel(n_jobs=n)(delayed(job)(*j) for j in jobs)
        df = pd.DataFrame([r for rows in res for r in rows])
        df.to_csv(os.path.join(OUT, f"synthetic_{d}x{L}.csv"), index=False)
        print(f"grid {L}^{d} = {L ** d:,} done in {time.time() - t0:.0f}s", flush=True)
        print(df[df.budget.isin(BUDGETS)].pivot_table(
            index=["func", "method"], columns="budget", values="rank_pct",
            aggfunc="median").to_string(float_format=lambda v: f"{v:.2e}"), flush=True)


if __name__ == "__main__":
    main()
