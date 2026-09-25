"""
Large real search space: HistGradientBoosting on Covertype (class 1 vs 2),
7 hyperparameters, 103,680 grid points. Far too big to grid-search, so
there is no exact optimum; methods are compared on the held-out test
score of their pick (10,000 unseen rows) at several budgets, plus cost.

Anytime checkpoints come from a single run per (method, seed): the pick
after t evaluations is scored on test (refit on full train, cached).
"""

import argparse
import os
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

OUT = os.path.join(os.path.dirname(__file__), "results")
SCORING = "roc_auc"
CHECKPOINTS = [25, 50, 100, 150]

GRID = {
    "learning_rate": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
    "max_leaf_nodes": [4, 8, 16, 31, 63, 127],
    "max_depth": [2, 3, 4, 6, 8, None],
    "min_samples_leaf": [2, 5, 10, 20, 50, 100],
    "l2_regularization": [0.0, 0.01, 0.1, 1.0, 10.0],
    "max_features": [0.3, 0.5, 0.7, 1.0],
    "max_iter": [50, 100, 200],
}
NAMES = list(GRID)
EST = HistGradientBoostingClassifier(random_state=0, early_stopping=False)


def data():
    X, y = fetch_covtype(return_X_y=True)
    m = np.isin(y, [1, 2])
    X, y = X[m], (y[m] == 2).astype(int)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    tr, te = idx[:3000], idx[3000:13000]
    return X[tr], X[te], y[tr], y[te]


def params(state):
    return {n: GRID[n][i] for n, i in zip(NAMES, state)}


def cv_eval(Xtr, ytr, state, seed):
    est = clone(EST).set_params(**params(state))
    return cross_val_score(est, Xtr, ytr, scoring=SCORING,
                           cv=StratifiedKFold(5, shuffle=True, random_state=seed)).mean()


def run_ags(D, seed, budget, **kw):
    from ags import AdaptiveGreedySearch
    Xtr, _, ytr, _ = D
    s = AdaptiveGreedySearch(EST, GRID, cv=5, scoring=SCORING, max_evaluations=budget,
                             random_state=seed, **kw).fit(Xtr, ytr)
    return [(h["state"], h["score"], h["n_folds_used"]) for h in s.history]


def run_random(D, seed, budget):
    Xtr, _, ytr, _ = D
    rng = np.random.default_rng(seed)
    sizes = [len(GRID[n]) for n in NAMES]
    flat = rng.choice(int(np.prod(sizes)), size=budget, replace=False)
    states = [tuple(int(v) for v in np.unravel_index(i, sizes)) for i in flat]
    return [(s, cv_eval(Xtr, ytr, s, seed), 5) for s in states]


def run_optuna(D, seed, budget):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    Xtr, _, ytr, _ = D
    cache, trace = {}, []

    def obj(trial):
        s = tuple(trial.suggest_int(n, 0, len(GRID[n]) - 1) for n in NAMES)
        if s not in cache:
            cache[s] = cv_eval(Xtr, ytr, s, seed)
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
    "ags_default": lambda D, s, b: run_ags(D, s, b),
    "ags_no_early_stop": lambda D, s, b: run_ags(D, s, b, early_stopping_patience=None),
    "random": run_random,
    "optuna_tpe": run_optuna,
}


def job(method, seed):
    warnings.filterwarnings("ignore")
    D = data()
    Xtr, Xte, ytr, yte = D
    t0 = time.perf_counter()
    trace = METHODS[method](D, seed, max(CHECKPOINTS))
    wall = time.perf_counter() - t0
    test_cache, rows = {}, []
    best_own, pick, folds = -np.inf, None, 0
    for t, (s, own, nf) in enumerate(trace, 1):
        folds += nf
        if own > best_own:
            best_own, pick = own, s
        if t in CHECKPOINTS or t == len(trace):
            if pick not in test_cache:
                m = clone(EST).set_params(**params(pick)).fit(Xtr, ytr)
                test_cache[pick] = get_scorer(SCORING)(m, Xte, yte)
            rows.append(dict(method=method, seed=seed, budget=t, n_evals=len(trace),
                             folds=folds, own_cv=best_own, test=test_cache[pick],
                             pick=str(params(pick)), wall_total=wall))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    data()  # warm the download cache before forking
    t0 = time.time()
    res = Parallel(n_jobs=args.n_jobs)(
        delayed(job)(m, s) for s in range(args.seeds) for m in METHODS)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(os.path.join(OUT, "real_large.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s")
    print(df.pivot_table(index="method", columns="budget", values="test",
                         aggfunc="mean").to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
