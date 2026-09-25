"""
Benchmark: AGS vs GridSearch vs RandomSearch vs Optuna-TPE.

Protocol
--------
1. Truth table: every grid config is scored once with a fixed reference
   5-fold CV (the exact work GridSearchCV does) plus a held-out test score
   (fit on full train split). This IS the grid-search baseline, and it lets
   us score any config a searcher picks on a common yardstick.
2. Each searcher gets the same budget of B *unique* configs and, per seed,
   the same CV splitter AGS builds internally (shuffled (Stratified)KFold,
   random_state=seed). Searchers pick their winner by their OWN scores.
3. Reported per run: reference-CV score of the winner, regret vs grid
   optimum, held-out test score, folds actually fitted, wall time.
   Anytime curves: winner-so-far after t evaluations.

Usage:
    python run_benchmark.py                 # all tasks, 10 seeds
    python run_benchmark.py --tasks svc_digits --seeds 3
"""

import argparse
import os
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone, is_classifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from tasks import TASKS, grid_size

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
BUDGETS = {"svc_digits": 40, "tree_cancer": 60, "hgb_synth": 40, "knn_housing": 24}
REF_CV_SEED = 12345


def configs(grid):
    import itertools
    names = list(grid)
    return names, list(itertools.product(*[range(len(grid[n])) for n in names]))


def to_params(grid, names, state):
    return {n: grid[n][i] for n, i in zip(names, state)}


def splitter(est, seed, k=5):
    cls = StratifiedKFold if is_classifier(est) else KFold
    return cls(n_splits=k, shuffle=True, random_state=seed)


# ----------------------------------------------------------------- truth ----

def _truth_row(task, names, state):
    warnings.filterwarnings("ignore")
    Xtr, Xte, ytr, yte = task["data"]
    est = clone(task["estimator"]).set_params(**to_params(task["grid"], names, state))
    t0 = time.perf_counter()
    cv = cross_val_score(est, Xtr, ytr, cv=splitter(est, REF_CV_SEED),
                         scoring=task["scoring"]).mean()
    cv_time = time.perf_counter() - t0
    test = get_scorer(task["scoring"])(clone(est).fit(Xtr, ytr), Xte, yte)
    return dict(state=str(state), ref_cv=cv, test=test, cv_time=cv_time)


def truth_table(name, n_jobs):
    path = os.path.join(RESULTS, f"truth_{name}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    task = TASKS[name]()
    names, states = configs(task["grid"])
    rows = Parallel(n_jobs=n_jobs)(delayed(_truth_row)(task, names, s) for s in states)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# ------------------------------------------------------------- searchers ----

def run_ags(task, seed, budget, **kw):
    from ags import AdaptiveGreedySearch
    Xtr, _, ytr, _ = task["data"]
    s = AdaptiveGreedySearch(task["estimator"], task["grid"], cv=5,
                             scoring=task["scoring"], max_evaluations=budget,
                             random_state=seed, **kw)
    t0 = time.perf_counter()
    s.fit(Xtr, ytr)
    wall = time.perf_counter() - t0
    trace = [(str(h["state"]), h["score"], h["n_folds_used"]) for h in s.history]
    return trace, wall


def _cv_eval(task, names, state, seed):
    Xtr, _, ytr, _ = task["data"]
    est = clone(task["estimator"]).set_params(**to_params(task["grid"], names, state))
    return cross_val_score(est, Xtr, ytr, cv=splitter(est, seed),
                           scoring=task["scoring"]).mean()


def run_random(task, seed, budget):
    names, states = configs(task["grid"])
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(states), size=budget, replace=False)
    t0 = time.perf_counter()
    trace = [(str(states[i]), _cv_eval(task, names, states[i], seed), 5) for i in pick]
    return trace, time.perf_counter() - t0


def run_optuna(task, seed, budget):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    names, _ = configs(task["grid"])
    cache, trace = {}, []
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))

    def objective(trial):
        # Index-encoded ints: same ordinal information AGS gets.
        state = tuple(trial.suggest_int(n, 0, len(task["grid"][n]) - 1) for n in names)
        if state not in cache:
            cache[state] = _cv_eval(task, names, state, seed)
            trace.append((str(state), cache[state], 5))
        return cache[state]

    t0 = time.perf_counter()
    # Stop at `budget` unique configs (duplicates are free), hard cap on trials.
    for _ in range(budget * 10):
        if len(cache) >= budget:
            break
        study.optimize(objective, n_trials=1)
    return trace, time.perf_counter() - t0


METHODS = {
    "ags_default": lambda t, s, b: run_ags(t, s, b),
    "ags_no_early_stop": lambda t, s, b: run_ags(t, s, b, early_stopping_patience=None),
    "ags_no_prune_no_stop": lambda t, s, b: run_ags(t, s, b, early_stopping_patience=None,
                                                    pruning_strategy="none"),
    "ags_gp_no_stop": lambda t, s, b: run_ags(t, s, b, early_stopping_patience=None,
                                              surrogate_type="gp"),
    "random": run_random,
    "optuna_tpe": run_optuna,
}


def _job(name, method, seed, budget):
    warnings.filterwarnings("ignore")
    task = TASKS[name]()
    trace, wall = METHODS[method](task, seed, budget)
    return dict(task=name, method=method, seed=seed, wall=wall, trace=trace)


# ---------------------------------------------------------------- scoring ---

def summarize(jobs, truth, name):
    ref = dict(zip(truth.state, truth.ref_cv))
    test = dict(zip(truth.state, truth.test))
    best_ref = truth.ref_cv.max()
    grid_pick = truth.loc[truth.ref_cv.idxmax()]
    runs, curves = [], []
    for j in jobs:
        tr = j["trace"]
        best_own, best_state, folds = -np.inf, None, 0
        for t, (state, own, nf) in enumerate(tr, start=1):
            folds += nf
            if own > best_own:
                best_own, best_state = own, state
            curves.append(dict(task=name, method=j["method"], seed=j["seed"], t=t,
                               folds=folds, regret=best_ref - ref[best_state]))
        runs.append(dict(task=name, method=j["method"], seed=j["seed"],
                         n_evals=len(tr), folds=folds, wall=j["wall"],
                         own_best=best_own, ref_cv=ref[best_state],
                         regret=best_ref - ref[best_state], test=test[best_state],
                         hit_optimum=bool(np.isclose(ref[best_state], best_ref))))
    runs.append(dict(task=name, method="grid_search", seed=-1, n_evals=len(truth),
                     folds=5 * len(truth), wall=truth.cv_time.sum(),
                     own_best=best_ref, ref_cv=best_ref, regret=0.0,
                     test=grid_pick.test, hit_optimum=True))
    return pd.DataFrame(runs), pd.DataFrame(curves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    all_runs, all_curves = [], []
    for name in args.tasks:
        t0 = time.time()
        truth = truth_table(name, args.n_jobs)
        budget = BUDGETS[name]
        print(f"[{name}] grid={grid_size(TASKS[name]()['grid'])} budget={budget} "
              f"truth ready ({time.time() - t0:.0f}s)", flush=True)
        jobs = Parallel(n_jobs=args.n_jobs)(
            delayed(_job)(name, m, s, budget) for m in METHODS for s in range(args.seeds))
        runs, curves = summarize(jobs, truth, name)
        all_runs.append(runs)
        all_curves.append(curves)
        print(runs.groupby("method")[["regret", "test", "n_evals", "folds", "wall"]]
              .mean().round(4).to_string(), flush=True)
        print(f"[{name}] done in {time.time() - t0:.0f}s\n", flush=True)

    pd.concat(all_runs).to_csv(os.path.join(RESULTS, "runs.csv"), index=False)
    pd.concat(all_curves).to_csv(os.path.join(RESULTS, "curves.csv"), index=False)


if __name__ == "__main__":
    main()
