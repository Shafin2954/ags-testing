"""
v3 parallelism check: wall time of one AGS run with n_jobs=1 vs n_jobs=3
(n_climbers=3, so a batch has up to 3 candidates), on a cheap and an
expensive model. Runs alone on the machine; results must be identical.

Writes results/v<version>/parallel_speed.md
"""

import os
import time
import warnings

import numpy as np

import ags
from ags import AdaptiveGreedySearch
from tasks import TASKS

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", f"v{ags.__version__}")
CASES = [("knn_housing", 24), ("hgb_synth", 40)]
SEEDS = [0, 1]


def run(task, budget, seed, n_jobs):
    Xtr, _, ytr, _ = task["data"]
    s = AdaptiveGreedySearch(task["estimator"], task["grid"], cv=5, scoring=task["scoring"],
                             max_evaluations=budget, random_state=seed, n_climbers=3,
                             n_jobs=n_jobs)
    t0 = time.perf_counter()
    s.fit(Xtr, ytr)
    return time.perf_counter() - t0, [h["state"] for h in s.history], s.best_score


def main():
    lines = [f"# Parallel speed, AGS {ags.__version__}", "",
             "One run at a time on a 4-core machine, n_climbers=3. Wall seconds, mean of "
             f"{len(SEEDS)} seeds.", "",
             "| task | budget | n_jobs=1 | n_jobs=3 | speed-up | identical results |",
             "|---|---|---|---|---|---|"]
    for name, budget in CASES:
        task = TASKS[name]()
        w1, w3, same = [], [], True
        for seed in SEEDS:
            t1, h1, b1 = run(task, budget, seed, 1)
            t3, h3, b3 = run(task, budget, seed, 3)
            w1.append(t1)
            w3.append(t3)
            same &= (h1 == h3) and (b1 == b3)
        lines.append(f"| {name} | {budget} | {np.mean(w1):.1f} | {np.mean(w3):.1f} | "
                     f"{np.mean(w1) / np.mean(w3):.2f}x | {'yes' if same else 'NO'} |")
    text = "\n".join(lines) + "\n"
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "parallel_speed.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
