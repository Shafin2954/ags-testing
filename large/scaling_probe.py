"""
Scaling probe: how AGS construction time / memory and per-step overhead
grow with grid size. Uses a zero-cost fake model so only AGS's own
bookkeeping is measured. Each size runs in a fresh subprocess so peak RSS
is clean; a size that exceeds MEM_LIMIT_GB or TIME_LIMIT_S is reported as such.
"""

import json
import os
import subprocess
import sys

MEM_LIMIT_GB = 6
TIME_LIMIT_S = 300

CHILD = r'''
import json, resource, sys, time
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from ags import AdaptiveGreedySearch

class Fake(BaseEstimator, RegressorMixin):
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)
    def get_params(self, deep=True):
        return {k: v for k, v in self.__dict__.items()}
    def set_params(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)
        return self
    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))

dims, levels, evals = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
grid = {f"p{i}": list(range(levels)) for i in range(dims)}
def scorer(est, X, y):
    return -sum((getattr(est, f"p{i}") - levels / 3) ** 2 for i in range(dims))

X, y = np.zeros((20, 1)), np.zeros(20)
t0 = time.perf_counter()
s = AdaptiveGreedySearch(Fake(**{k: 0 for k in grid}), grid, cv=2, scoring=scorer,
                         max_evaluations=evals, early_stopping_patience=None,
                         pruning_strategy="none")
t_init = time.perf_counter() - t0
t0 = time.perf_counter()
s.fit(X, y)
t_fit = time.perf_counter() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
print(json.dumps(dict(init_s=t_init, fit_s=t_fit, per_eval_ms=1000 * t_fit / s.n_evaluations,
                      peak_rss_gb=rss)))
'''


def run(dims, levels, evals=100):
    size = levels ** dims
    limit = f"ulimit -v {MEM_LIMIT_GB * 1024 * 1024};"
    cmd = f"{limit} {sys.executable} -c '{CHILD.replace(chr(39), chr(34))}' {dims} {levels} {evals}"
    try:
        out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                             timeout=TIME_LIMIT_S)
    except subprocess.TimeoutExpired:
        return dict(dims=dims, levels=levels, grid=size, status=f"timeout>{TIME_LIMIT_S}s")
    if out.returncode != 0:
        err = (out.stderr.strip().splitlines() or ["?"])[-1][:80]
        return dict(dims=dims, levels=levels, grid=size, status=f"FAIL: {err}")
    return dict(dims=dims, levels=levels, grid=size, status="ok", **json.loads(out.stdout))


if __name__ == "__main__":
    rows = []
    for dims, levels in [(3, 10), (4, 10), (5, 10), (6, 10), (4, 32), (7, 10), (5, 32), (8, 10)]:
        r = run(dims, levels)
        print(r, flush=True)
        rows.append(r)
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(os.path.dirname(__file__), "results", "scaling.json"), "w"),
              indent=1)
