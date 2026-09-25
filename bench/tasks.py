"""
Benchmark tasks for adaptive-greedy-search (AGS).

Each task = (estimator, discrete param grid, scoring, train/test data).
Grids are chosen to cover different landscape shapes:

  svc_digits      smooth 2-D log grid (classic "AGS should shine" case)
  tree_cancer     large, plateau-heavy, mixed ordinal + categorical grid
  hgb_synth       boosted trees, 4-D, moderately smooth
  knn_housing     regression via Pipeline params, neg_mean_squared_error
"""

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_digits,
    make_classification,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _split(X, y, stratify=True):
    return train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y if stratify else None
    )


def svc_digits():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    X, _, y, _ = train_test_split(X, y, train_size=1000, random_state=0, stratify=y)
    return dict(
        name="svc_digits",
        estimator=SVC(),
        grid={
            "C": list(np.logspace(-2, 3, 11)),
            "gamma": list(np.logspace(-4, 1, 11)),
        },
        scoring="accuracy",
        data=_split(X, y),
    )


def tree_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    return dict(
        name="tree_cancer",
        estimator=DecisionTreeClassifier(random_state=0),
        grid={
            "max_depth": [1, 2, 3, 4, 5, 6, 8, 10, 12, None],
            "min_samples_leaf": [1, 2, 3, 5, 8, 13, 21, 34],
            "min_samples_split": [2, 4, 8, 16, 32],
            "criterion": ["gini", "entropy", "log_loss"],
        },
        scoring="accuracy",
        data=_split(X, y),
    )


def hgb_synth():
    X, y = make_classification(
        n_samples=1500, n_features=20, n_informative=8, flip_y=0.05, random_state=0
    )
    return dict(
        name="hgb_synth",
        estimator=HistGradientBoostingClassifier(random_state=0, early_stopping=False),
        grid={
            "learning_rate": [0.01, 0.03, 0.1, 0.3, 1.0],
            "max_leaf_nodes": [4, 8, 16, 31, 63],
            "l2_regularization": [0.0, 0.1, 1.0, 10.0],
            "max_iter": [25, 50, 100],
        },
        scoring="roc_auc",
        data=_split(X, y),
    )


def knn_housing():
    X, y = fetch_california_housing(return_X_y=True)
    X, _, y, _ = train_test_split(X, y, train_size=3000, random_state=0)
    return dict(
        name="knn_housing",
        estimator=Pipeline([("sc", StandardScaler()), ("knn", KNeighborsRegressor())]),
        grid={
            "knn__n_neighbors": [1, 2, 3, 5, 8, 12, 17, 25, 35, 50, 70, 100],
            "knn__weights": ["uniform", "distance"],
            "knn__p": [1, 2],
        },
        scoring="neg_mean_squared_error",
        data=_split(X, y, stratify=False),
    )


TASKS = {f.__name__: f for f in (svc_digits, tree_cancer, hgb_synth, knn_housing)}


def grid_size(grid):
    return int(np.prod([len(v) for v in grid.values()]))
