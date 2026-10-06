# Benchmark summary

Mean ± sd over seeds. `regret` = grid-optimum reference-CV score minus the reference-CV score of the config each method picked (0 = found grid's best). `test` = held-out score of the picked config. `folds` = CV fits actually run. `wall` = seconds, single core.

## svc_digits

| method | regret | hit optimum | test | evals | folds | wall s |
|---|---|---|---|---|---|---|
| grid_search | 0.0000 | 1/1 | 0.9880 | 121.0 | 605 | 13.2 |
| ags_default | 0.0047 ± 0.0028 | 0/10 | 0.9888 ± 0.0062 | 15.0 | 75 | 1.4 |
| ags_no_early_stop | 0.0024 ± 0.0026 | 4/10 | 0.9892 ± 0.0038 | 40.0 | 177 | 3.3 |
| ags_no_prune_no_stop | 0.0024 ± 0.0026 | 4/10 | 0.9892 ± 0.0038 | 40.0 | 200 | 3.8 |
| ags_gp_no_stop | 0.0024 ± 0.0026 | 4/10 | 0.9892 ± 0.0038 | 40.0 | 175 | 4.7 |
| random | 0.0027 ± 0.0024 | 2/10 | 0.9912 ± 0.0041 | 40.0 | 200 | 4.5 |
| optuna_tpe | 0.0020 ± 0.0024 | 4/10 | 0.9900 ± 0.0043 | 40.0 | 200 | 5.5 |

## tree_cancer

| method | regret | hit optimum | test | evals | folds | wall s |
|---|---|---|---|---|---|---|
| grid_search | 0.0000 | 1/1 | 0.8951 | 1200.0 | 6000 | 36.4 |
| ags_default | 0.0098 ± 0.0073 | 1/10 | 0.9245 ± 0.0208 | 14.1 | 70 | 0.4 |
| ags_no_early_stop | 0.0077 ± 0.0062 | 1/10 | 0.9217 ± 0.0183 | 60.0 | 268 | 1.3 |
| ags_no_prune_no_stop | 0.0087 ± 0.0055 | 1/10 | 0.9252 ± 0.0206 | 60.0 | 300 | 1.5 |
| ags_gp_no_stop | 0.0078 ± 0.0055 | 1/10 | 0.9224 ± 0.0193 | 60.0 | 271 | 3.7 |
| random | 0.0056 ± 0.0030 | 1/10 | 0.9280 ± 0.0132 | 60.0 | 300 | 1.4 |
| optuna_tpe | 0.0113 ± 0.0113 | 0/10 | 0.9238 ± 0.0220 | 60.0 | 300 | 1.7 |

## hgb_synth

| method | regret | hit optimum | test | evals | folds | wall s |
|---|---|---|---|---|---|---|
| grid_search | 0.0000 | 1/1 | 0.9699 | 300.0 | 1500 | 130.1 |
| ags_default | 0.0057 ± 0.0032 | 0/10 | 0.9680 ± 0.0039 | 19.0 | 93 | 8.6 |
| ags_no_early_stop | 0.0054 ± 0.0031 | 0/10 | 0.9670 ± 0.0030 | 40.0 | 180 | 18.5 |
| ags_no_prune_no_stop | 0.0050 ± 0.0034 | 0/10 | 0.9669 ± 0.0029 | 40.0 | 200 | 20.2 |
| ags_gp_no_stop | 0.0052 ± 0.0029 | 1/10 | 0.9666 ± 0.0043 | 40.0 | 180 | 21.5 |
| random | 0.0038 ± 0.0016 | 0/10 | 0.9692 ± 0.0026 | 40.0 | 200 | 16.6 |
| optuna_tpe | 0.0037 ± 0.0018 | 0/10 | 0.9689 ± 0.0030 | 40.0 | 200 | 22.9 |

## knn_housing

| method | regret | hit optimum | test | evals | folds | wall s |
|---|---|---|---|---|---|---|
| grid_search | 0.0000 | 1/1 | -0.3363 | 48.0 | 240 | 3.5 |
| ags_default | 0.0027 ± 0.0035 | 6/10 | -0.3337 ± 0.0034 | 16.8 | 84 | 1.2 |
| ags_no_early_stop | 0.0027 ± 0.0035 | 6/10 | -0.3337 ± 0.0034 | 24.0 | 115 | 1.6 |
| ags_no_prune_no_stop | 0.0027 ± 0.0035 | 6/10 | -0.3337 ± 0.0034 | 24.0 | 120 | 1.6 |
| ags_gp_no_stop | 0.0027 ± 0.0035 | 6/10 | -0.3337 ± 0.0034 | 24.0 | 114 | 2.3 |
| random | 0.0048 ± 0.0033 | 3/10 | -0.3349 ± 0.0065 | 24.0 | 120 | 1.8 |
| optuna_tpe | 0.0020 ± 0.0033 | 7/10 | -0.3343 ± 0.0032 | 24.0 | 120 | 1.9 |
