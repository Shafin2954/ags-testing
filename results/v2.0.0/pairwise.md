# AGS vs rivals, paired by seed

Per seed, compare the reference-CV regret of the picked config. W = AGS better, T = tie (same score), L = AGS worse.

| task | AGS variant | vs | W | T | L |
|---|---|---|---|---|---|
| svc_digits | ags_default | random | 1 | 5 | 4 |
| svc_digits | ags_default | optuna_tpe | 0 | 4 | 6 |
| svc_digits | ags_no_early_stop | random | 3 | 5 | 2 |
| svc_digits | ags_no_early_stop | optuna_tpe | 2 | 5 | 3 |
| tree_cancer | ags_default | random | 1 | 0 | 9 |
| tree_cancer | ags_default | optuna_tpe | 3 | 2 | 5 |
| tree_cancer | ags_no_early_stop | random | 3 | 1 | 6 |
| tree_cancer | ags_no_early_stop | optuna_tpe | 5 | 3 | 2 |
| hgb_synth | ags_default | random | 3 | 2 | 5 |
| hgb_synth | ags_default | optuna_tpe | 2 | 2 | 6 |
| hgb_synth | ags_no_early_stop | random | 2 | 3 | 5 |
| hgb_synth | ags_no_early_stop | optuna_tpe | 2 | 2 | 6 |
| knn_housing | ags_default | random | 4 | 6 | 0 |
| knn_housing | ags_default | optuna_tpe | 0 | 9 | 1 |
| knn_housing | ags_no_early_stop | random | 4 | 6 | 0 |
| knn_housing | ags_no_early_stop | optuna_tpe | 0 | 9 | 1 |
