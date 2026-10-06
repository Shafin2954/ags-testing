# AGS vs rivals, paired by seed

Per seed, compare the reference-CV regret of the picked config. W = AGS better, T = tie (same score), L = AGS worse.

| task | AGS variant | vs | W | T | L |
|---|---|---|---|---|---|
| svc_digits | ags_default | random | 5 | 3 | 2 |
| svc_digits | ags_default | optuna_tpe | 3 | 3 | 4 |
| svc_digits | ags_1_climber | random | 2 | 8 | 0 |
| svc_digits | ags_1_climber | optuna_tpe | 2 | 5 | 3 |
| tree_cancer | ags_default | random | 2 | 2 | 6 |
| tree_cancer | ags_default | optuna_tpe | 5 | 3 | 2 |
| tree_cancer | ags_1_climber | random | 2 | 3 | 5 |
| tree_cancer | ags_1_climber | optuna_tpe | 5 | 4 | 1 |
| hgb_synth | ags_default | random | 5 | 1 | 4 |
| hgb_synth | ags_default | optuna_tpe | 4 | 1 | 5 |
| hgb_synth | ags_1_climber | random | 4 | 0 | 6 |
| hgb_synth | ags_1_climber | optuna_tpe | 5 | 0 | 5 |
| knn_housing | ags_default | random | 3 | 1 | 6 |
| knn_housing | ags_default | optuna_tpe | 0 | 4 | 6 |
| knn_housing | ags_1_climber | random | 2 | 3 | 5 |
| knn_housing | ags_1_climber | optuna_tpe | 0 | 3 | 7 |
