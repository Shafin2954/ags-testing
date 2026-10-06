# Large real space: HistGradientBoosting on Covertype (103,680 configs), AGS 3.0.0

Held-out ROC AUC (10,000 unseen rows) of each method's pick. Mean ± sd over seeds.

Rows for optuna_tpe, random are reused from the v2.0.0 run: those methods don't use AGS and are seeded, so re-running them gives the same picks.

| method | @25 | @50 | @100 | @150 | folds @150 | wall s (full run) |
|---|---|---|---|---|---|---|
| AGS 3.0.0 (3 climbers) | 0.8737 ± 0.0127 | 0.8734 ± 0.0133 | 0.8756 ± 0.0097 | 0.8785 ± 0.0087 | 694 | 210 |
| AGS 3.0.0 (1 climber) | 0.8701 ± 0.0093 | 0.8723 ± 0.0063 | 0.8777 ± 0.0071 | 0.8806 ± 0.0041 | 584 | 213 |
| Optuna TPE | 0.8845 ± 0.0030 | 0.8873 ± 0.0010 | 0.8876 ± 0.0006 | 0.8878 ± 0.0008 | 750 | 843 |
| Random search | 0.8781 ± 0.0060 | 0.8790 ± 0.0051 | 0.8818 ± 0.0036 | 0.8832 ± 0.0030 | 750 | 175 |
| AGS 2.0.0 (no early stop) — reference | 0.8748 ± 0.0086 | 0.8780 ± 0.0055 | 0.8810 ± 0.0050 | 0.8808 ± 0.0054 | 694 | 479 |

Picks at 150 evals:

- AGS 3.0.0 (3 climbers), seed 0: test 0.8854 — `{'learning_rate': 0.02, 'max_leaf_nodes': 127, 'max_depth': None, 'min_samples_leaf': 2, 'l2_regularization': 0.1, 'max_features': 0.3, 'max_iter': 200}`
- AGS 3.0.0 (3 climbers), seed 1: test 0.8754 — `{'learning_rate': 0.05, 'max_leaf_nodes': 63, 'max_depth': 8, 'min_samples_leaf': 5, 'l2_regularization': 1.0, 'max_features': 0.5, 'max_iter': 100}`
- AGS 3.0.0 (3 climbers), seed 2: test 0.8647 — `{'learning_rate': 0.02, 'max_leaf_nodes': 127, 'max_depth': None, 'min_samples_leaf': 10, 'l2_regularization': 0.01, 'max_features': 0.7, 'max_iter': 50}`
- AGS 3.0.0 (3 climbers), seed 3: test 0.8818 — `{'learning_rate': 0.1, 'max_leaf_nodes': 127, 'max_depth': None, 'min_samples_leaf': 10, 'l2_regularization': 0.1, 'max_features': 1.0, 'max_iter': 100}`
- AGS 3.0.0 (3 climbers), seed 4: test 0.8852 — `{'learning_rate': 0.2, 'max_leaf_nodes': 63, 'max_depth': None, 'min_samples_leaf': 5, 'l2_regularization': 0.01, 'max_features': 0.5, 'max_iter': 200}`
- AGS 3.0.0 (1 climber), seed 0: test 0.8752 — `{'learning_rate': 0.05, 'max_leaf_nodes': 63, 'max_depth': None, 'min_samples_leaf': 50, 'l2_regularization': 1.0, 'max_features': 1.0, 'max_iter': 200}`
- AGS 3.0.0 (1 climber), seed 1: test 0.8782 — `{'learning_rate': 0.1, 'max_leaf_nodes': 31, 'max_depth': None, 'min_samples_leaf': 5, 'l2_regularization': 0.01, 'max_features': 0.7, 'max_iter': 100}`
- AGS 3.0.0 (1 climber), seed 2: test 0.8861 — `{'learning_rate': 0.05, 'max_leaf_nodes': 127, 'max_depth': None, 'min_samples_leaf': 2, 'l2_regularization': 0.0, 'max_features': 1.0, 'max_iter': 200}`
- AGS 3.0.0 (1 climber), seed 3: test 0.8816 — `{'learning_rate': 0.2, 'max_leaf_nodes': 127, 'max_depth': 8, 'min_samples_leaf': 10, 'l2_regularization': 10.0, 'max_features': 0.7, 'max_iter': 200}`
- AGS 3.0.0 (1 climber), seed 4: test 0.8820 — `{'learning_rate': 0.05, 'max_leaf_nodes': 63, 'max_depth': None, 'min_samples_leaf': 5, 'l2_regularization': 0.01, 'max_features': 0.5, 'max_iter': 200}`
- Optuna TPE, seed 0: test 0.8885 — `{'learning_rate': 0.1, 'max_leaf_nodes': 127, 'max_depth': None, 'min_samples_leaf': 2, 'l2_regularization': 0.01, 'max_features': 0.5, 'max_iter': 200}`
- Optuna TPE, seed 1: test 0.8889 — `{'learning_rate': 0.1, 'max_leaf_nodes': 127, 'max_depth': None, 'min_samples_leaf': 2, 'l2_regularization': 0.0, 'max_features': 0.5, 'max_iter': 200}`
- Optuna TPE, seed 2: test 0.8876 — `{'learning_rate': 0.1, 'max_leaf_nodes': 127, 'max_depth': None, 'min_samples_leaf': 2, 'l2_regularization': 0.01, 'max_features': 1.0, 'max_iter': 200}`
- Optuna TPE, seed 3: test 0.8870 — `{'learning_rate': 0.1, 'max_leaf_nodes': 127, 'max_depth': None, 'min_samples_leaf': 2, 'l2_regularization': 0.0, 'max_features': 0.7, 'max_iter': 200}`
- Optuna TPE, seed 4: test 0.8871 — `{'learning_rate': 0.05, 'max_leaf_nodes': 127, 'max_depth': None, 'min_samples_leaf': 2, 'l2_regularization': 0.01, 'max_features': 0.3, 'max_iter': 200}`
- Random search, seed 0: test 0.8795 — `{'learning_rate': 0.1, 'max_leaf_nodes': 127, 'max_depth': None, 'min_samples_leaf': 10, 'l2_regularization': 0.01, 'max_features': 0.7, 'max_iter': 50}`
- Random search, seed 1: test 0.8856 — `{'learning_rate': 0.05, 'max_leaf_nodes': 127, 'max_depth': None, 'min_samples_leaf': 10, 'l2_regularization': 0.0, 'max_features': 0.5, 'max_iter': 200}`
- Random search, seed 2: test 0.8822 — `{'learning_rate': 0.05, 'max_leaf_nodes': 127, 'max_depth': None, 'min_samples_leaf': 10, 'l2_regularization': 0.0, 'max_features': 0.7, 'max_iter': 100}`
- Random search, seed 3: test 0.8819 — `{'learning_rate': 0.2, 'max_leaf_nodes': 127, 'max_depth': None, 'min_samples_leaf': 2, 'l2_regularization': 0.01, 'max_features': 0.5, 'max_iter': 50}`
- Random search, seed 4: test 0.8870 — `{'learning_rate': 0.05, 'max_leaf_nodes': 127, 'max_depth': None, 'min_samples_leaf': 5, 'l2_regularization': 0.1, 'max_features': 0.5, 'max_iter': 200}`
