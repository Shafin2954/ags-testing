# Large real space: HistGradientBoosting on Covertype (103,680 configs)

Held-out ROC AUC (10,000 unseen rows) of each method's pick. Mean ± sd over seeds.

| method | @25 | @50 | @100 | @150 | folds @150 | wall s (full run) |
|---|---|---|---|---|---|---|
| AGS (no early stop) | 0.8748 ± 0.0086 | 0.8780 ± 0.0055 | 0.8810 ± 0.0050 | 0.8808 ± 0.0054 | 694 | 479 |
| Optuna TPE | 0.8845 ± 0.0030 | 0.8873 ± 0.0010 | 0.8876 ± 0.0006 | 0.8878 ± 0.0008 | 750 | 843 |
| Random search | 0.8781 ± 0.0060 | 0.8790 ± 0.0051 | 0.8818 ± 0.0036 | 0.8832 ± 0.0030 | 750 | 175 |
| AGS default (stopped at 21 evals avg) | final: 0.8733 ± 0.0076 | | | | 102 | 29 |

Picks at 150 evals:

- AGS (no early stop), seed 0: test 0.8837 — `{'learning_rate': 0.2, 'max_leaf_nodes': 63, 'max_depth': None, 'min_samples_leaf': 5, 'l2_regularization': 0.01, 'max_features': 1.0, 'max_iter': 200}`
- AGS (no early stop), seed 1: test 0.8795 — `{'learning_rate': 0.1, 'max_leaf_nodes': 63, 'max_depth': 8, 'min_samples_leaf': 5, 'l2_regularization': 1.0, 'max_features': 0.5, 'max_iter': 200}`
- AGS (no early stop), seed 2: test 0.8811 — `{'learning_rate': 0.3, 'max_leaf_nodes': 63, 'max_depth': None, 'min_samples_leaf': 10, 'l2_regularization': 0.1, 'max_features': 0.7, 'max_iter': 200}`
- AGS (no early stop), seed 3: test 0.8725 — `{'learning_rate': 0.5, 'max_leaf_nodes': 127, 'max_depth': 6, 'min_samples_leaf': 10, 'l2_regularization': 0.0, 'max_features': 0.3, 'max_iter': 200}`
- AGS (no early stop), seed 4: test 0.8871 — `{'learning_rate': 0.05, 'max_leaf_nodes': 127, 'max_depth': None, 'min_samples_leaf': 2, 'l2_regularization': 0.01, 'max_features': 0.3, 'max_iter': 200}`
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
