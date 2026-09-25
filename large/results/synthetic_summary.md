# Large-space synthetic benchmark

Median over 10 seeds of the **rank percentile** of each method's pick: the fraction of the whole grid that is strictly better (0 = global optimum; 1e-3 = top 0.1%). Lower is better. Fold noise sigma = 0.05 grid-std.

## Grid 16^4 = 65,536 points

| landscape | AGS default (evals used) | AGS @50 | AGS @100 | AGS @200 | Optuna @50 | Optuna @100 | Optuna @200 | Random @200 |
|---|---|---|---|---|---|---|---|---|
| low_eff_dim | 2.7e-02 (18) | 1.2e-02 | 7.8e-03 | 7.8e-03 | 2.0e-03 | 2.0e-03 | 3.9e-03 | 5.9e-03 |
| rastrigin | 1.3e-02 (18) | 1.3e-02 | 7.4e-03 | 1.2e-03 | 1.4e-03 | 1.8e-04 | 1.5e-05 | 4.3e-03 |
| rosenbrock | 1.5e-02 (21) | 1.1e-03 | 1.2e-04 | 6.1e-05 | 5.6e-04 | 1.3e-04 | 1.5e-05 | 4.0e-03 |
| sphere | 1.6e-02 (27) | 1.8e-03 | 3.1e-04 | 5.3e-05 | 3.3e-04 | 2.1e-04 | 7.6e-05 | 7.3e-03 |

## Grid 10^6 = 1,000,000 points

| landscape | AGS default (evals used) | AGS @50 | AGS @100 | AGS @200 | Optuna @50 | Optuna @100 | Optuna @200 | Random @200 |
|---|---|---|---|---|---|---|---|---|
| low_eff_dim | 3.5e-02 (21) | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| rastrigin | 1.8e-02 (27) | 3.6e-03 | 2.8e-04 | 1.4e-04 | 1.6e-04 | 3.0e-06 | 1.0e-06 | 3.5e-03 |
| rosenbrock | 1.7e-02 (25) | 1.1e-03 | 2.8e-05 | 2.0e-06 | 4.0e-04 | 1.1e-05 | 3.0e-06 | 3.9e-03 |
| sphere | 1.2e-02 (23) | 3.3e-03 | 2.5e-06 | 2.0e-06 | 3.8e-04 | 1.5e-06 | 2.0e-06 | 4.7e-03 |

## Grid 8^8 = 16,777,216 points

| landscape | AGS default (evals used) | AGS @50 | AGS @100 | AGS @200 | Optuna @50 | Optuna @100 | Optuna @200 | Random @200 |
|---|---|---|---|---|---|---|---|---|
| low_eff_dim | 3.1e-02 (16) | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| rastrigin | 2.0e-02 (20) | 1.2e-03 | 3.3e-04 | 3.0e-05 | 1.0e-03 | 2.6e-05 | 3.3e-07 | 6.8e-03 |
| rosenbrock | 1.4e-02 (21) | 2.2e-03 | 2.1e-04 | 6.0e-08 | 5.0e-04 | 5.1e-05 | 2.4e-07 | 4.4e-03 |
| sphere | 3.9e-02 (24) | 6.6e-03 | 9.1e-04 | 3.2e-06 | 2.5e-05 | 6.6e-07 | 5.4e-07 | 2.9e-03 |

## Paired by seed, all landscapes and grids pooled

W = AGS (no early stop) pick strictly better, T = equal, L = worse.

| budget | vs Optuna W/T/L | vs Random W/T/L |
|---|---|---|
| 50 | 26/18/76 | 82/7/31 |
| 100 | 27/26/67 | 89/15/16 |
| 200 | 28/48/44 | 91/20/9 |
