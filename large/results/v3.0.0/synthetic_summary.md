# Large-space synthetic benchmark (AGS 3.0.0)

Median over 10 seeds of the **rank percentile** of each method's pick: the fraction of the whole grid that is strictly better (0 = global optimum; 1e-3 = top 0.1%). Lower is better. Fold noise sigma = 0.05 grid-std.

## Grid 16^4 = 65,536 points

| landscape | AGS @50 | AGS @100 | AGS @200 | AGS 1-climber @200 | AGS 2.0.0 @200 | Optuna @50 | Optuna @100 | Optuna @200 | Random @200 |
|---|---|---|---|---|---|---|---|---|---|
| low_eff_dim | 9.2e-02 | 2.7e-02 | 2.0e-02 | 2.5e-02 | 7.8e-03 | 2.0e-03 | 2.0e-03 | 3.9e-03 | 5.9e-03 |
| rastrigin | 3.2e-02 | 2.6e-02 | 1.4e-02 | 1.0e-02 | 1.2e-03 | 1.4e-03 | 1.8e-04 | 1.5e-05 | 4.3e-03 |
| rosenbrock | 4.3e-02 | 3.7e-02 | 3.6e-02 | 1.4e-02 | 6.1e-05 | 5.6e-04 | 1.3e-04 | 1.5e-05 | 4.0e-03 |
| sphere | 1.5e-01 | 9.6e-02 | 6.4e-02 | 2.9e-02 | 5.3e-05 | 3.3e-04 | 2.1e-04 | 7.6e-05 | 7.3e-03 |

## Grid 10^6 = 1,000,000 points

| landscape | AGS @50 | AGS @100 | AGS @200 | AGS 1-climber @200 | AGS 2.0.0 @200 | Optuna @50 | Optuna @100 | Optuna @200 | Random @200 |
|---|---|---|---|---|---|---|---|---|---|
| low_eff_dim | 1.0e-01 | 5.5e-02 | 4.0e-02 | – | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| rastrigin | 7.1e-02 | 3.1e-02 | 3.1e-02 | – | 1.4e-04 | 1.6e-04 | 3.0e-06 | 1.0e-06 | 3.5e-03 |
| rosenbrock | 4.7e-02 | 2.9e-02 | 2.9e-02 | – | 2.0e-06 | 4.0e-04 | 1.1e-05 | 3.0e-06 | 3.9e-03 |
| sphere | 2.4e-02 | 1.9e-02 | 1.9e-02 | – | 2.0e-06 | 3.8e-04 | 1.5e-06 | 2.0e-06 | 4.7e-03 |

## Paired by seed, all landscapes and grids pooled

W = AGS 3.0.0 (3 climbers) pick strictly better, T = equal, L = worse.

| budget | vs Optuna W/T/L | vs Random W/T/L | vs AGS 2.0.0 W/T/L |
|---|---|---|---|
| 50 | 2/1/77 | 18/1/61 | 11/2/67 |
| 100 | 1/3/76 | 16/1/63 | 3/3/74 |
| 200 | 3/2/75 | 9/3/68 | 5/3/72 |
