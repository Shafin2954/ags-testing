# Diagnosing the v3.0.0 synthetic regression

Grid 16^4, 200 evaluations, 10 seeds, n_climbers=3. Median rank percentile of the pick (fraction of grid strictly better; lower is better). `uphill_only` patches `_update_climbers` so a climber only moves when it improves. `all_three` also restarts new climbers at the best-UCB unvisited point instead of the most uncertain one. For reference, AGS 2.0.0 (no early stop) medians on this grid: sphere 5.3e-05, rosenbrock 6.1e-05, rastrigin 1.2e-03, low_eff_dim 7.8e-03.

| landscape | default | patience_8 | uphill_only | uphill_only_patience_8 | all_three |
|---|---|---|---|---|---|
| low_eff_dim | 2.0e-02 | 7.8e-02 | 1.2e-02 | 9.8e-03 | 9.8e-03 |
| rastrigin | 1.4e-02 | 3.5e-02 | 5.0e-03 | 1.7e-03 | 1.7e-03 |
| rosenbrock | 3.6e-02 | 4.9e-02 | 2.0e-02 | 6.9e-05 | 6.9e-05 |
| sphere | 6.4e-02 | 1.4e-01 | 6.9e-02 | 4.3e-03 | 4.3e-03 |

Seed positions on sphere 16^4 (10 seeds x 3 climbers): 61% of seed coordinates sit on the grid edge (0 or 15); median distance to the optimum 25 steps vs 18 for random points; median rank of the seed point 0.84 vs 0.51 for random points (0 = best, 1 = worst).

Note: `all_three` equals `uphill_only_patience_8` because with patience 8 and uphill-only moves no climber was retired within 200 evaluations (0 respawns in the runs checked), so the respawn patch never triggered. The remaining gap to 2.0.0 on sphere comes from where the climbers start (line above): starting from grid edges, 3 climbers sharing 200 single-step evaluations don't reach an interior optimum.
