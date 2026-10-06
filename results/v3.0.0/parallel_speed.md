# Parallel speed, AGS 3.0.0

One run at a time on a 4-core machine, n_climbers=3. Wall seconds, mean of 2 seeds.

| task | budget | n_jobs=1 | n_jobs=3 | speed-up | identical results |
|---|---|---|---|---|---|
| knn_housing | 24 | 1.5 | 1.4 | 1.13x | yes |
| hgb_synth | 40 | 18.4 | 9.7 | 1.89x | yes |
