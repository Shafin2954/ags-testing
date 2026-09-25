# Locality probe

AGS with early stopping off, 200 evaluations, 10 seeds per row. Share of evaluations (after the 8 random starts) by grid distance from the current best point.

| grid | landscape | 1 step | 2 steps | 3 steps | 4 steps | 5 steps | 6+ steps |
|---|---|---|---|---|---|---|---|
| 16^4 | sphere | 37% | 28% | 33% | 2% | 0% | 0% |
| 16^4 | rosenbrock | 30% | 18% | 38% | 14% | 0% | 0% |
| 16^4 | rastrigin | 19% | 36% | 45% | 0% | 0% | 0% |
| 16^4 | low_eff_dim | 21% | 28% | 38% | 13% | 0% | 0% |
| 8^8 | sphere | 85% | 15% | 0% | 0% | 0% | 0% |
| 8^8 | rosenbrock | 64% | 36% | 0% | 0% | 0% | 0% |
| 8^8 | rastrigin | 44% | 56% | 0% | 0% | 0% | 0% |
| 8^8 | low_eff_dim | 31% | 60% | 9% | 0% | 0% | 0% |

Grid points at exact distance r from one interior point:

| parameters | r=1 | r=2 | r=3 | r=4 |
|---|---|---|---|---|
| 2 | 4 | 8 | 12 | 16 |
| 4 | 8 | 32 | 88 | 192 |
| 7 | 14 | 98 | 462 | 1,666 |
| 8 | 16 | 128 | 688 | 2,816 |
