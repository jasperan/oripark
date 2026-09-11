# Hard-arena skill test (`results/run19`, stochastic chaser)

100 matches, stochastic evader, mixed 0.7/0.85 arenas, frozen chaser (`stoch`).

| policy | escape rate | caught | timeout | avg len |
|---|---:|---:|---:|---:|
| trained (best) | 44.0% (44/100) | 23 | 33 | 234 |
| random | 38.0% (38/100) | 23 | 39 | 231 |
| BC-pretrained | 21.0% (21/100) | 26 | 53 | 255 |

Caveat: a deterministic frozen chaser is partly exploitable by chaos;
the stochastic-chaser number is the honest one.