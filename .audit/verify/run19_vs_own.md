# Hard-arena skill test (`results/run19`, stochastic chaser)

100 matches, stochastic evader, mixed 0.7/0.85 arenas, frozen chaser (`stoch`).

| policy | escape rate | caught | timeout | avg len |
|---|---:|---:|---:|---:|
| random | 40.0% (40/100) | 13 | 47 | 210 |
| trained (best) | 38.0% (38/100) | 19 | 43 | 221 |
| BC-pretrained | 27.0% (27/100) | 18 | 55 | 241 |

Caveat: a deterministic frozen chaser is partly exploitable by chaos;
the stochastic-chaser number is the honest one.