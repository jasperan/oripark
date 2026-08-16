# Tournament — evader policies vs the frozen chaser

Run: `results/run18` · 30 fixed arenas (adversary mean `[0.58 0.22 0.46 0.16 0.3  0.06]`) · 60 matches each · stochastic evader, deterministic chaser (the honest PPO protocol).

| rank | entrant | escape rate | caught | avg len |
|---|---|---:|---:|---:|
| 1 | checkpoint b000 | 60% (36/60) | 24 | 188 |
| 2 | checkpoint b060 | 47% (28/60) | 32 | 116 |
| 3 | checkpoint b399 | 45% (27/60) | 33 | 161 |
| 4 | checkpoint b230 | 40% (24/60) | 36 | 143 |
| 5 | BC-pretrained | 38% (23/60) | 37 | 205 |
| 6 | random-init | 35% (21/60) | 39 | 156 |
| 7 | checkpoint b290 | 35% (21/60) | 39 | 136 |
| 8 | checkpoint b170 | 33% (20/61) | 41 | 113 |
| 9 | checkpoint b110 | 30% (18/60) | 42 | 104 |
| 10 | checkpoint b340 | 30% (18/60) | 42 | 122 |
| 11 | final/best | 30% (18/60) | 42 | 122 |
| 12 | scripted expert | 2% (1/60) | 10 | 85 |

## Recorded videos

Each entrant plays the same arena (seed 5100) with the same chaser; the clip shows its natural behavior. Watch the arc: random flails, BC traverses, checkpoints gain evasion, the best checkpoint escapes.

### #1 checkpoint b000 — escaped
![checkpoint b000](media/tournament_run18/entrant_01_checkpoint b000.gif)

### #2 checkpoint b060 — caught
![checkpoint b060](media/tournament_run18/entrant_02_checkpoint b060.gif)

### #3 checkpoint b399 — caught
![checkpoint b399](media/tournament_run18/entrant_03_checkpoint b399.gif)

### #4 checkpoint b230 — escaped
![checkpoint b230](media/tournament_run18/entrant_04_checkpoint b230.gif)

### #5 BC-pretrained — caught
![BC-pretrained](media/tournament_run18/entrant_05_BC-pretrained.gif)

### #6 random-init — escaped
![random-init](media/tournament_run18/entrant_06_random-init.gif)

### #7 checkpoint b290 — timeout
![checkpoint b290](media/tournament_run18/entrant_07_checkpoint b290.gif)

### #8 checkpoint b170 — escaped
![checkpoint b170](media/tournament_run18/entrant_08_checkpoint b170.gif)

### #9 checkpoint b110 — timeout
![checkpoint b110](media/tournament_run18/entrant_09_checkpoint b110.gif)

### #10 checkpoint b340 — caught
![checkpoint b340](media/tournament_run18/entrant_10_checkpoint b340.gif)

### #11 final/best — caught
![final/best](media/tournament_run18/entrant_11_final_best.gif)

## Reading the table

- **random-init**: the untrained policy — lower bound.
- **BC-pretrained**: the scripted expert's traversal cloned into the NN.
- **checkpoint bN**: evader mid-training (self-play is non-stationary, so the best checkpoint often beats the final block).
- **final/best**: `evader.zip` (select.py output).
- **scripted expert**: reference player (BFS waypoint + flee), not an NN.

Escape rate is measured on the run's own (hard) arena distribution, so numbers are comparable within a run but not across runs with different adversary means.