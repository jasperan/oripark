# Results

> Reference run: `results/run17` (300 self-play blocks, physics v2, flee-BC,
> hindsight shaping, asymmetric wall-climb). Reproduce with
> `python train.py --blocks 300 --out results/run17 --device cpu
> --evader-net 384,384 --evader-lr 4e-4`.

## The skill metric: escape rate on hard arenas

Mixed hard arenas (uniform params 0.7 + 0.85, 150 matches, stochastic
evader vs the frozen adapted chaser — the honest PPO protocol):

| policy | escape rate (150 matches) |
|---|---:|
| random baseline | 37% |
| BC-pretrained (scripted teacher cloned) | 35% |
| **trained (best checkpoint b100)** | **47%** |

The trained evader escapes hard arenas **10 pts above chaotic luck and 12
above its own teacher**. It also sustained the peak through 100 blocks of
the escalating curriculum (the CEM converged to **spike_prob 0.92** —
near-maximum hazard density — with the evader's win-rate signal pulling it
up all the way).

## Training dynamics (run17)

- **BC quality jumped** with the bigger network: demo NLL 0.26 → **0.17**
  (384² vs 256²), the scripted teacher's traversal + flee patterns clone
  into the NN cleanly.
- **Early dominance window**: the BC'd evader + warmup hit **eval-wr 0.93
  and Elo 1310 (block 20) — the highest of any run** — before the chaser
  adapted.
- **Self-play sawtooth**: evader Elo 1310 → 1150 as the chaser adapted
  (it learns the catch task faster every run). Best checkpoint selection
  is therefore the final decision rule, exactly what `select.py` and the
  [tournament](tournament.md) implement.
- Training escapes were flat-but-sustained: 2601 total, 396→448 per
  50-block window (no collapse, no runaway).

## Physics v2 — Ori WotW fidelity (movement)

| change | value | verified |
|---|---|---|
| Wall climb (hold toward a wall) | 170 px/s | climbs a 26-tile tower; **evader-only** (the pursuer can't — Ori lore) |
| Snappy descent | gravity × 1.12 falling | full jump apex unchanged at 144 px (4.5 tiles) |
| Terminal velocity | 1600 px/s | no tile tunnelling at speed |
| Learned behavior shift | — | policies traded wall-jump chains (wj ≈ 4-5/block) for wall climbing (wj ≈ 1-2/block) |

## Visual proof

- `docs/media/evader_improvement*.gif` — three arenas where the **trained
  evader escapes in 319-582 steps** (portal reached, chaser outrun) while
  the **untrained policy flails until timeout/caught**. Same arena, same
  chaser, same seed, rendered in the painterly Ori-style style.
- `docs/media/tournament/*.gif` — **one recorded video per entrant** on a
  common arena: random flails and gets caught, BC hits a hazard, the
  champion (b130) escapes cleanly.
- `docs/media/progress_curve_v4.png` — run17's fixed-arena improvement
  curve; `training_curves_v4.png` — Elo/agility curves.

## Tournament (ranked, combined arena set)

Full table in [tournament.md](tournament.md). Highlights (60 matches each):

| entrant | escape rate |
|---|---:|
| scripted expert (teacher, privileged grid) | 83% |
| **checkpoint b130 (best NN)** | **48%** |
| BC-pretrained | 45% |
| final/best (b100) | 43% |
| random baseline | 37% |

## Reproduce

```bash
python train.py --blocks 300 --out results/run17 --device cpu \
    --evader-net 384,384 --evader-lr 4e-4        # ~60 min on CPU
python progress.py --run results/run17           # improvement curve
python tournament.py --run results/run17         # ranked table + videos
python eval.py --run results/run17 --baseline    # head-to-head report
python gifdemo.py --run results/run17 --out docs/media/hero.gif --params 0.7
```

## Honest caveats

- The **deterministic frozen chaser is partly exploitable by chaos** — a
  random policy escapes 37% of hard arenas just by being unpredictable.
  All escape-rate claims above are made *relative to that baseline* (and
  to the BC teacher), and with 150-match samples (±8% CI).
- Self-play is non-stationary: the final block is usually in a post-peak
  slump because the chaser adapts. The shipped `evader.zip` is the
  best-checkpoint selection (b100, 47% hard-arena), not the final block.
- Across physics versions the absolute numbers are not comparable: the
  wall-climb era raised the random baseline from 17% → 37% (the chaser's
  deterministic pursuit became easier for chaos to dodge), so run14's 48%
  and run17's 47% are *different games* — the skill gap over random is the
  comparable quantity.
