# Results

## The skill metric (reproducible)

`python hardarena.py --run results/runN --matches 100 --chaser stoch --set 07`

- **Arenas**: uniform-hard terrain params 0.7 (or a 2:1 mix of 0.7/0.85),
  fixed seeds (5000+k), **filtered by the scripted-expert oracle** — an
  arena only counts if a competent player can actually win it.
- **Chaser**: frozen, **stochastic** (action sampling). A deterministic
  frozen chaser is partly exploitable by chaos; sampling closes the hole.
- **Evader**: stochastic, **identical seeded draws for every entrant**
  (policy differences only), so cross-run comparisons are valid.
- **Cross-chaser control**: `--chaser-run results/runA` evaluates a run's
  evaders against another run's frozen chaser — the 3×3 matrix below
  removes the "weaker chaser inflates escape" confound.
- **Random-population floor**: `--random-seeds K` evaluates K independent
  deterministic random inits and reports mean/min/max — the chaotic-init
  baseline as a measured distribution instead of one lucky draw.

## The controlled 3×3 matrix (trained evaders, 100 matches each)

| evader \ frozen chaser | run17 ch | run19 ch | run20 ch | row mean |
|---|---:|---:|---:|---:|
| run17 trained (b299, old arch) | 31% | 25% | 34% | 30% |
| **run19 trained (b270, new arch)** | **44%** | **38%** | **62%** | **48%** |
| run20 trained (b170, new arch) | 35% | 32% | 54% | 40% |

Every cell is the escape rate of a *trained* evader against a *frozen*
chaser, same arena set, same seeded draws. Reads:

1. **The architectural gain generalizes**: both new-arch runs beat the old
   arch on all three chasers (run19 +13/+13/+28; run20 +4/+7/+20).
2. **Magnitude is seed-dependent**: run19's trained is the strongest
   evader this project has produced (44% against the strongest chaser;
   62% against the weakest).
3. **Chaser strength matters**: run20's chaser ended weak (final Elo 1180
   vs run19's 1216), inflating the whole run20 column — random inits
   escape 45-57% there. The matrix, not the own-chaser number, is the
   honest read.

## The random floor is ~4-8%, not 17-45%

`--random-seeds 6` (six deterministic inits per run, same draws):

| run | floor mean | floor min | floor max | trained (own chaser) | edge over floor |
|---|---:|---:|---:|---:|---:|
| run17 | 4.3% | 0% | 20% (its own init) | 31% | **+27** |
| run19 | 6.9% | 0% | 40% (its own init) | 38% | **+31** |
| run20 | 7.7% | 0% | 45% (its own init) | 57% | **+49** |

Every run's saved `pool_ev_0` was a **chaos-lucky init draw** (17-45%
escape), and earlier docs reported those draws as "the random baseline".
Most random inits score 0-2%. Measured properly, the trained policy's
edge over the true floor is **+27 to +49 points** — the single-init
"chaos is strong, trained barely beats random" framing was an artifact
of measuring one lucky init per run.

## run summaries

### run19 / run20 — the new architecture (forward-biased patch + escape rewards)

run17's recipe plus: a **forward-biased 19×10 observation patch** (5
behind / 13 ahead / 7 up / 2 down — the old centered 13×9 showed only 4
tiles up, hiding the landing zone at the 4.5-tile jump apex) and
**escape-dominant rewards** (portal 10.0 vs timeout −10.0 — the old 3.0
escape bonus was smaller than the max milestone haul, teaching "run
right" over "reach the portal").

- run19 (seed 3): evader peak Elo 1352; ships b270 (protocol-selected);
  row-mean 48% across chasers.
- run20 (seed 7): evader peak Elo 1370; ships b170 (protocol-selected);
  row-mean 40% across chasers.

### run17 (old arch, 384² net)

Ships b299. Cross-chaser row mean 30%.

### run18 (gap-sprinkling + 512²) — negative result

Gap-sprinkling + capacity did not transfer ([E12](EXPERIMENTS.md)); its
own terrain drifted so far the scripted expert collapsed to 2% on its
tournament set. Superseded.

## Training dynamics

- run19: BC NLL 0.15; evader peak Elo 1352 / eval-wr 1.00 at block 61
  before the chaser adapted; final evader 1183 vs chaser 1216.
- run20: evader peak Elo 1370; final evader 1221 vs chaser 1180 — the
  chaser never re-adapted, which is why the run20 column is inflated.
- Both: classic sawtooth; CEM converged to spike-heavy mu; final-block
  numbers are post-peak slump — **selection is by protocol, not block**.

## Physics v2 — Ori WotW fidelity (movement)

| change | value | verified |
|---|---|---|
| Wall climb (hold toward a wall) | 170 px/s | climbs a 26-tile tower; **evader-only** (the pursuer can't — Ori lore) |
| Snappy descent | gravity × 1.12 falling | full-jump apex unchanged at 144 px (4.5 tiles) |
| Terminal velocity | 1600 px/s | no tile tunnelling at speed |
| Learned behavior shift | — | policies traded wall-jump chains (wj ≈ 4-5/block) for climbing (wj ≈ 1-2/block) |

## Visual proof

- `docs/media/evader_improvement*.gif` — run19 trained (b270) escapes in
  197-454 steps on hard arenas where the untrained policy is caught or
  times out.
- `docs/media/tournament_run19/*.gif`, `docs/media/tournament_run20/*.gif`
  — one recorded video per entrant on a common arena.
- `docs/media/progress_curve_v4.png` / `training_curves_v4.png` — fixed-
  arena curve and Elo curves.

## Reproduce

```bash
python train.py --blocks 300 --out results/run20 --evader-net 384,384 \
    --evader-lr 4e-4 --seed 7 --device cuda       # ~50 min on CUDA
python hardarena.py --run results/run20 --matches 100 --chaser stoch --set 07
python hardarena.py --run results/run20 --random-seeds 6    # floor distribution
python hardarena.py --run results/run20 --chaser-run results/run17 \
    --matches 100 --chaser stoch --set 07        # cross-chaser control
python tournament.py --run results/run20         # ranked table + videos
python gifdemo.py --run results/run20 --out docs/media/hero.gif --params 0.7
```

## Caveats

- ±8-10% CI at 100 matches; treat single-digit margins as directional.
- The oracle filter (scripted expert) is itself a skill assumption — it
  marks ~half of hard arenas winnable.
- Selection is protocol-based (07 stoch, own chaser, seeded): run19 ships
  b270, run20 ships b170; within-terrain tournament champions can differ
  (e.g. run19 b210 at 68%).
- Self-play non-stationarity: policies and numbers are run-specific; the
  ranking rules (protocol + seeded draws + selection + population floor)
  are the transferable contribution. Each run's exact config is frozen in
  its `params.json` (local) and archived at `docs/configs/run*_params.json`
  (tracked) — every metric stays one command away.
