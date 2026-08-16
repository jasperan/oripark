# Results

## The skill metric (reproducible)

`python hardarena.py --run results/runN --matches 100 --chaser stoch --set 07`

- **Arenas**: uniform-hard terrain params 0.7 (or a 2:1 mix of 0.7/0.85),
  fixed seeds (5000+k), **filtered by the scripted-expert oracle** — an
  arena only counts if a competent player can actually win it.
- **Chaser**: frozen, **stochastic** (action sampling). A deterministic
  frozen chaser is partly exploitable by chaos; sampling closes the hole.
- **Evader**: stochastic, identical seeded draws for every entrant (policy
  differences only), so cross-run comparisons are valid.
- **Cross-chaser control**: `--chaser-run results/runA` evaluates a run's
  evaders against another run's frozen chaser — the 2×2 matrix below
  removes the "weaker chaser inflates escape" confound.

### run19 — the architectural lever (`results/run19`, 300 blocks, 384² net)

run17's recipe plus: (1) a **forward-biased 19×10 observation patch**
(5 behind / 13 ahead / 7 up / 2 down — the old centered 13×9 showed only
4 tiles up, hiding the landing zone at the 4.5-tile jump apex), and
(2) **escape-dominant rewards** (portal 10.0 vs timeout −10.0 — the old
3.0 escape bonus was smaller than the max milestone haul, so the policy
was implicitly taught "run right" over "reach the portal").

**Hard-arena escape, 100 matches, stochastic chaser, seeded draws:**

| evader \ frozen chaser | run17 chaser | run19 chaser |
|---|---:|---:|
| random (run17 init) | 20% | 30% |
| BC-pretrained (run17) | 21% | 16% |
| **run17 trained (b299)** | **31%** | **25%** |
| random (run19 init) | 38% | 40% |
| BC-pretrained (run19) | 21% | 27% |
| **run19 trained (b270 → `evader.zip`)** | **44%** | **38%** |

**The trained evader improved +13 pts against both frozen chasers** (31→44
vs run17's chaser, 25→38 vs run19's). The run19 random init is unusually
chaotic (38-40% escape — seed-3 init luck), so the within-run margin over
chaos is small, but trained beats BC decisively (+11 to +23 pts) and the
cross-chaser gain over run17's trained policy is the controlled claim.

### run17 (`results/run17`, 300 blocks, 384² net) — previous reference

| policy | escape (0.7-only, stoch chaser) |
|---|---:|
| random | 20% |
| BC-pretrained | 21% |
| best checkpoint b299 | 31% |

### run18 (`results/run18`, 400 blocks, 512² net + gap-sprinkling) — negative result

| policy | escape (0.7-only) |
|---|---:|
| random | 27% |
| BC-pretrained | 13% |
| best checkpoint b340 | 26% |

Gap-sprinkling + capacity did not transfer ([E12](EXPERIMENTS.md)); its
own terrain drifted so far the scripted expert collapsed to 2% on its
tournament set. run17 superseded it; run19 supersedes run17.

## What the numbers mean (honest framing)

- **Within its own terrain** the trained evader is clearly better than
  untrained: run19's combined-set tournament ranks b210 first among NNs at
  68% (run17's champion was 48%); random-init is at 53%, BC at 33%. The
  self-play sawtooth (evader peaks mid-run, chaser re-adapts) makes
  best-checkpoint selection by protocol mandatory — never read the final
  block.
- **Cross-distribution** (uniform 4-tile gap gauntlets): run19's trained
  policy escapes 38-44% vs run17's 25-31% against either chaser — the
  patch/reward architecture fixed part of the gap-commitment ceiling.
- **Chaos is strong**: a random init escapes 20-40% of hard gauntlets
  depending on init seed and chaser. Every trained-policy claim is made
  relative to the same-draw baseline.

## Training dynamics

- run19: BC NLL 0.15; evader peak Elo 1352 / eval-wr 1.00 at block 61
  before the chaser adapted; CEM converged to spike-heavy mu; final-block
  evader Elo 1183 vs chaser 1216 (post-peak slump — selection is by
  protocol, not block).
- run17: evader peak Elo 1310 at block 20; CEM converged to spike_prob
  0.92.
- run18: gap-sprinkling drifted the terrain; BC'd policy topped its own
  combined set at 60%.

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
- `docs/media/tournament_run19/*.gif` — one recorded video per entrant on
  a common arena.
- `docs/media/progress_curve_v4.png` / `training_curves_v4.png` — run17's
  fixed-arena curve and Elo curves.

## Reproduce

```bash
python train.py --blocks 300 --out results/run19 --evader-net 384,384 \
    --evader-lr 4e-4 --seed 3 --device cuda     # ~50 min on CUDA
python hardarena.py --run results/run19 --matches 100 --chaser stoch --set 07
python hardarena.py --run results/run19 --chaser-run results/run17 \
    --matches 100 --chaser stoch --set 07        # cross-chaser control
python tournament.py --run results/run19         # ranked table + videos
python progress.py --run results/run19           # within-run curve
python gifdemo.py --run results/run19 --out docs/media/hero.gif --params 0.7
```

## Caveats

- ±8-10% CI at 100 matches; treat single-digit margins as directional.
- The oracle filter (scripted expert) is itself a skill assumption — it
  marks ~half of hard arenas winnable.
- Self-play non-stationarity: policies and numbers are run-specific; the
  ranking rules (protocol + seeded draws + selection) are the
  transferable contribution. Each run's exact config is frozen in its
  `params.json` (local) and archived at `docs/configs/run*_params.json`
  (tracked) — every metric stays one command away.
