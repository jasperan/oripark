# Results

## The skill metric (reproducible)

`python hardarena.py --run results/runN --matches 100 --chaser stoch --set 07`

- **Arenas**: uniform-hard terrain params 0.7 (or a 2:1 mix of 0.7/0.85),
  fixed seeds, **filtered by the scripted-expert oracle** — an arena only
  counts if a competent player can actually win it, so the test measures
  skill, not luck of reachability.
- **Chaser**: frozen run chaser, **stochastic** (action sampling). A
  deterministic frozen chaser is partly exploitable by chaos — a random
  policy escapes 17-37% of hard arenas just by being unpredictable. The
  stochastic chaser closes that hole; this is the honest number.
- **Evader**: stochastic, the honest PPO protocol.

### run17 (`results/run17`, 300 blocks, 384² net)

| policy | escape (0.7-only, stoch chaser) | escape (mixed 0.7/0.85) |
|---|---:|---:|
| random | 17% | 10% |
| BC-pretrained | 12% | 15% |
| **best checkpoint b299 → `evader.zip`** | **26%** | 16% |

The trained evader escapes hard gauntlets **+9 pts over chaotic luck** and
+14 over its own teacher. Best checkpoint was selected by this protocol
(b299); b130 is the within-terrain champion (see tournament).

### run18 (`results/run18`, 400 blocks, 512² net + gap-sprinkling)

| policy | escape (0.7-only) | escape (mixed) |
|---|---:|---:|
| random | 27% | 18% |
| BC-pretrained | 13% | 14% |
| best checkpoint b340 → `evader.zip` | 26% | 21% |

**Verdict: no cross-distribution gain.** run18's levers (gap-sprinkling at
gap 0.6 for 1/3 of training arenas, 512² capacity, 400 blocks) tie run17 on
the 0.7-only set (26% both) and edge it on mixed (21% vs 16%, within
noise). The gap-sprinkling hypothesis — that forcing wide-gap training
would teach cross-distribution traversal — is a documented negative
([E12](EXPERIMENTS.md)). run17 remains the reference run.

## What the numbers mean (honest framing)

- **Within its own terrain** the trained evader is clearly better than
  untrained: tournament combined set (mu spike gauntlets + hard uniforms)
  ranks the trained checkpoints above random 37% — b130 48%, b299 37% —
  and the run's own-distribution eval shows trained 28% vs untrained 15%
  (run17). The self-play sawtooth means the FINAL block is usually in a
  post-peak slump; best-checkpoint selection is mandatory.
- **Cross-distribution** (uniform 4-tile gap gauntlets the curriculum never
  produced — the CEM keeps gap easy because wide gaps hurt the chaser too):
  the edge over random is real but modest (+9 pts run17). Earlier
  "hard-arena 47%" numbers in this repo were measured on an
  unreconstructable, easier protocol and are **superseded** by this
  reproducible one.
- **Chaos is strong**: a random policy escapes 17-27% of hard gauntlets.
  Every trained-policy claim is made relative to that baseline.

## Training dynamics

- run17: BC NLL 0.17 (384²); evader all-time peak Elo 1310 / eval-wr 0.93
  at block 20 before the chaser adapted; CEM converged to spike_prob 0.92.
- run18: evader dominated early (eval-wr 0.85 at b61) then the chaser
  adapted (sawtooth); final-block evader at Elo 1076 vs chaser 1319 — the
  classic post-peak slump, exactly why selection is by protocol, not block.

## Physics v2 — Ori WotW fidelity (movement)

| change | value | verified |
|---|---|---|
| Wall climb (hold toward a wall) | 170 px/s | climbs a 26-tile tower; **evader-only** (the pursuer can't — Ori lore) |
| Snappy descent | gravity × 1.12 falling | full-jump apex unchanged at 144 px (4.5 tiles) |
| Terminal velocity | 1600 px/s | no tile tunnelling at speed |
| Learned behavior shift | — | policies traded wall-jump chains (wj ≈ 4-5/block) for climbing (wj ≈ 1-2/block) |

## Visual proof

- `docs/media/evader_improvement*.gif` — trained (b299) escapes in 123-202
  steps on hard arenas where the untrained policy is caught or times out.
- `docs/media/tournament/*.gif` — one recorded video per entrant on a
  common arena: random flails, BC hits a hazard, the champion escapes.
- `docs/media/progress_curve_v4.png` / `training_curves_v4.png` — run17's
  fixed-arena curve and Elo curves.

## Reproduce

```bash
python train.py --blocks 300 --out results/run17 --evader-net 384,384 \
    --evader-lr 4e-4                       # ~60-80 min (cpu) or faster (cuda)
python hardarena.py --run results/run17 --matches 100 --chaser stoch --set 07
python tournament.py --run results/run17   # ranked table + videos
python progress.py --run results/run17     # within-run curve
python gifdemo.py --run results/run17 --out docs/media/hero.gif --params 0.7
```

## Caveats

- ±7-8% CI at 100 matches; treat single-digit margins as directional.
- The oracle filter (scripted expert) is itself a skill assumption — it
  marks ~half of hard arenas winnable.
- Self-play non-stationarity: policies and numbers are run-specific; the
  ranking rules (protocol + selection) are the transferable contribution.
