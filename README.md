# ori-park — Ori-style movement physics + adversarial RL tag game

A from-scratch 2D physics engine that recreates the **movement model of Ori
and the Will of the Wisps**, wrapped in a vectorized gymnasium environment,
trained by a **self-play league of two adversarial neural-network policies**
(an agile evader vs a pursuer), with a **third adversarial learner** (a CEM
level generator) that keeps the curriculum at the edge of the agents' skill.

> **Untrained vs trained evader, same arena, same frozen chaser** (the
> trained clip escapes through the portal, the untrained one impales itself):
>
> ![improvement](docs/media/evader_improvement.gif)
>
> ### Measurement status — read [docs/VERIFICATION.md](docs/VERIFICATION.md)
>
> A full verification pass over this repo re-ran every claim and found a
> movement bug: `wall_dir` was never cleared and wall
> climb/jump/slide never tested for actual wall contact, so agents could climb
> at 170 px/s **in mid-air** anywhere on the map. The earlier headline results
> are retracted. Measured on the corrected protocol — fixed physics, 310
> expert-winnable hard arenas, stochastic frozen chaser, 100 seeded matches,
> identical draws for every entrant:
>
> | entrant | escape | survival (not caught / not impaled) | mean zone of 12 |
> |---|---:|---:|---:|
> | scripted expert (privileged state) | **56%** | 66% | 12.7 |
> | run17 trained (old arch) | 38% | 59% | 10.6 |
> | run19 trained (new arch) | 33% | 57% | 10.0 |
> | run20 trained (new arch) | 30% | 62% | 11.0 |
> | random init | 37% | 59% | 10.2 |
> | BC-pretrained | 1–4% | 58% | 5.9 |
>
> So the pre-fix training runs **do not beat a random init** once free flight
> is removed, and roughly a third of their episodes end on spikes. `run21_fixed`
> is the first legitimate training run (same recipe as run19, fixed physics).
> Reproduce any row with:
> `python hardarena.py --run results/run17 --chaser-run results/run17 --matches 100 --chaser stoch --set 07 --expert`
>
> The renderer is painterly Ori-style: blurred god-light, a vine-hung canopy,
> parallax hills, moss-capped rock with drop shadows, glowing orbs, a gate
> portal with a light column, luminous spirit agents with comet trails,
> drifting fireflies, and a HUD with portal progress and an end-of-clip
> outcome banner.

```
              ┌─────────────────────────────────────────────────┐
              │  OriArenaVecEnv (vectorized numpy, 120 Hz phys) │
              │                                                 │
   evader ──► │  run · jump · double-jump · wall-jump · dash     │
   chaser ──► │  · bash-orbs · coyote time · input buffering    │
              └───────┬──────────────────────────┬──────────────┘
                      │                          │
        ┌─────────────▼──────────┐   ┌───────────▼─────────────┐
        │ PPO (evader policy)    │   │ PPO (chaser policy)     │
        │ trained vs chaser pool │   │ trained vs evader pool  │
        └────────────────────────┘   └─────────────────────────┘
                      │  self-play league: alternating blocks,
                      │  opponent snapshots, Elo ratings
                      ▼
        TerrainAdversary (CEM) ──► arena params ──► ArenaGenerator
        keeps evader win-rate ≈ 50%  (reachability-gated levels)
```

## What it recreates from Ori WotW

The game's signature feel comes from its movement *parameters*, which are
implemented faithfully (tuned in `oripark/config.py` → `MoveParams`):

| Ori WotW mechanic | Implementation |
|---|---|
| Snappy ground control / instant turns | 3600 px/s² accel, 4400 decel |
| Variable-height jump (hold = high, tap = low) | release cuts `vy` by 50% (`jump_cut`) |
| Jump buffering (press early, jump on land) | 0.15 s input buffer |
| Coyote time (jump just after leaving a ledge) | 0.10 s grace |
| Double jump (once per airtime) | `can_djump`, refreshed on landing; ground jump does **not** consume it |
| Wall slide | capped fall speed 170 px/s against walls |
| Wall jump (chainable) | 640 px/s horizontal kick + 900 px/s pop, 0.14 s re-stick grace |
| **Wall climb (hold toward a wall)** | 170 px/s — the hero's signature move, **evader-only** (the pursuer can't) |
| Dash (8 directions, momentum retention) | 1250 px/s burst for 0.12 s, post-dash bleed-off |
| Bash (launch off targets, refreshes dash+djump) | static orbs, 8-way aim, 1150 px/s launch, 0.22 s cooldown |
| Slight "float" while rising, snappy descent | gravity × 0.92 ascending, × 1.12 falling, terminal velocity 1600 px/s |

Arenas are tile-based (60×40, 32 px tiles) with a **path-first generator**:
it lays a canonical platform chain from spawn to the light portal, then
decorates with climb towers, spike pits, dash gaps and bash orbs. A coarse
**reachability BFS** (jump/double-jump/dash/wall-climb move model) gates
every level, so the adversarial generator can never propose an impossible
map.

## The three adversarial learners

1. **Evader (PPO)** — full Ori kit. Rewards: portal progress + rightward
   milestones (dense, un-gameable), a one-time bonus for passing the chaser,
   survival time, separation gain, agility usage (dash/wall-jump/
   double-jump/bash), **+10 for escaping through the portal** — dominant
   over the dense shaping so the objective is unambiguous — penalties for
   being caught (−8) or hitting spikes (−8), **−10 for failing to escape
   before timeout**. (The old +3/−3 scheme was smaller than the max
   milestone haul, teaching "run right" over "reach the portal" — fixed
   in run19, [E14](docs/EXPERIMENTS.md).)
2. **Chaser (PPO)** — same physics minus bash. Rewards mirror the evader's:
   closing distance, proximity, catching, penalty when the evader escapes.
3. **Terrain adversary (cross-entropy method)** — a small policy over the 6
   arena parameters (gap scale, tower height, spike probability, orb count,
   wander, dash-gap frequency). It is updated to keep the evader's win rate
   near 50%, so levels stay just past the agents' current skill — the
   classic asymmetric-self-play curriculum. It starts from easy levels and
   hardens only as the evader learns to win.

Self-play loop (`oripark/selfplay.py`): a **warmup phase** (both sides train
against random opponents so traversal/escape skills develop first), then
**alternating training blocks**. Each block trains one PPO against a snapshot
sampled from the opponent's league pool (60% latest, 40% older checkpoints —
prevents degenerate cycling). Latest-vs-latest evaluations update two-player
Elo ratings. Agility metrics (dashes, wall jumps, double jumps, bashes,
airtime, max traversal zone per episode) are logged per block and plotted.

## Run it

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# quick smoke test (~1 min)
./.venv/bin/python train.py --quick

# real self-play run (defaults: 120 blocks, 16 parallel arenas)
./.venv/bin/python train.py --blocks 120 --out results/run1

# render demos: random baseline vs trained, same arena seed
./.venv/bin/python demo.py --run results/run1 --mode both --arena-seed 4242 --out demo.gif

# head-to-head eval report vs the untrained baseline (markdown)
./.venv/bin/python eval.py --run results/run1 --matches 40 --baseline

# tournament: rank every checkpoint vs the frozen chaser + record videos
./.venv/bin/python tournament.py --run results/run1 --matches 60

# THE definitive metric: escape rate on oracle-filtered hard arenas
# (stochastic frozen chaser, seeded draws; --random-seeds K measures the
# true random floor as a population instead of one lucky init)
./.venv/bin/python hardarena.py --run results/run1 --matches 100 --chaser stoch --set 07
./.venv/bin/python hardarena.py --run results/run1 --random-seeds 6

# best-checkpoint selection (self-play is non-stationary)
./.venv/bin/python select.py --run results/run1
```

Outputs in `results/<run>/`: `blocks.jsonl` (per-block Elo/win-rate/agility),
`episodes.jsonl` (every episode), `curves.png` (learning curves),
`evader.zip`/`chaser.zip` (final policies), `pool_ev_*.pt`/`pool_ch_*.pt`
(league snapshots — `pool_ev_0.pt` is the true random-init baseline;
`evader_init.pt` is the BC-pretrained policy), `adv_mu.npy` (final
terrain-adversary parameters).

## Structure

```
train.py / demo.py / eval.py      CLI entry points
oripark/
  config.py       MoveParams / EnvParams / TrainParams
  physics.py      vectorized Ori-style movement + tile collision
  arena.py        path-first procedural levels + reachability BFS
  env.py          OriArenaVecEnv (stable-baselines3 VecEnv, no subprocesses)
  adversaries.py  TerrainAdversary (CEM) + StaticSampler
  selfplay.py     self-play league: alternating PPO blocks, pools, Elo
  metrics.py      JSONL logging + matplotlib curves
  render.py       headless pygame replay -> GIF / ASCII replay
```

## Notes & extensions

- The physics is Ori-**inspired**, not a copy of the game's binaries/assets
  (no extraction — everything is re-implemented and tunable).
- Next steps that fit this codebase: bash-target projectiles, wall-grapple,
  multi-agent evader teams, PPO with recurrent policies, or a pixel-based
  CNN observation if you want the agents to see rather than read state.
