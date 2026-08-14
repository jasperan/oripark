# Tuning guide

Everything that matters for training is a dataclass in `oripark/config.py`:
`MoveParams` (physics feel), `EnvParams` (rewards / episode rules),
`TrainParams` (self-play loop). This doc explains the design decisions and
what to change when you want different behavior.

## Reward design (the hard-won lessons)

The evader's reward started as "survive + gain distance + agility" and
initially *failed to produce escapes*. Three concrete problems were found
and fixed — the same traps apply to most tag/escape games:

1. **Survival rewards make timeouts attractive.**
   With `r_time = 0.02/step`, sitting still for the 25 s episode cap was
   worth `0.02 × 1500 = 30` points — far more than the portal bonus. The
   agent "learned" to stall. Fix: `r_time = 0.002` (timeout ≈ 0 total),
   `r_portal = +3.0`, `r_timeout = -3.0`. The only way to score is to
   actually escape.

2. **"Flee the chaser" and "reach the portal" conflicted.**
   The chaser spawns *between* the evader and the portal, so moving toward
   the goal also moves toward the chaser. With `r_dist_gain = 0.015/50px`
   and `r_portal_progress = 0.02/100px`, the net reward for advancing was
   negative and the evader drifted backward. Fix: make portal progress the
   primary signal (`r_portal_progress = 0.05/100px`) and separation gain
   secondary (`r_dist_gain = 0.01/50px`).

3. **Timeouts polluted the metrics.**
   Counting "timeout" as an evader win made an untrained, immobile policy
   look 100% successful. Eval (`eval.py --baseline`) now reports *escape
   rate* (portal reached) and *catch rate* separately; timeouts are draws.

| reward | value | purpose |
|---|---|---|
| `r_time` | 0.002 | negligible survival tick |
| `r_portal_progress` | 0.05 / 100 px | primary: navigate toward the light portal |
| `r_dist_gain` | 0.01 / 50 px | secondary: keep separation from the chaser |
| `r_proximity` | 0.004 | mild aversion to being close |
| `r_agility` | 0.06 | encourage dash / wall-jump / double-jump / bash usage |
| `r_portal` | +3.0 | clean escape (dominant terminal) |
| `r_caught` | -2.5 | tagged |
| `r_hazard` | -2.0 | spike / void death |
| `r_timeout` | -3.0 | failing to escape is a loss for the evader |

Chaser rewards mirror the evader's (closing distance, proximity, catch,
+3.0 for catching, -3.0 if the evader escapes).

## Self-play loop

- **Alternating blocks** — each block trains one side for
  `block_steps` timesteps against a snapshot from the other side's pool.
- **League pools** — snapshots are kept (`pool_size = 8`); the opponent is
  sampled with `opp_latest_prob = 0.6` from the latest, otherwise from an
  older checkpoint. Old checkpoints prevent degenerate cycling
  (A beats B, B beats C, C beats A forever).
- **Elo** — latest-vs-latest matches update two-player Elo (draws = 0.5).
- **Terrain adversary (CEM)** — every `adv_update_every` blocks, evaluates
  `adv_pop` candidate arena-parameter vectors against the current policies
  and fits a Gaussian over the elites so the evader's win rate drifts
  toward `adv_target_wr = 0.50`. This keeps the curriculum at the edge of
  skill (asymmetric self-play, like OpenAI's hide-and-seek level curator).

## Physics tuning (the Ori feel)

`MoveParams` holds the movement constants. Key knobs:

| param | default | effect |
|---|---|---|
| `gravity` / `jump_v` | 2600 / -820 | jump height ≈ 4 tiles (`v²/2g`) |
| `jump_cut` | 0.50 | release early → ~half velocity (variable height) |
| `coyote_time` | 0.10 s | ledge grace |
| `jump_buffer` | 0.15 s | input buffering |
| `air_accel` | 2900 | strong air control (~80% of ground) |
| `dash_speed` / `dash_time` | 1250 / 0.12 s | burst ≈ 39 px/frame |
| `walljump_h` / `walljump_v` | 640 / -900 | horizontal kick + vertical pop |
| `bash_speed` | 1150 | orb launch velocity |

Lower `gravity` / raise `air_accel` for a floatier feel; raise `dash_time`
for more powerful dashes. `substeps` (2 = 120 Hz physics) controls
collision robustness vs. speed.

## Arena generation

`ArenaGenerator` parameters (the terrain adversary's action space):
`[gap_scale, climb_scale, spike_prob, orb_count, wander, dash_gap_prob]`.
Levels are generated path-first (platform chain spawn → portal), then
decorated. A coarse reachability BFS (single jump, double jump, dash,
wall-climb chains) gates every level — the adversary can never propose an
impossible map. `elevated >= 2` keeps levels interesting (no boring
ground-only arenas).

## Typical failure modes

| symptom | likely cause | fix |
|---|---|---|
| episodes end in 1–5 steps | spawn/agent-index mismatch in the env | check `_load_arena` indices (contiguous evader/chaser slices) |
| evader never reaches portal | survival reward > portal reward, or flee-vs-progress conflict | see reward design above |
| chaser dominates forever | curriculum too hard / too few blocks | raise `blocks`, soften adversary (`adv_target_wr`), lower `opp_latest_prob` |
| Elo flatlines | LR annealed to 0 (SB3 linear schedule across blocks) | constant LR (`lambda _: tp.lr`) — already the default |
| level generation returns same trivial arena | all `_try` attempts failed the gates | check BFS move caps vs. generator deltas |

## Reproducibility

Everything is seeded (`TrainParams.seed`); deterministic eval uses
`fixed_seed_sample` so Elo compares like-for-like arenas across blocks.
