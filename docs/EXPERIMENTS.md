# Experiments log

Running log of improvement experiments (hypothesis → change → measured
result). The goal: a professional-grade evader that reliably reaches the
portal.

## E1 (run2→run9 era) — reward engineering
- Portal/timeout rebalance, portal-progress + milestone + pass bonuses,
  warmup, asymmetric capacity, CEM easy-start/reset.
- Result: 0% → 30% final escape rate (progress curve), but capped —
  see E2/E3 for the ceiling.

## E2 — physics bug: grounded running was impossible
- **Finding**: `_integrate_x` probed the agent's bottom-front corner for
  wall collision. A grounded agent's feet rest *exactly* on the ground tile
  boundary, so the bottom probe always hit the floor → every grounded move
  was zeroed (`vx=0, wall_dir=1`). Agents only moved while airborne
  (jumping/dashing constantly) — crippling traversal.
- **Fix**: the feet probe only counts as a wall if the tile directly below
  the agent's center is NOT solid (i.e. it is a ledge, not the floor).
- **Result**: grounded running works (26 tiles in 2 s at max_run); jump
  peak 4.5 tiles; dash/wall-jump chains verified. This is a *physics
  change*, so pre-fix training runs are not directly comparable — run11+
  are the reference.

## E3 — scripted expert + behavior cloning
- Built `oripark/scripted.py`: a BFS waypoint-following controller that
  uses the full Ori kit (run, jump, double jump, wall-jump chains, dash,
  bash) with privileged grid access.
- **Iterative fixes found by profiling stalls**:
  - jump/dash/bash are consumed on *rising edges* — the controller must
    PULSE, not hold (holding fires exactly one action).
  - overshoot steering: never steer left except when stuck on the ground.
  - dashing into wall pockets: dash only toward a FAR waypoint (ddx>200)
    with a 3-height clear corridor.
  - `wall_grace` chaining: pulse every 7 steps while hugging a wall.
  - the double-jump cooldown was 1<<20 steps — after one djump, all jumps
    were blocked for a million steps (landing didn't reset it). Capped at
    60 and reset on landing.
- **Result**: scripted traversal escape rate ~53-66% depending on arena
  distribution (vs 0% for random, 30% for the pre-fix RL policy; 77% vs
  the frozen chaser on one validated arena set).
- **BC**: pretrain the evader policy on 400 escaping rollouts (89k pairs,
  NLL 1.40 → 0.27), then self-play refines vs the chaser. run11 is the
  first full BC + fixed-physics training.

## E4 (in progress) — full BC + fixed-physics self-play
- run11: 150 blocks, BC-pretrained evader, physics fixed. Watch:
  - Does the evader keep its traversal skill while learning evasion?
  - Escape rate on the fixed eval arena set (target ≥ 80%).
  - Whether the chaser's grounded running makes it much harder (it can
    now run at 430 px/s — the chase is genuinely fast).

## E4 — full BC + fixed-physics self-play (run11/run12/run13)
- run11: BC (400 demos) + self-play, physics fixed. Progress curve:
  BC baseline 62% escape → trained 42-58% (chaser adaptation pulls it down).
- run12: added **BC regularization** (fine-tune the evader on demos every
  5 blocks) → BC 54% → trained 67% final (peak 71%). All non-escapes are
  catches by the strong adaptive chaser.
- run13 (250 blocks): 2434 training escapes (vs 235 in run10), plateaus at
  58-71%; best checkpoint block 240 = 71% on the fixed arena set.

## E5 — CEM curriculum collapse bug (biggest single find of this round)
- **Finding**: `_cand_eval` evaluated the CEM's candidate arenas with the
  DETERMINISTIC evader — which gets 0% escape (argmax stuck in repetitive
  patterns). The adversary therefore always saw win-rate ≈ 0, fired its
  "unwinnable → drift easy" reset forever, and the curriculum **collapsed
  to easy arenas** (mu 0.1-0.3). The evader never trained on hard terrain
  — on hard eval arenas (params 0.7-0.85) it scored no better than a
  random policy (28% both).
- Also found: on EASY arenas escape rate is luck-dominated — a RANDOM
  policy escapes 65% vs the deterministic chaser (chaotic dash-spam is
  unpredictable), so easy-arena escape rates do not measure skill.
- **Fix**: `_cand_eval` uses stochastic play (matching the official eval
  protocol); `adv_matches` 2 → 4 for stable win-rate estimates; CEM sigma
  capped at 0.35 with decay instead of the +0.02 random walk that inflated
  it to 0.8.
- **Result (run14)**: adv_wr rose from 0.17 → 0.67 across blocks — the
  curriculum now actually escalates; evader Elo reached 1188 vs chaser
  1211 (closest of any run). Skill should now be measurable on hard arenas.

## Failure-mode autopsy (run10 telemetry, pre-physics-fix)
- 70% of training episodes ended "caught" — the chaser was the dominant
  killer, not the terrain.
- 595 evader deaths in zone 14 (portal zone) — reached the final approach
  but couldn't finish.
- Training escapes ramped 4 → 26 → 92 per 30 blocks (blocks 0-30, 60-90,
  120-150): the skill was there, the finish was the problem.
