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

## E6 — physics fidelity: WotW wall climb, fall gravity, terminal velocity
- **Motivation**: the WotW video review (IGN, first 17 minutes) showed the
  signature wall-climb (hold toward a wall → Ori climbs, no jump needed)
  and snappy descents — the old model only had wall-JUMP chains and uniform
  gravity.
- **Fix**: hold-into-wall while touching climbs at `climb_speed=170 px/s`;
  rising probe's toward-wall side is neutralized while climbing (else the
  climbed wall counts as a ceiling and climbing can never start);
  `fall_gravity_mult=1.12` (snappy descent), `max_fall=1600` (terminal
  velocity, also prevents tile tunnelling at speed).
- **Verified**: full jump apex 144 px (4.5 tiles — unchanged), wall climb
  rises at 170 px/s, terminal fall caps at 1600. Scripted expert now
  escapes 100% of mid-difficulty arenas even vs the strong scripted chaser
  (before: 4%!) — wall climb is a big power-up for both sides.

## E7 — hindsight reward shaping
- **Finding**: failures teach nothing about partial progress — "got 80% of
  the way then caught" and "got 5%" were equally punished, so the value
  function under-weights the final approach.
- **Fix**: track best-ever portal distance per episode; on any non-escape
  termination add `r_hindsight=1.5 × (1 − best/spawn)` — a hindsight-style
  credit for how close the run came, directly into PPO's terminal signal.

## E8 — chaser-strength curriculum (self-play ladder)
- **Finding**: the evader jumped straight from a random opponent to the
  LATEST (strongest) chaser every block; no gradual pressure ramp.
- **Fix**: `sample_ladder_opponent` anneals the latest-snapshot probability
  0.10 → 0.65 over 120 blocks, and when NOT picking the latest it weights
  OLD (weak) snapshots more early on — a natural difficulty ladder carved
  out of self-play history. The evader learns to beat weak chasers first,
  then progressively stronger ones.

## E9 — flee-aware BC demos
- **Finding**: the scripted evader now escapes 100% of arenas even vs the
  competent scripted chaser (wall climb + flee rules), so REAL-chaser
  demos contain no pressure — nothing to imitate.
- **Fix**: BC corpus is two-phase — traversal demos (ghost chaser) plus
  "pursued" demos where a fake chaser hovers 120 px behind, triggering the
  expert's flee rules (dash bursts, hops) so the NN starts self-play
  already knowing how to run with a pursuer on its heels.

## E10 — the chaser-strength LADDER backfired (run15)
- **Finding**: feeding the evader old/weak chaser snapshots early made the
  CEM see high win-rates and escalate difficulty faster than the learner
  could consolidate. Late training the evader was still losing to the
  strong chaser (eval-wr 0.19), the CEM reset to easy, and hard-arena
  skill stayed at +7 pts over random (vs +31 without the ladder in
  run14-era physics). The CEM's win-rate signal is only meaningful
  against the STRONGEST opponent.
- **Fix**: revert to latest-chaser 60% sampling + cap the CEM mu move at
  0.12/update so the curriculum cannot outrun the learner.

## E11 — asymmetric wall-climb (the balance fix)
- **Finding**: after physics v2, the chaser used wall-climb too and
  nullified the evader's escape routes; every run's evader plateaued at
  ~1150 Elo while the chaser hit 1250+. The game had no movement
  asymmetry.
- **Fix**: climb is evader-only (`can_climb` flag) — game-faithful (Ori's
  spirit kit vs a pursuer that can't climb). Combined with a 384² net +
  lr 4e-4, the evader hit its all-time peak (Elo 1310, eval-wr 0.93 at
  block 20) and the curriculum converged to near-max spike density
  (0.92). Hard-arena skill: trained 47% vs random 37% vs BC 35%.
- **Note**: the deterministic chaser is partly exploitable by chaos
  (random escapes 37% of hard arenas), so escape rates are only
  meaningful relative to the random/BC baselines.

## E12 — gap-sprinkling + capacity did NOT transfer (run18, negative result)
- **Hypothesis**: the CEM never raises gap_scale (wide gaps hurt the
  chaser too, so its win-rate signal saturates on spike loss), leaving
  the evader untrained on 4-5-tile gap gauntlets — the measured
  cross-distribution weakness. Force wide-gap training by sprinkling
  gap-forced arenas into self-play (33% at gap 0.6), add capacity
  (512² net), extend horizon (400 blocks).
- **Result**: best checkpoint ties run17 on the 0.7-only hard protocol
  (26% both) and edges it on mixed (21% vs 16%, within noise). The
  trained evader never beats the random baseline on the 0.7-only draw
  (26% vs 27% — vs run17's +9 over random at 17%).
- **Lesson**: distribution coverage is not the binding constraint; the
  local 13×9 observation patch + sparse escape reward limit
  gap-commitment learning, and sprinkling dilutes the spike-gauntlet
  skill that gives the within-terrain edge. The reproducible protocol
  (`hardarena.py`) also exposed that the earlier "hard-arena 47%" claim
  was measured on an easier, unreconstructable protocol — superseded.

## E13 — the stochastic-chaser protocol (honest metric)
- **Finding**: with a deterministic frozen chaser, a random evader
  escapes 17-37% of hard arenas by chaos alone. Sampling the chaser's
  actions (still frozen weights) closes the exploitation hole: random
  drops to 10-18% and the trained policy's ordering becomes
  trained > BC > random on the same arenas (run17 0.7-only: 26% vs 12%
  vs 17%; mixed: 16% vs 15% vs 10%).
- **Fix**: `hardarena.py --chaser stoch` is the definitive metric;
  deterministic-chaser numbers are reported only for reference.

## E14 — architectural lever: forward-biased patch + escape-dominant rewards (run19, POSITIVE)
- **Diagnosis**: the old centered 13×9 patch showed only 4 tiles up — a
  full jump apex is ~4.5 tiles, so the landing zone was invisible at
  apex — and only 6 tiles ahead at dash speed. And the escape bonus
  (+3.0) was SMALLER than the max milestone haul (15 zones × 0.4 = 6.0),
  teaching "run right" over "reach the portal".
- **Fix**: forward-biased 19×10 patch (5 back / 13 ahead / 7 up / 2
  down; obs 141 → 214) + escape-dominant rewards (portal +10.0,
  timeout −10.0, caught −8.0, portal-progress ×2).
- **Result (seeded draws, 100 matches, stochastic chaser)**:
  run19 trained 44% vs run17 trained 31% against run17's chaser, and
  38% vs 25% against run19's chaser — **+13 pts against both frozen
  chasers**. run19's own random init is unusually chaotic (38-40%), so
  the within-run margin over chaos is small, but trained beats BC by
  +11-23 and dominates the combined-set tournament (b210 68% vs run17's
  champion 48%).
- **Lesson**: observation geometry and reward dominance were the binding
  constraints, not distribution coverage (E12) or capacity. Also landed:
  per-run `params.json` persistence + seeded per-entrant draws +
  `--chaser-run` cross-chaser control — the metric is now fully
  reproducible and confound-controlled.
