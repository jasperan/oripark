# Experiments log

> **E14, E15 and E16 are retracted** — see [E17](#e17--verification-pass-fixes-physics-oracle-and-the-metric-retracts-e14e15e16)
> and [VERIFICATION.md](VERIFICATION.md). They were measured under a movement
> bug that let agents climb in mid-air and with an arena oracle that tested a
> panicking expert, and every headline number they report reproduces only under
> that bug. E17 records the fixes and the first legitimate training run.

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
- **Superseded (E16)**: the 10-18% "random" figures here used ONE init
  draw per run. The K=6 random-population floor shows the true random
  floor is 4-8% — see [E16](#e16--the-random-floor-is-4-8-not-17-45-chaotic-init-baseline-measured).

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

## E15 — second arch run (run20, seed 7): the gain generalizes (POSITIVE)
- Same arch/rewards as run19 (forward-biased 19×10 patch, escape-dominant
  rewards, 384² net, 300 blocks), new seed. Evader trained even higher
  (peak Elo 1370 vs run19's 1352) but its chaser never re-adapted (final
  Elo 1180 vs run19 chaser's 1216) — a weak chaser inflates own-chaser
  escape numbers, which is why run20's column in the 3×3 matrix runs hot
  for everyone.
- **Result**: run20 trained beats run17 trained on all three frozen
  chasers (+4/+7/+20), confirming the arch gain generalizes. Between the
  two arch runs, run19's trained remains the strongest evader (row mean
  48% vs run20's 40% — seed variance matters).

## E16 — the random floor is ~4-8%, not 17-45% (chaotic-init baseline, measured)
- Each run's saved `pool_ev_0` (labeled "random" in every earlier table)
  is ONE init draw, and it turns out the runs got lucky: run17's init
  escapes 20%, run19's 40%, run20's 45% of hard arenas. Fresh inits from
  other seeds escape 0-2%.
- **Fix**: `--random-seeds K` evaluates K deterministic inits (member 0 =
  the run's own init) with identical draws and reports mean/min/max.
  K=6 floors: run17 4.3%, run19 6.9%, run20 7.7%. The trained edge over
  the TRUE floor is +27 to +49 pts (was reported as +6-11 vs a lucky
  init). Chaos is real but rare — most random inits flail and die.
- **Retraction**: the "chaos is strong — random escapes 17-27%" framing
  in earlier docs (E13, Round-4 tuning) measured a single lucky draw per
  run, not the floor. Single-init baselines are retired from the metric.


## E17 — verification pass: fixes physics, oracle and the metric; retracts E14/E15/E16

**Hypothesis**: the published improvement runs (E14/E15) measure movement
skill. **Method**: re-run every claim from a clean checkout — pytest-style
probes, an A/B harness over two physics builds with identical seeded draws,
and a re-evaluation of the saved policies.

**Three defects, all fixed and regression-tested:**

1. **Free flight (physics).** `wall_dir` was written on a leading-edge
   collision and cleared only by a probe requiring `|vx| < 1e-3`, and wall
   climb / wall jump / wall slide never tested for contact — so an airborne
   agent holding that direction rose at `climb_speed` (170 px/s) anywhere.
   A probe climbed with the nearest solid tile 99 tiles away; a trained run19
   evader did it for 9,856 steps across 40 environments. Fix: `_wall_contact()`
   re-derives the wall side from head/centre/feet probes every substep.
   *A/B, one shared arena set, 100 seeded matches, identical draws:*
   run17 31%→26%, **run19 44%→20%**, run19-vs-own 38%→26%, **run20 57%→35%**.

2. **Panic-mode oracle (arena filter).** `expert_winnable()` fed the scripted
   expert its own position as `chaser_x/chaser_y`, pinning `flee=True` for the
   whole episode, while a random chaser wandered the test arena. The documented
   "an arena only counts if a competent player can win it" admitted 90/600
   hard arenas, not ~half. Fix: `--oracle calm` parks the chaser and feeds the
   real position → 310/600.

3. **Metric denominators.** The metric tool (`hardarena.py`) had a fourth copy
   of the episode loop and computed `tm = n - escaped - caught`, so both hazard
   outcomes were reported as "timeout", and it printed `esc/args.matches`
   beside a rate computed over `esc/n_done`. Fix: every entrant now goes
   through `oripark.evals.play_episodes`, artifacts carry five outcome columns
   that sum to `n`, plus `survival` and `mean zone`.

**Measured result (corrected protocol — fixed physics, 310 calm-oracle arenas,
stochastic frozen chaser, 100 seeded matches):**

| entrant | escape | survival | ev_hazard | mean zone |
|---|---:|---:|---:|---:|
| scripted expert (privileged) | 56% | 66% | 0 | 12.7 |
| run17 trained | 38% | 59% | 33 | 10.6 |
| run19 trained | 33% | 57% | 27 | 10.0 |
| run20 trained | 30% | 62% | 18 | 11.0 |
| random init | 37% | 59% | 34 | 10.2 |
| BC-pretrained | 1-4% | 58% | 35 | 5.9 |

**Retractions.** E14's "+13 pt architectural gain" and E15's "the gain
generalizes" do not survive: run19 depended on the exploit ~2.4x more than
run17 and lands *below* it. E16's "random floor is 4-8%, trained beats it by
+27 to +49" is a property of the old oracle: on the corrected arena set a
random init escapes ~37%, i.e. the pre-fix trained policies are at
random-init level. Also retracted as unsupported: the BC NLL 0.15 figure
(nothing persists it), the run19 peak Elo (actual 1364.2 @ b63, not 1352 @
b61), the run20 matrix cell (artifact says 57%, docs said 54%), and
`selected.txt`'s protocol lines (hand-written; `select.py` cannot emit them).

**New bottleneck.** The frozen chasers now catch only 7-27% of episodes —
they were trained against evaders that could fly. Escape rates are therefore
partly "can you avoid the spikes and drift to the portal", and the honest
opponent is a chaser retrained under fixed physics. `run21_fixed` is that run
(run19's exact recipe, 300 blocks, seed 3, fixed physics); its results are
appended below when it completes.

## E18 — the "random" baseline was a trained snapshot (bug, FIXED)
- `pool_ev_0.pt` was loaded as "random"/"untrained" by hardarena, eval,
  progress, tournament and the hero GIF's UNTRAINED panel. It is neither: the
  league trims its pool front (`ev_pool.pop(0)`) once it exceeds `pool_size`,
  and the final save writes `pool_ev_{i}` from the trimmed list, so
  `pool_ev_0` is the oldest *surviving trained* snapshot.
- Evidence: weight L2 norm 77-92 vs 46 for a fresh init (finished policy:
  73-92); escape rate 18-42% on hard arenas where five fresh inits escape
  **0.0%** (500 episodes).
- This also explains E13/E16's "each run's init was a chaos-lucky draw" — the
  files it measured were early trained snapshots. The "true random floor"
  table was built with a trained policy as member 0.
- **Fix**: `selfplay.py` writes `pool_ev_boot.pt`/`pool_ch_boot.pt` at startup;
  `oripark/baselines.py` resolves the pre-training checkpoint and labels
  `pool_ev_0` honestly; `hardarena --random-seeds K` builds K fresh inits; a
  test pins the norm relationship so the mislabel cannot silently return.

## E19 — DAgger distillation of the scripted expert (POSITIVE)
- Hypothesis: plain BC fails because the teacher is *privileged* (BFS planner
  + exact state), so the student must be trained on states it actually visits.
  Measured BC result before this: a 56% teacher distils into a 1-4% student
  (8-19% in its own obs convention) that never leaves the first third of the
  arena (zone 5.8-6.9 of 12).
- Change: `oripark/dagger.py` — roll the student out, label every visited
  state with the expert's action, train on those labels, with a decaying
  expert mixing rate beta (1.0 -> 0.0 over 4-6 rounds). Also fixed the demo
  collector pairing each action with the *next* observation.
- **Result**: distillation alone (no RL, ~20 minutes) reaches **29%** escape
  against the competence-restored chaser, versus 7-19% for BC and 29-30% for
  the pre-fix PPO runs. Pure-student rollouts at beta=0 escape 69% on the
  traversal arenas. The pursued variant (teacher sees a real chaser) is much
  stronger than the ghost variant (29% vs 20%) — evasion must be demonstrated.
- `train.py --init-evader <ckpt>` warm-starts PPO from it (run22).

## E20 — first legitimate training run (run21_fixed, POSITIVE)
- run19's exact recipe (300 blocks, 384² net, lr 4e-4, seed 3) re-run under
  the fixed physics, so both sides learn a game where flying is impossible.
- **Result**: 40% escape against its own chaser and 41% against run17's —
  chaser-robust, versus the pre-fix runs at 29-30% and a fresh random init at
  0%. The scripted expert swings 27.7-54.5% depending on the opponent, so
  against the legacy chaser the learned policy is *better* than the expert.
- Its chaser is the first competent pursuer in the repo (catches the expert in
  50% of episodes), but it still dies on spikes in 19-43% of episodes — the
  chaser-side failure modes are the next lever.

## E21 — DAgger scaling, and the chaser side (mixed)
- **Scaling the evader distillation works.** Same method, more data:
  ghost teacher 20% -> pursued-vs-snapshot teacher 24% -> pursued-vs-final-chaser
  teacher, 8 rounds x 96 eps, 25.5k pairs: **30.7%** escape (100 matches vs
  run21's chaser), i.e. the distillation alone now beats every pre-fix PPO run
  (29-30%) and sits at 56% of the scripted expert's 54.5%. Pure-student
  rollouts (beta=0, no expert help) escape 37.5% against the same chaser.
- **The chaser-side distillation fixes suicide but is too weak to use.**
  Learned chasers die on spikes in 19-43% of episodes, handing the evader wins
  unrelated to evasion, so the scripted pursuer (never suicides) was distilled
  into a learned chaser: ch_hazard fell to ~2%, but the student catches only
  15-35% of episodes where the scripted teacher catches 60-87% and run21's
  trained chaser catches ~30-45%. Result: against the distilled chaser every
  entrant's escape rate inflates (run21 57%, its early snapshot 68%, the expert
  64.4%), so it is a bad training opponent. Kept as a "no-suicide" reference
  opponent rather than as a warm start.
- Consequence for the ladder: run23 uses the stronger *evader* init
  (E21's 30.7% distillation), not the chaser warm start.

## E22 — a fixed, non-suicidal opponent for the metric (POSITIVE, tooling)
- The learned chasers drift (retrained as recipes change) and hand out
  `ch_hazard` wins, so `hardarena --scripted-chaser` was added: the privileged
  pursuer as a frozen opponent. It never suicides (ch_hazard ~0-1%) and never
  changes, so it is a stable yardstick for comparing evaders across runs.
- First measurements (100 matches, 310 calm arenas): run17 trained 37%,
  run21 trained 31%, run21 early snapshot 36%, fresh init 0%, BC init 1-11%
  (with 82% ev_hazard), scripted expert **11%** — the expert is caught in 87%
  of episodes by its own scripted pursuer, while learned evaders survive it
  three times as often. Being unpredictable is worth more than planning
  against this particular pursuer; the finding is worth remembering when
  reading "expert" as a ceiling.

## E23 — the reported checkpoint was the last block, not the best (DEFECT)
Sweeping every saved checkpoint on one arena block exposed the biggest
reporting defect in the repo: `evader.zip` is the LAST-block save, but every
tool labelled it `trained (best)`, and `select.py` — which exists to pick the
best block by escape rate on a fixed arena set — had never been run for the
recent runs (`selected.txt` was hand-authored, which the audit caught).

| run | best block (selection block seed0=3000) | last block b299 |
|---|---:|---:|
| run22 | b220 62% | 26% |
| run21 | b200 59% | 37% |
| run23 | b260 31% (b250 43% on a 60-match sweep) | 18% |

Those 59-62% do NOT survive re-measurement on a disjoint arena block, which is
the second half of the lesson: with ~31 checkpoints to choose from and 24-48
matches each, the maximum is a winner's-curse estimate (n=48 -> SE ~7%). On the
disjoint reporting block (seed0=9000, 100 matches, vs run21's chaser):

| policy | escape | ev_hazard |
|---|---:|---:|
| run21 trained (final b299) | **37%** | 19 |
| run23 DAgger init (distillation only) | 36% | 19 |
| scripted expert | 33% | 4 |
| run22 trained (selected b220) | 33% | 15 |
| dagger_mix (4 rotating opponents, ~190k pairs) | 31% | 17 |
| run22 init / pool b0 | 28% / 27% | 18 / 20 |
| run22 trained (final b299) | 26% | 18 |
| run23 trained (selected b260 / final b299) | 25% / 22% | 12 / 23 |

**The arena block alone moves the expert from 54.5% (seed0=5000) to 33%
(seed0=9000), so only same-block comparisons are meaningful** — everything
above is one block, one opponent, 100 matches (SE ~4.7%).

Protocol fixes: `hardarena --arena-seed0` (report on a different arena block
than the one used for selection), `select.py --chaser-run` (a fixed opponent
makes runs comparable) and `--arena-seed0`, `evader_final.zip` (the last-block
save is preserved when selection overwrites `evader.zip`), self-describing
`selected.txt`, and entrant labels that distinguish `selected bN` from
`final b299`.

## E24 — hazard-death incentive (SETUP)
`r_hazard == r_caught == -8`: the evader is exactly indifferent between dying on
spikes and being caught, while the hand-coded expert treats spikes as a hard
avoidance constraint. Learned evaders therefore pay 17-23 spike deaths per 100
episodes where the expert pays 4, and every spike death is an escape that never
happened. Two 150-block fine-tunes warm-started from run21's final policy (the
best measured policy) isolate the fix: run24 `--r-hazard -14` (evader side
only) and run25 `--r-hazard -14 --r-hazard-ch -14` (also punish the chaser's
own spike suicide, its worst defect). Early signal: with the chaser penalty the
co-trained chaser survives to ~233 steps instead of ~176.

## E25 — the improvement loop has hit a plateau, and 100-match claims were noise
Definitive measurement: ONE hardarena call (identical arena set seed0=9000, identical
env seeds, 300 matches each, vs run21's chaser), so the comparison is paired:

| policy | escape | ev_hazard |
|---|---:|---:|
| DAgger distillation only (dagger_run23, 15 min, no RL) | **35.3%** | 42 |
| run21 trained (final b299) | 32.1% | 44 |
| run23 trained (selected b260) | 30.5% | 42 |
| run27 trained (150 blocks, frozen opponent, no warmup) | 30.0% | 81 |
| run26 trained (frozen opponent + hazard rewards) | **19.3%** | 112 |

Everything learned lands at or below the distillation; the scripted expert measured
33% on this block at 100 matches. Conclusions that survive:
- **No strategy tested clearly beats run21's final policy or a plain distillation.**
  Distillation reaches that level in ~15 minutes where RL takes hours.
- **The hazard-reward change fails** (run26: 19.3%, i.e. spike deaths rose to 112
  per ~300 episodes, the worst of any policy): making spike death worse than
  capture pushes the evader into more spikes, not fewer.
- **Freezing the opponent does not recover the erosion either**: run27 (frozen
  run23 chaser, warmup disabled) ends level with its own starting point (30.0%
  vs run21's 32.1%) and doubles the spike-death rate.
- Every learned policy dies on spikes 4-11x as often as the scripted expert
  (42-112 vs ~10 per test): the dominant remaining failure mode is hazard
  handling, and it is not fixed by penalizing it harder.

### Measurement protocol (two defects found while making the table above)
1. **A 100-match, single-block comparison cannot resolve the differences being
   claimed.** run27's last block scored 43% at 100 matches and 30% at 300 with
   the same arenas; run22's checkpoint sweep produced 57-62% estimates that fell
   to 26-33% on a disjoint block. Treat anything under ~300 matches and without a
   paired design as a hypothesis, not a result.
2. **The `--run` flag changes the measurement, not just the entrant.** run21's
   evader_final.zip scores 32.1% with `--run results/run21_fixed` and 28.9% with
   `--run results/run24_haz_ev` on the same arena block (identical weights, same
   command otherwise): the run supplies the env params (n_envs -> batch layout,
   reward values, obs geometry). Therefore competing policies must be measured in
   ONE call via repeated `--evader-ckpt`, which is what `--evader-ckpt` exists for.
