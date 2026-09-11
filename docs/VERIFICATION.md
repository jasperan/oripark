# Verification report

*A full audit of this repository's claims, re-run from scratch. Every number
below has a command behind it; the raw artifacts live in `.audit/verify/`
(git-excluded, force-added on the audit branch) and every fix has a test in
`tests/`.*

**TL;DR** — the physics engine, the environment and the metric tooling are
sound, and their bugs are now fixed and regression-tested. The **published
training results are not**: the agents could climb in mid-air, and once that
is fixed the pre-fix "trained" evaders are statistically indistinguishable
from a random init. The first legitimate training run is `run21_fixed`.

```bash
python -m pytest tests -q                     # 21 tests, all green
ORI_LEGACY_PHYSICS=1 python -m pytest tests/test_physics.py -q   # 4 fail (proof of teeth)
```

---

## 1. Claim-by-claim verdict

| claim | verdict | evidence |
|---|---|---|
| `hardarena --set 07 --chaser stoch` reproduces run19 44% / run17 31% / run20 57% | **REPRODUCED** — under the buggy physics | `.audit/verify/run19_vs_run17ch.md` |
| those numbers measure movement skill | **REFUTED** | 34-48% of the escape gain came from non-physical flight, §2 |
| run19 peak Elo 1352 @ b61 | **MISMATCH** | actual max 1364.2 @ b63 (`results/run19/blocks.jsonl`) |
| run20 matrix cell 54% | **MISMATCH** | stored artifact says 57% (`results/run20/hardarena_stoch.md`) |
| run17 "0.7-only = 26%" | **MISMATCH** | stored artifact says 31% |
| run19 BC NLL 0.15 | **UNSUPPORTED** | `bc.py` prints NLL, nothing persists it |
| `selected.txt` "ships b270 / b170" | **UNSUPPORTED as reproducible** | hand-written; `select.py` emits a different format |
| the arena oracle marks "~half of hard arenas winnable" | **REFUTED** (it reported 15%) | the oracle ran the expert in permanent panic mode, §3 |
| "the random floor is 4-8%, trained beats it by +27 to +49" | **REFUTED** as stated; the direction survives | the floor is ~0% (fresh inits), so the edge is larger, but the "lucky init" it blamed was an early *trained* snapshot (§4b) |
| docs/configs/run{N}_params.json == results/run{N}/params.json | VERIFIED | byte-identical |
| every `docs/media` link resolves | VERIFIED | all files present |
| full-jump apex 144 px, terminal velocity 1600 px/s, no tunnelling | VERIFIED | `tests/test_physics.py` |
| the scripted expert climbs a 26-tile tower | VERIFIED | `tests/test_physics.py`, `python -m oripark.scripted` |

## 2. P0: free flight in the movement model

`wall_dir` was written when a leading-edge collision fired and cleared only by
a probe that requires `|vx| < 1e-3`, so it survived for the rest of an episode.
`climbing`, `wall jump` and `wall slide` keyed off `wall_dir` with **no contact
test**, so an airborne agent holding that direction rose at `climb_speed`
(170 px/s) anywhere on the map.

A probe script touching a pillar, wall-jumping away and then holding the stale
direction climbed with the nearest solid tile **99 tiles away**; a trained
run19 evader did this for **9,856 steps across 40 environments** (~26 tiles of
free height each, 40/40 environments affected).

Fixed by deriving `wall_dir` from head/centre/feet tile probes every substep
(`_wall_contact()`), with the "the floor is not a wall" rule preserved.

**Impact** — one shared arena set, 100 seeded matches, stochastic evader and
chaser, identical draws for both physics builds
(`.audit/verify/exploit_impact.py`):

| evader / frozen chaser | legacy | fixed | Δ |
|---|---:|---:|---:|
| run17 / run17 ch | 31% | 26% | −5 |
| run19 / run17 ch | 44% | **20%** | **−24** |
| run19 / run19 ch | 38% | 26% | −12 |
| run20 / run20 ch | 57% | **35%** | −22 |

The architecture comparison (E14/E15) is therefore retracted: run19 depended on
the exploit roughly 2.4x more than run17 and falls *below* it once flight is
removed.

## 3. The arena oracle was testing the wrong expert

`expert_winnable()` handed the scripted expert **its own position** as
`chaser_x/chaser_y`, which pins `flee=True` for the whole episode (panic hops,
panic dashes, panic target-jumping), while a *random* chaser wandered the test
arena and killed either the expert or itself. The documented filter ("an arena
only counts if a competent player can actually win it") admitted 90/600 hard
arenas, not the documented ~half.

Now: `--oracle calm` (default) parks the chaser and feeds the real position →
**310/600** winnable; `--oracle panic` reproduces the legacy filter.
`tests/test_contracts.py::test_oracle_calm_filter_is_not_panic_mode` pins it.

## 4. Corrected measurements

Three protocol errors had to be fixed to get an honest number, and each one
moved the results in a different direction. The final protocol is: fixed
physics, 310 expert-winnable arenas (calm oracle), stochastic frozen chaser,
100 seeded matches with identical draws per entrant, and **every policy fed the
observation convention it was trained in** (`--legacy-obs`,
`--legacy-obs-chaser`).

**(a) The frozen chaser's observation convention.** The obs-units fix (F3)
changed the direction/distance channels by 32x. A pre-fix chaser fed post-fix
observations is crippled — it catches 7-34% of episodes instead of 27-46% —
which inflates *every* entrant's escape rate. Without this correction the
scripted expert appears to escape 56%; with it, 28-33% (protocol-dependent).

**(b) The "random" baseline was not random.** `pool_ev_0.pt` is loaded as
"random"/"untrained" by every tool here, and it is neither: the league trims
its pool front (`ev_pool.pop(0)`, `selfplay.py`) and the final save writes
`pool_ev_{i}` from the trimmed list, so `pool_ev_0` is the oldest *surviving
trained* snapshot. Evidence: weight L2 norm 77-92 (a fresh init is 46, and the
finished policy is 73-92); escape rate 18-42% on the hard-arena protocol where
five fresh inits escape 0.0%. This also explains the old "each run's init was a
chaos-lucky draw" story (E13/E16) — those files were early trained snapshots.
`selfplay.py` now writes `pool_ev_boot.pt` at startup, `oripark/baselines.py`
resolves the true pre-training checkpoint, and `hardarena --random-seeds K`
builds K fresh inits instead of reusing `pool_ev_0`.

**(c) My own interim conclusion was wrong.** After (a) alone the table read
"trained 33-38% vs random 37%", and I reported that the pre-fix training runs
were at random-init level. That comparison used the mislabeled baseline: the
真 fresh-init floor is 0%, and the pre-fix runs are far above it. The corrected
table, with cross-run columns against `run21_fixed`'s chaser (the only pursuer
trained under fixed physics):

| evader | vs run21 ch | vs run17 ch | zone |
|---|---:|---:|---:|
| scripted expert (privileged BFS) | **54.5%** | 27.7% | 10.5 |
| run21 trained (fixed physics) | **40%** | **41%** | 10.1 |
| run17 trained (pre-fix) | 29% | 21% | 8.3 |
| run19 trained (pre-fix) | 29% | 14% | 8.2 |
| run20 trained (pre-fix) | 30% | 15% | 8.0 |
| DAgger distillation of the expert | 29% | — | 8.8 |
| BC/init policy | 7-19% | 0-17% | 5.8-7.9 |
| fresh random init | **0%** (5 inits, 500 eps) | — | ~2 |

Readings that hold across every protocol variant we tried:

* **Training produces a large, real gain**: 0% (fresh init) → 29-30% (pre-fix
  runs) → 40% (fixed-physics retrain).
* **The pre-fix published numbers are still retracted**: 44% and 57% depended
  on free flight and on a chaser crippled by the obs mismatch.
* **The scripted expert is chaser-dependent** (27.7% vs 54.5%) while run21's
  learned evader is not (40/41%), so the learned policy is the more robust
  controller against the strongest opponent this repo has.
* **Spikes, not capture, are the dominant failure mode** for weak policies
  (BC/init: 18-40% ev_hazard, zone 5.8-7.9 of 12) and the chasers themselves
  die on hazards in 19-43% of episodes, run21's included.
* **Distillation works when done with DAgger** (29% with no RL) and fails
  badly when done with plain behaviour cloning (7-19%) — the teacher is
  privileged, so the student must be trained on states it actually reaches.

## 5. Metric defects (each found by re-running, each now tested)

| id | defect | effect | fix |
|---|---|---|---|
| F1 | `wall_dir` never cleared, no contact test | free flight (§2) | `_wall_contact()` every substep |
| F2 | eval/progress/tournament denominators dropped `ev_hazard`/`ch_hazard` | inflated escape rates, worst for weak policies | one shared loop, `oripark/evals.py` |
| F3 | obs offsets divided by arena size in tiles | features ±30 against a declared `Box(-5,5)`; "no orb" sentinel read as an orb 9 px away | divide by pixel size, clip |
| F4 | BC regularizer trained on `obs_list[:k]` (a prefix) | every flee demo excluded from fine-tuning | random subset |
| F5 | dash/bash read `last_actions` one step late | stale aim; expert bash always launched east | assign current-step channels first |
| F6 | CEM scored `info["ev_win"]`, which counts a timeout as a win | curriculum escalated against a learner that was only surviving | `cem_score()`: escape 1.0 / timeout 0.5 / death 0.0 |
| F7 | chaser Elo updated against the evader's post-update rating | pair sum drifted upward every block | one pre-update snapshot |
| F8 | in-training eval re-drew arenas and used argmax | noisy Elo; protocol differed from the documented stochastic one | fixed arena list + stochastic play |
| F9 | `progress.py`/`demo.py` used today's defaults for old runs | old policies silently re-measured under new physics/env | `load_run_params()` |
| F10 | oracle ran the expert in panic mode (§3) | wrong arena set | `--oracle calm` |
| F11 | `hardarena.py` folded hazard deaths into "timeout" and printed `esc/args.matches` against a rate computed over `esc/n_done` | a policy that impaled itself was reported as surviving; artifact denominators disagreed with the arithmetic | all entrants through `play_episodes`, five outcome columns that sum to `n` |

## 6. What survives

* **The movement model**, once the contact rule is enforced: sub-pixel
  collision, coyote time, jump buffering, wall jump, wall slide, wall climb,
  dash, bash-orbs, orb/portal/spike semantics — all now pinned by tests.
* **The metric protocol**: seeded draws, a stochastic frozen chaser,
  cross-chaser controls, expert-filtered arenas, and the random population.
  It is what made the exploit measurable at all.
* **The infrastructure**: 120 Hz vectorised physics, CEM curriculum, league
  snapshots, the renderer.
* **Traversal skill in the scripted expert**: 96/96 demo escapes on
  mid-difficulty arenas, and the only entrant that clearly beats random.

## 7. Reproduce

```bash
python -m pytest tests -q
python hardarena.py --run results/run17 --chaser-run results/run17 \
       --matches 100 --chaser stoch --set 07 --expert
python .audit/verify/exploit_impact.py --run results/run19 \
       --chaser-run results/run17 --matches 100 --set 07
ORI_LEGACY_PHYSICS=1 python -m pytest tests/test_physics.py -q
```

## 8. Open items

1. **Chaser policy quality is now the bottleneck.** The frozen chasers only
   catch 7-27% of episodes: they were trained against evaders that could fly.
   Evader escape rates are therefore partly a measure of "can you avoid spikes
   and drift to the portal". A chaser retrained under fixed physics is the
   honest opponent (run21 trains both sides).
2. **Residual 1-2 step "phantom" transitions** at a wall's top edge are
   measurement lag in the detector, not flight.
3. `select.py` and the hand-written `selected.txt` files should be reconciled
   so "ships b270" is reproducible from the committed tool.
