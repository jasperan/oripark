"""Environment/measurement contracts.

These pin the behaviours that silently corrupted earlier results:
observation units inside the declared Box, dash/bash reading the CURRENT
step's channels, the CEM's win semantics, Elo pair updates, and the
escape-rate denominator.
"""
from __future__ import annotations

import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from oripark.arena import Arena  # noqa: E402
from oripark.env import OriArenaVecEnv
from oripark.metrics import Elo, elo_pair_update
from oripark.physics import ORB, PORTAL, SOLID
from oripark.selfplay import cem_score


def make_arena(mp, orbs=(), portal=(57, 36)):  # noqa: D401
    """Arena sized to MoveParams (the env asserts that shape)."""
    W, H = mp.arena_w, mp.arena_h
    grid = np.zeros((H, W), dtype=np.int32)
    grid[H - 3:, :] = SOLID
    for (ox, oy) in orbs:
        grid[oy, ox] = ORB
    grid[portal[1], portal[0]] = PORTAL
    return Arena(grid=grid,
                 ev_spawn=(3.5 * mp.tile, (H - 3) * mp.tile - 16.0),
                 ch_spawn=(12.5 * mp.tile, (H - 3) * mp.tile - 16.0),
                 portal=((portal[0] + 0.5) * mp.tile, (portal[1] + 0.5) * mp.tile),
                 orbs=[((ox + 0.5) * mp.tile, (oy + 0.5) * mp.tile) for ox, oy in orbs],
                 params=np.full(6, 0.5))


class OneArena:
    def __init__(self, arena):
        self.arena = arena

    def sample(self, n):
        return [self.arena for _ in range(n)]

    def mean_params(self):
        return self.arena.params


def one_env(arena, mp, ep, role="evader", **kw):
    return OriArenaVecEnv(role, 1, OneArena(arena), mp, ep, seed=0, **kw)


# ------------------------------------------------------------------- obs units
def test_observations_stay_inside_the_declared_box(mp, ep):
    """Every obs feature must sit inside Box(-5, 5).

    The old scalars divided pixel offsets by the arena size in TILES, so
    portal/opponent/orb offsets reached ±25-30 — 5x outside the space SB3
    declares for the policy.
    """
    arena = make_arena(mp, orbs=[(5, 34), (20, 34)], portal=(57, 36))
    env = one_env(arena, mp, ep)
    rng = np.random.default_rng(0)
    obs = env.reset()
    worst = 0.0
    for _ in range(300):
        a = np.stack([rng.integers(0, [9, 2, 2, 2, 8])])
        obs, _, done, _ = env.step(a)
        worst = max(worst, float(np.abs(obs).max()))
        if done[0]:
            obs = env.reset()
    env.close()
    assert worst <= 5.0, f"obs feature {worst:.1f} outside Box(-5, 5)"
    assert env.obs_dim == obs.shape[1]


# --------------------------------------------------- current-step action use
def test_dash_uses_the_current_step_move_direction(mp, ep):
    """A dash pressed while holding LEFT must dash left, not along the
    previous step's direction (which is what the old ordering produced)."""
    env = one_env(make_arena(mp), mp, ep)
    env.reset()
    env.step(np.array([[2, 0, 0, 0, 0]]))       # previous step: moving right
    env.step(np.array([[1, 0, 1, 0, 0]]))       # this step: hold left + dash
    assert float(env.phys.dash_dx[0]) < 0, "dash used the stale move channel"
    assert float(env.phys.vx[0]) < -100, "dash did not push left"
    env.close()


def test_bash_uses_the_current_step_aim(mp, ep):
    """The aim channel is consumed on the same step as the bash press; the
    old ordering always launched along the previous frame's aim (east)."""
    arena = make_arena(mp, orbs=[(10, 35)])
    env = one_env(arena, mp, ep)
    env.reset()
    # sit right next to the orb, on the ground
    env.phys.x[0] = float(env.phys.orbs[0][0, 0]) + 40.0
    env.phys.y[0] = float(env.phys.orbs[0][0, 1])
    env.step(np.array([[0, 0, 0, 1, 2]]))       # bash with aim=2 (straight up)
    assert float(env.phys.vy[0]) < -500, (
        f"bash did not use the current aim (vy={env.phys.vy[0]:.0f})")
    env.close()


# ------------------------------------------------------------- metric semantics
def test_cem_scores_timeout_as_a_draw():
    assert cem_score("escaped") == 1.0
    assert cem_score("ch_hazard") == 1.0
    assert cem_score("timeout") == 0.5, "timeout must be a draw, not a win"
    assert cem_score("caught") == 0.0
    assert cem_score("ev_hazard") == 0.0


def test_elo_pair_update_does_not_drift():
    ev, ch = Elo(), Elo()
    start = ev.rating + ch.rating
    for _ in range(200):
        elo_pair_update(ev, ch, 0.5)            # perfectly balanced match
    assert abs((ev.rating + ch.rating) - start) < 1e-9, "Elo pair drifted"
    assert abs(ev.rating - 1200.0) < 1e-9


def test_elo_pair_update_moves_both_sides_equally():
    ev, ch = Elo(), Elo()
    elo_pair_update(ev, ch, 1.0)                # evader wins everything
    assert ev.rating > 1200 > ch.rating
    assert abs((ev.rating - 1200) + (ch.rating - 1200)) < 1e-9


# --------------------------------------------------------------- denominators
class _StubEnv:
    """Minimal env stub that finishes one episode per step, cycling outcomes."""

    def __init__(self, outcomes, obs_dim=3):
        self.outcomes = list(outcomes)
        self.i = 0
        self.closed = False
        self.obs_dim = obs_dim

    def reset(self):
        return np.zeros((1, self.obs_dim), dtype=np.float32)

    def step(self, actions):
        out = self.outcomes[self.i % len(self.outcomes)]
        self.i += 1
        info = {"episode": {"r": 0.0, "l": 10}, "outcome": out, "ev_max_zone": 3}
        self.closed = False
        return (np.zeros((1, self.obs_dim), dtype=np.float32),
                np.zeros(1, dtype=np.float32), np.array([True]), [info])

    def close(self):
        self.closed = True


def test_play_episodes_counts_hazard_outcomes_in_the_denominator():
    from oripark.evals import play_episodes
    env = _StubEnv(["escaped", "ev_hazard", "caught", "ch_hazard", "timeout"])
    r = play_episodes(env, lambda obs: np.zeros((1, 5), dtype=np.int32),
                      max_steps=100, n_target=5)
    assert r["n"] == 5, "hazard episodes were dropped from the denominator"
    assert r["esc"] == 1 and r["caught"] == 1 and r["tm"] == 1
    assert r["ev_hazard"] == 1 and r["ch_hazard"] == 1
    assert abs(r["esc_rate"] - 1 / 5) < 1e-9
    assert env.closed


def test_play_episodes_stops_at_the_target():
    from oripark.evals import play_episodes
    env = _StubEnv(["caught"])
    r = play_episodes(env, lambda obs: np.zeros((1, 5), dtype=np.int32),
                      max_steps=1000, n_target=3)
    assert r["n"] == 3, f"ran {r['n']} episodes for a target of 3"


# ------------------------------------------------------------------ oracle
def test_oracle_calm_filter_is_not_panic_mode(mp, ep):
    """The arena oracle must hand the expert the REAL chaser position.

    It used to pass the expert its own position as `chaser_x/y`, which pins
    its flee logic permanently on (panic hops/dashes) and made the filter
    reject arenas a calm expert traverses easily — the documented '~half of
    hard arenas are winnable' became 15%.
    """
    from hardarena import expert_winnable
    from oripark.arena import ArenaGenerator
    gen = ArenaGenerator(mp, np.random.default_rng(1), chaser_frac=ep.chaser_spawn_frac)
    cands = []
    k = 0
    while len(cands) < 48:
        a = gen.generate(np.full(6, 0.7), seed=5000 + k)
        if a is not None:
            cands.append(a)
        k += 1
    calm = expert_winnable(cands, mp, ep, batch=16, mode="calm")
    panic = expert_winnable(cands, mp, ep, batch=16, mode="panic")
    assert len(calm) > len(panic), (
        f"calm oracle ({len(calm)}) should accept more arenas than the "
        f"panic-mode filter ({len(panic)})")
    assert len(calm) >= 8, f"calm oracle accepted only {len(calm)}/48 arenas"


# ------------------------------------------------------- metric accounting
class _OutcomeEnv:
    """Stub vec-env that emits one episode per step, cycling the outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.i = 0
        self.closed = False

    def reset(self):
        return None

    def step(self, _a):
        import numpy as np
        o = self.outcomes[self.i % len(self.outcomes)]
        self.i += 1
        info = {"episode": {"l": 100}, "outcome": o, "ev_max_zone": 3}
        return None, np.zeros(1), np.array([True]), [info]

    def close(self):
        self.closed = True


def test_play_episodes_counts_every_outcome():
    """Hazard endings must not be folded into `timeout` (they used to be, in
    three separate copies of this loop — including the one that defines the
    headline metric in hardarena.py)."""
    from oripark.evals import OUTCOMES, play_episodes
    stub = _OutcomeEnv(OUTCOMES)
    r = play_episodes(stub, lambda obs: None, max_steps=50, n_target=5)
    for o in OUTCOMES:
        assert r["counts"][o] == 1, f"{o} mis-counted: {r['counts']}"
    assert sum(r["counts"].values()) == r["n"] == 5
    assert r["esc"] == 1 and r["caught"] == 1 and r["tm"] == 1
    assert r["ev_hazard"] == 1 and r["ch_hazard"] == 1
    assert abs(r["esc_rate"] - 1 / 5) < 1e-9, r["esc_rate"]
    assert stub.closed


def test_artifact_tables_share_one_denominator():
    """Every escape-rate artifact must show escaped/n with n = all finished
    episodes, and the outcome columns must sum to n."""
    import re
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "hardarena.py", "--run", "results/run17",
         "--chaser-run", "results/run17", "--matches", "6", "--chaser",
         "stoch", "--set", "07", "--expert", "--out", "/tmp/_t_metric.md"],
        capture_output=True, text=True, cwd=PROJECT_ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    text = open("/tmp/_t_metric.md").read()
    rows = [l for l in text.splitlines() if l.startswith("| ") and "%" in l]
    assert rows, text
    for line in rows:
        cells = [c.strip() for c in line.strip("|").split("|")]
        m = re.match(r"([\d.]+)%\s*\((\d+)/(\d+)\)", cells[1])
        assert m, line
        pct, esc, n = float(m.group(1)), int(m.group(2)), int(m.group(3))
        nums = [int(cells[i]) for i in (2, 3, 4, 5, 6)]
        assert sum(nums) == n, f"outcome columns must sum to n: {line}"
        assert abs(pct - 100 * esc / n) < 0.6, f"percent must be esc/n: {line}"
