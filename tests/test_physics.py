"""Movement-model invariants for the Ori-style physics.

The headline regression is the wall-contact rule: an agent may only climb,
wall-jump or wall-slide while a solid tile is actually beside its body.
The pre-fix model carried `wall_dir` over from any earlier collision, which
turned the evader into a flying agent (see .audit/verify/REPORT.md, F1).
"""
from __future__ import annotations

import numpy as np

from oripark.physics import SOLID, SPIKE
from conftest import (Driver, flat_floor, is_climbing, make_physics,
                      wall_side_now)


def spawn_on_floor(ph, mp, tile_x: float, floor_y: float):
    ph.place(np.array([tile_x * mp.tile]), np.array([floor_y * mp.tile - mp.half_h - 1.0]),
             ground=True)


# --------------------------------------------------------------- wall contact
def test_climb_never_happens_without_an_adjacent_wall(mp):
    """The P0 invariant: climbing implies a solid tile beside the body."""
    g = flat_floor(40, 24)
    g[8:20, 8] = SOLID                      # a full-height pillar
    ph, mp = make_physics(g, mp)
    spawn_on_floor(ph, mp, 10.0, 19.0)
    d = Driver(ph, mp)
    rng = np.random.default_rng(0)
    for t in range(2000):
        mv = int(rng.integers(0, 9))
        jump = int(rng.random() < 0.15)
        dash = int(rng.random() < 0.05)
        d.step(move=mv, jump=jump, dash=dash)
        assert ph.x[0] == ph.x[0]          # no NaN
        if is_climbing(ph, mp, 0):
            assert int(ph.wall_dir[0]) != 0, "climb with wall_dir == 0"
            assert wall_side_now(ph, 0) == int(ph.wall_dir[0]), (
                f"step {t}: climbing at ({ph.x[0]/mp.tile:.2f}, {ph.y[0]/mp.tile:.2f}) "
                f"with wall_dir={int(ph.wall_dir[0])} but no solid on that side")


def test_wall_jump_away_from_wall_then_hold_stale_direction_falls(mp):
    """Wall-jump off a pillar, then hold the old wall direction: the agent
    must fall back to the floor instead of flying (legacy: it rose for the
    rest of the episode)."""
    g = flat_floor(40, 24)
    g[3:19, 8] = SOLID                      # pillar from near-ceiling to floor
    ph, mp = make_physics(g, mp)
    spawn_on_floor(ph, mp, 10.0, 19.0)
    d = Driver(ph, mp)

    d.run(70, move=1)                       # run left into the pillar, climb
    assert is_climbing(ph, mp, 0), "legitimate wall climb stopped working"
    d.step(move=0, jump=1)                  # wall jump (kicked right, away)
    d.step(move=0)
    y_after_jump = float(ph.y[0])
    apex_after_jump = y_after_jump  # noqa: F841

    climbed_without_wall = 0
    streak = worst_streak = 0
    for _ in range(200):
        d.step(move=1)                      # hold the STALE wall direction
        if is_climbing(ph, mp, 0) and wall_side_now(ph, 0) == 0:
            climbed_without_wall += 1
            streak += 1
            worst_streak = max(worst_streak, streak)
        else:
            streak = 0
    # one transition step at the wall's top edge is physical (contact was
    # real when the substep began); a sustained run means exploitation
    assert climbed_without_wall <= 3, f"{climbed_without_wall} phantom climb steps"
    assert worst_streak <= 2, f"{worst_streak} consecutive phantom climb steps"
    assert ph.on_ground[0], (
        f"agent never landed: y={ph.y[0]/mp.tile:.2f} tiles "
        f"(apex {apex_after_jump/mp.tile:.2f}, floor at 19.0)")


def test_wall_jump_needs_contact(mp):
    """After a wall jump, a second jump press in mid-air with no contact and
    no double jump must not produce a fresh wall-jump kick (legacy: the stale
    wall_dir made wall jumps chainable in open air -> sustained ascent)."""
    g = flat_floor(40, 24)
    g[3:19, 8] = SOLID
    ph, mp = make_physics(g, mp)
    spawn_on_floor(ph, mp, 10.0, 19.0)
    d = Driver(ph, mp)
    d.run(70, move=1)                       # contact + climb
    d.step(move=0, jump=1)                  # wall jump
    ph.can_djump[:] = False                 # no double jump left
    d.run(30, move=2)                       # out of contact, past wall_grace
    assert not ph.on_ground[0]
    assert wall_side_now(ph, 0) == 0, "test setup: still touching a wall"
    assert float(ph.vy[0]) > 0, "test setup: still rising"
    vy_before = float(ph.vy[0])
    d.step(move=2, jump=1)
    assert float(ph.vy[0]) > vy_before - 60, (
        f"wall jump fired with no contact: vy {vy_before:.0f} -> {ph.vy[0]:.0f}")


def test_wall_slide_only_next_to_a_wall(mp):
    """Against a wall the fall is capped at wall_slide_max; in open air it
    accelerates well past it."""
    g = flat_floor(40, 30)
    g[2:20, 8] = SOLID
    ph, mp = make_physics(g, mp)
    # in contact with the pillar's right face, falling
    ph.place(np.array([8 * mp.tile + mp.half_w + 1.0]), np.array([4 * mp.tile]), ground=False)
    d = Driver(ph, mp)
    d.run(40, move=2)
    assert float(ph.vy[0]) <= mp.wall_slide_max + 1e-3, "wall slide not capped"

    ph2, mp = make_physics(flat_floor(40, 30), mp)
    ph2.place(np.array([20 * mp.tile]), np.array([2 * mp.tile]), ground=False)
    d2 = Driver(ph2, mp)
    d2.run(40, move=0)
    assert float(ph2.vy[0]) > mp.wall_slide_max * 2, "open-air fall was capped"


# ------------------------------------------------------------- core movement
def test_grounded_running_reaches_max_run(mp):
    """E2 regression: the feet probe must not treat the floor as a wall."""
    g = flat_floor(40, 24, floor_y=19)
    ph, mp = make_physics(g, mp)
    spawn_on_floor(ph, mp, 3.0, 19.0)
    d = Driver(ph, mp)
    d.run(120, move=2)
    assert float(ph.vx[0]) > mp.max_run * 0.95
    assert float(ph.x[0]) / mp.tile > 20.0, "did not traverse the floor"


def test_double_jump_is_once_per_airtime(mp):
    g = flat_floor(40, 30)
    ph, mp = make_physics(g, mp)
    spawn_on_floor(ph, mp, 5.0, 25.0)
    d = Driver(ph, mp)
    d.step(move=0, jump=1)
    apex1 = float(ph.y[0])
    for _ in range(22):                     # rise to the first apex
        d.step(move=0, jump=1)
        apex1 = min(apex1, float(ph.y[0]))
    d.step(move=0, jump=0)                  # release (edge semantics)
    d.step(move=0, jump=1)                  # double jump
    assert not ph.can_djump[0], "double jump did not consume the air jump"
    apex2 = float(ph.y[0])
    for _ in range(22):                     # rise to the second apex
        d.step(move=0, jump=1)
        apex2 = min(apex2, float(ph.y[0]))
    assert apex2 < apex1 - 4, (
        f"double jump did not gain height (apex1 {apex1:.1f}, apex2 {apex2:.1f})")
    d.step(move=0, jump=0)                  # release again
    vy_before = float(ph.vy[0])
    d.step(move=0, jump=1)                  # third jump must do nothing
    assert float(ph.vy[0]) > vy_before - 1e-6, "a third jump fired in mid-air"


def test_full_jump_apex_matches_documented_height(mp):
    """Docs claim a full jump apex of ~144 px (4.5 tiles)."""
    g = flat_floor(40, 30)
    ph, mp = make_physics(g, mp)
    spawn_on_floor(ph, mp, 5.0, 25.0)
    d = Driver(ph, mp)
    y0 = float(ph.y[0])
    d.step(move=0, jump=1)
    ys = []
    for _ in range(40):
        d.step(move=0, jump=1)              # hold to full height
        ys.append(float(ph.y[0]))
    apex = y0 - min(ys)
    assert 125.0 <= apex <= 165.0, f"apex {apex:.1f} px ({apex/mp.tile:.2f} tiles) outside documented band"


def test_coyote_time_allows_late_jump(mp):
    g = flat_floor(40, 30)
    g[18:, :12] = SOLID                     # ledge ends at x=12
    ph, mp = make_physics(g, mp)
    spawn_on_floor(ph, mp, 10.0, 18.0)
    d = Driver(ph, mp)
    d.run(20, move=2)                       # walk off the ledge
    assert not ph.on_ground[0]
    d.step(move=2, jump=1)
    assert float(ph.vy[0]) < -300, "coyote-time jump did not fire"


def test_terminal_velocity_is_capped(mp):
    g = flat_floor(40, 40)
    ph, mp = make_physics(g, mp)
    ph.place(np.array([20 * mp.tile]), np.array([2 * mp.tile]), ground=False)
    d = Driver(ph, mp)
    d.run(60, move=0)
    assert float(ph.vy[0]) <= mp.max_fall + 1e-3


def test_no_tunnelling_at_terminal_velocity(mp):
    """A 1-tile-thick platform must stop an agent falling at max_fall."""
    g = flat_floor(40, 40)
    g[20, 15:25] = SOLID
    ph, mp = make_physics(g, mp)
    ph.place(np.array([20 * mp.tile]), np.array([3 * mp.tile]), ground=False)
    d = Driver(ph, mp)
    d.run(80, move=0)
    assert ph.on_ground[0], "fell through a one-tile platform"
    assert abs(float(ph.y[0]) - (20 * mp.tile - mp.half_h)) < 2.0
