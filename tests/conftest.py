"""Shared pytest fixtures/helpers for the ori-engine test suite.

Set ORI_LEGACY_PHYSICS=1 to run the suite against the historical
`oripark/physics.py` from git HEAD (the pre-fix movement model). That is how
the exploit-regression tests are proven to have teeth: they must FAIL on the
legacy model and PASS on the fixed one.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from oripark.config import EnvParams, MoveParams  # noqa: E402
from oripark.physics import EMPTY, ORB, PORTAL, SOLID, SPIKE  # noqa: E402


def _install_legacy_physics(rev: str = "HEAD"):
    """Replace oripark.physics.OriPhysics with the git-HEAD implementation."""
    import oripark.physics as phys_mod

    src = subprocess.check_output(["git", "show", f"{rev}:oripark/physics.py"],
                                  cwd=ROOT, text=True)
    src = src.replace("from .config import MoveParams",
                      "from oripark.config import MoveParams")
    spec = importlib.util.spec_from_loader("oripark._legacy_physics", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, "<legacy physics>", "exec"), mod.__dict__)
    sys.modules["oripark._legacy_physics"] = mod
    phys_mod.OriPhysics = mod.OriPhysics
    return mod


if os.environ.get("ORI_LEGACY_PHYSICS"):
    _install_legacy_physics(os.environ.get("ORI_LEGACY_PHYSICS_REV", "HEAD"))

from oripark.physics import OriPhysics  # noqa: E402  (after the patch above)


@pytest.fixture
def mp() -> MoveParams:
    return MoveParams()


@pytest.fixture
def ep() -> EnvParams:
    return EnvParams()


def blank_grid(w: int = 40, h: int = 24) -> np.ndarray:
    return np.zeros((h, w), dtype=np.int32)


def make_physics(grid: np.ndarray, mp: MoveParams | None = None,
                 seed: int = 0) -> tuple[OriPhysics, MoveParams]:
    """One-agent physics on a single arena grid."""
    mp = mp or MoveParams()
    g = np.asarray(grid, dtype=np.int32)[None, ...]
    return OriPhysics(1, g, mp, np.random.default_rng(seed)), mp


def flat_floor(w: int = 40, h: int = 24, floor_y: int | None = None) -> np.ndarray:
    g = blank_grid(w, h)
    g[floor_y if floor_y is not None else h - 5:, :] = SOLID
    return g


class Driver:
    """Edge-accurate action driver (mirrors OriArenaVecEnv's edge handling)."""

    def __init__(self, ph: OriPhysics, mp: MoveParams):
        self.ph = ph
        self.mp = mp
        self.prev = np.zeros(5, dtype=np.int32)

    def step(self, move: int = 0, jump: int = 0, dash: int = 0,
             bash: int = 0, aim: int = 0) -> None:
        cur = np.array([move, jump, dash, bash, aim], dtype=np.int32)
        if cur[1] and not self.prev[1]:
            self.ph.press_jump(np.array([True]))
        if not cur[1] and self.prev[1]:
            self.ph.release_jump(np.array([True]))
        if cur[2] and not self.prev[2]:
            self.ph.press_dash(np.array([True]))
        if cur[3] and not self.prev[3]:
            self.ph.press_bash(np.array([True]))
        self.ph.step(cur[None, :])
        self.prev = cur

    def run(self, steps: int, **kw) -> None:
        for _ in range(steps):
            self.step(**kw)


def wall_side_now(ph: OriPhysics, i: int = 0) -> int:
    """-1/0/+1 from raw tile probes: is a solid really beside the body?"""
    p = ph.p
    under = ph._probe_y(np.array([ph.x[i]]), np.array([ph.y[i] + p.half_h + 1.0]))[0] == SOLID
    for side, val in ((-1, -1), (1, 1)):
        xs = np.full((1, 3), ph.x[i] + side * (p.half_w + 1.0))
        ys = np.array([[ph.y[i] - p.half_h, ph.y[i], ph.y[i] + p.half_h]])
        pr = ph._probe_cells(xs, ys)[0].copy()
        if under:
            pr[2] = 0
        if np.any(pr == SOLID):
            return val
    return 0


def is_climbing(ph: OriPhysics, mp: MoveParams, i: int = 0) -> bool:
    return (abs(float(ph.vy[i]) + mp.climb_speed) < 1e-3 and not ph.on_ground[i])
