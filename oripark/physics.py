"""Ori-style 2D platformer physics, vectorized over N agents (numpy).

Recreates the movement model of *Ori and the Will of the Wisps*:
  - snappy ground acceleration / braking with instant direction flips
  - variable-height jumps (release early = short hop), coyote time, input buffer
  - one air double-jump per airtime (refreshed on landing)
  - wall slide + wall jump (chainable, refreshes nothing but is chainable)
  - instant dash in any of 8 directions with momentum retention after
  - bash launches off static orbs (refresh dash + double jump)

All state arrays are shape (N,). Positions are pixel centers, +y is down.
One `OriPhysics` instance simulates a batch of agents (evaders + chasers
share grids by duplication).

Tile codes: 0 empty, 1 solid, 2 spike (hazard), 3 orb, 4 portal (goal).
"""
from __future__ import annotations

import numpy as np

from .config import MoveParams

# 9-way move input -> (dx, dy); id 0 = idle
MOVE_DX = np.array([0, -1, 1, 0, 0, -1, 1, -1, 1], dtype=np.float32)
MOVE_DY = np.array([0, 0, 0, -1, 1, -1, -1, 1, 1], dtype=np.float32)

# 8-way aim (bash launch) -> (dx, dy); id 0 = east, CCW
AIM_DX = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.float32)
AIM_DY = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.float32)
_AIM_NORM = np.sqrt(2.0)

EMPTY, SOLID, SPIKE, ORB, PORTAL = 0, 1, 2, 3, 4


class OriPhysics:
    """Vectorized Ori-like movement + tile collision for N agents."""

    def __init__(self, n: int, grids: np.ndarray, p: MoveParams, rng: np.random.Generator):
        assert grids.ndim == 3 and grids.shape[0] == n
        self.n = n
        self.p = p
        self.grids = grids                      # (N, H, W) int
        self.h, self.w = grids.shape[1], grids.shape[2]
        self.rng = rng
        self.tile = float(p.tile)

        z = np.zeros(n, dtype=np.float32)
        self.x, self.y = z.copy(), z.copy()
        self.vx, self.vy = z.copy(), z.copy()
        self.on_ground = np.zeros(n, dtype=bool)
        self.wall_dir = np.zeros(n, dtype=np.int8)
        self.coyote = z.copy()
        self.buffer = z.copy()
        self.can_djump = np.ones(n, dtype=bool)
        self.can_dash = np.ones(n, dtype=bool)
        self.dash_t = z.copy()
        self.dash_dx = np.ones(n, dtype=np.float32)
        self.dash_dy = z.copy()
        self.bash_cd = z.copy()
        self.jump_held = np.zeros(n, dtype=bool)
        self.jump_press_t = np.full(n, 99.0, dtype=np.float32)
        self.wall_grace = z.copy()
        self.facing = np.ones(n, dtype=np.float32)
        self.died = np.zeros(n, dtype=bool)
        self.escaped = np.zeros(n, dtype=bool)
        self.last_actions = np.zeros((n, 5), dtype=np.int32)  # move, jump, dash, bash, aim

        # per-agent orb positions (padded to max 8 orbs)
        self.orbs = self._collect_orbs()
        # per-agent portal pixel position
        self.portals = np.full((n, 2), np.nan, dtype=np.float32)
        self._collect_portals()

    # ---------------------------------------------------------------- setup
    def _collect_orbs(self) -> np.ndarray:
        """(N, 8, 2) pixel positions of orbs per agent's grid, padded with NaN."""
        out = np.full((self.n, 8, 2), np.nan, dtype=np.float32)
        for i in range(self.n):
            g = self.grids[i]
            ys, xs = np.where(g == ORB)
            k = min(len(xs), 8)
            if k:
                out[i, :k, 0] = (xs[:k] + 0.5) * self.tile
                out[i, :k, 1] = (ys[:k] + 0.5) * self.tile
        return out

    def _collect_portals(self):
        for i in range(self.n):
            g = self.grids[i]
            ys, xs = np.where(g == PORTAL)
            if len(xs):
                self.portals[i, 0] = (xs[0] + 0.5) * self.tile
                self.portals[i, 1] = (ys[0] + 0.5) * self.tile

    def place(self, xs: np.ndarray, ys: np.ndarray, ground: bool = True, idx=None):
        """Teleport agents (optionally a subset `idx`); snap onto ground below."""
        if idx is None:
            idx = np.arange(self.n)
        idx = np.asarray(idx)
        self.x[idx] = xs
        self.y[idx] = ys
        self.vx[idx] = 0.0
        self.vy[idx] = 0.0
        self.on_ground[idx] = False
        self.wall_dir[idx] = 0
        self.died[idx] = False
        self.escaped[idx] = False
        self.can_djump[idx] = True
        self.can_dash[idx] = True
        self.dash_t[idx] = 0.0
        self.bash_cd[idx] = 0.0
        self.wall_grace[idx] = 0.0
        self.buffer[idx] = 0.0
        self.coyote[idx] = 0.0
        if ground:
            self._snap_to_ground(idx)

    def _snap_to_ground(self, idx):
        """Drop agents to the first solid tile below their feet (for spawns)."""
        p = self.p
        idx = np.asarray(idx)
        for _ in range(80):  # at most 80 tiles
            below = self._probe_for(idx, self.x[idx], self.y[idx] + p.half_h + 1.0)
            hit = (below > 0) | self.died[idx]
            if np.all(hit):
                break
            self.y[idx] = np.where(hit, self.y[idx], self.y[idx] + 2.0)
        # agents resting on solid ground are grounded (coyote + refreshes)
        rest = self._probe_for(idx, self.x[idx], self.y[idx] + p.half_h + 1.0) == SOLID
        self.on_ground[idx] = rest
        self.coyote[idx] = np.where(rest, p.coyote_time, self.coyote[idx])
        self.can_djump[idx] = rest | self.can_djump[idx]
        self.can_dash[idx] = rest | self.can_dash[idx]

    # ------------------------------------------------------------ input edges
    def press_jump(self, mask: np.ndarray):
        mask = np.asarray(mask, dtype=bool)
        self.buffer[mask] = self.p.jump_buffer
        self.jump_press_t[mask] = 0.0
        self.jump_held[mask] = True

    def release_jump(self, mask: np.ndarray):
        mask = np.asarray(mask, dtype=bool)
        held = self.jump_held & mask
        self.jump_held[mask] = False
        cut = held & (self.vy < 0) & (self.jump_press_t > self.p.jump_min_hold)
        self.vy[cut] *= self.p.jump_cut

    def press_dash(self, mask: np.ndarray):
        mask = np.asarray(mask, dtype=bool)
        want = mask & self.can_dash & (self.dash_t <= 0)
        mv = self.last_actions[:, 0].astype(np.int64)
        dx = np.where(mv == 0, self.facing, MOVE_DX[mv])          # default to facing
        dy = MOVE_DY[mv]
        norm = np.sqrt(dx * dx + dy * dy)
        norm = np.where(norm < 0.5, 1.0, norm)
        self.dash_dx[want] = dx[want] / norm[want]
        self.dash_dy[want] = dy[want] / norm[want]
        self.dash_t[want] = self.p.dash_time
        self.can_dash[want] = False

    def press_bash(self, mask: np.ndarray):
        """Launch off the nearest orb within range, along the aim channel."""
        mask = np.asarray(mask, dtype=bool)
        want = mask & (self.bash_cd <= 0)
        if not np.any(want):
            return
        p = self.p
        idx = np.where(want)[0]
        for i in idx:
            orb = self.orbs[i]
            d = np.hypot(orb[:, 0] - self.x[i], orb[:, 1] - self.y[i])
            if not np.any(~np.isnan(d)):
                continue
            j = int(np.nanargmin(d))
            if d[j] <= p.bash_range:
                aim = self.last_actions[i, 4]
                dx = AIM_DX[aim]
                dy = AIM_DY[aim]
                norm = _AIM_NORM if (dx != 0 and dy != 0) else 1.0
                self.vx[i] = dx / norm * p.bash_speed
                self.vy[i] = dy / norm * p.bash_speed
                self.dash_t[i] = 0.0
                self.can_dash[i] = True if p.bash_refresh else self.can_dash[i]
                self.can_djump[i] = True if p.bash_refresh else self.can_djump[i]
                self.bash_cd[i] = p.bash_cooldown

    # ------------------------------------------------------------- simulation
    def step(self, actions: np.ndarray):
        """One 60 Hz step: record actions, run `substeps` physics substeps."""
        self.last_actions[:] = actions
        p = self.p
        dt = 1.0 / (p.step_hz * p.substeps)
        for _ in range(p.substeps):
            self._substep(dt)

    def _substep(self, dt: float):
        p = self.p
        on_g = self.on_ground
        dashing = self.dash_t > 0

        # --- timers
        self.coyote = np.where(on_g, p.coyote_time, np.maximum(0.0, self.coyote - dt))
        self.buffer = np.maximum(0.0, self.buffer - dt)
        self.bash_cd = np.maximum(0.0, self.bash_cd - dt)
        self.wall_grace = np.maximum(0.0, self.wall_grace - dt)
        self.jump_press_t = np.minimum(self.jump_press_t + dt, 1.0)
        self.dash_t = np.maximum(0.0, self.dash_t - dt)
        dashing = self.dash_t > 0

        mv = self.last_actions[:, 0].astype(np.int64)
        move_x = MOVE_DX[mv]
        target = move_x * p.max_run
        if np.any(move_x != 0):
            self.facing = np.where(move_x != 0, move_x, self.facing)

        # --- horizontal acceleration / braking (disabled while dashing)
        vx = self.vx
        if np.any(~dashing):
            accel = np.where(on_g & ~dashing,
                             np.where(np.abs(vx) < np.abs(target) + 1e-3, p.ground_accel, p.ground_decel),
                             p.air_accel)
            dv = np.clip(target - vx, -accel * dt, accel * dt)
            vx = np.where(~dashing, vx + dv, vx)
        # post-dash momentum bleed-off when faster than max_run
        over = (~dashing) & (np.abs(vx) > p.max_run)
        vx = np.where(over, vx - np.sign(vx) * np.minimum(np.abs(vx) - p.max_run, p.post_dash_decay * dt), vx)

        # --- vertical: gravity or wall slide (disabled while dashing)
        vy = self.vy
        sliding = (~on_g) & (self.wall_dir != 0) & (vy >= 0) & (~dashing)
        g = p.gravity * np.where((vy < 0) & (~sliding), p.rise_gravity_mult, 1.0)
        vy = np.where(sliding,
                      np.minimum(vy + p.wall_slide_accel * dt, p.wall_slide_max),
                      vy + np.where(dashing, 0.0, g * dt))

        # --- jumps (wall jump first; it consumes the buffer)
        buffered = self.buffer > 0
        wj = buffered & (self.wall_dir != 0) & (self.wall_grace <= 0) & (~on_g)
        if np.any(wj):
            vx = np.where(wj, -self.wall_dir.astype(np.float32) * p.walljump_h, vx)
            vy = np.where(wj, p.walljump_v, vy)
            self.wall_grace[wj] = p.walljump_grace
            self.buffer[wj] = 0.0
            self.coyote[wj] = 0.0
            self.on_ground[wj] = False
        buffered = self.buffer > 0
        ground_jump = buffered & (on_g | (self.coyote > 0)) & (~dashing)
        if np.any(ground_jump):
            vy = np.where(ground_jump, p.jump_v, vy)
            self.on_ground[ground_jump] = False
            self.coyote[ground_jump] = 0.0
            self.buffer[ground_jump] = 0.0
            # NOTE: ground jump does NOT consume the double jump (Ori feel)
        buffered = self.buffer > 0
        air_jump = buffered & self.can_djump & (~on_g) & (self.coyote <= 0) & (~dashing)
        if np.any(air_jump):
            vy = np.where(air_jump, p.djump_v, vy)
            self.can_djump[air_jump] = False
            self.buffer[air_jump] = 0.0

        # --- dash velocity override
        if np.any(dashing):
            vx = np.where(dashing, self.dash_dx * p.dash_speed, vx)
            vy = np.where(dashing, self.dash_dy * p.dash_speed, vy)

        self.vx = vx
        self.vy = vy

        # --- integrate + collide (per axis)
        self._integrate_x(dt)
        self._integrate_y(dt)

        # --- hazards / goal
        self._check_cells()

    # -------------------------------------------------------------- collision
    def _probe_y(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Gather tile value at (x, y) per agent; 0 if outside or empty."""
        t = self.tile
        xi = np.clip((xs / t).astype(np.int64), 0, self.w - 1)
        yi = np.clip((ys / t).astype(np.int64), 0, self.h - 1)
        flat = yi * self.w + xi
        g = self.grids.reshape(self.n, -1)
        return g[np.arange(self.n), flat]

    def _probe_for(self, idx, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Tile values at (x, y) for agent subset `idx` (positions per agent)."""
        idx = np.asarray(idx)
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        t = self.tile
        xi = np.clip((xs / t).astype(np.int64), 0, self.w - 1)
        yi = np.clip((ys / t).astype(np.int64), 0, self.h - 1)
        flat = yi * self.w + xi
        g = self.grids.reshape(self.n, -1)
        return g[idx, flat]

    def _probe_cells(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Gather tile values for a batch of (x, y) points -> (N, K)."""
        t = self.tile
        xs = np.atleast_2d(xs)
        ys = np.atleast_2d(ys)
        xi = np.clip((xs / t).astype(np.int64), 0, self.w - 1)
        yi = np.clip((ys / t).astype(np.int64), 0, self.h - 1)
        flat = yi * self.w + xi                      # (N, K)
        g = self.grids.reshape(self.n, -1)
        rows = np.arange(self.n)[:, None]
        return g[rows, flat]

    def _integrate_x(self, dt: float):
        p = self.p
        hw = p.half_w
        hh = p.half_h
        nx = self.x + self.vx * dt
        # probe leading edge at 3 heights
        edge = nx + np.sign(self.vx) * hw
        probes = self._probe_cells(
            np.stack([edge, edge, edge], axis=1),
            np.stack([self.y - hh, self.y, self.y + hh], axis=1),
        )
        hit = np.any(probes == SOLID, axis=1)
        moving = np.abs(self.vx) > 1e-6
        hit = hit & moving
        if np.any(hit):
            t = p.tile
            left_of = np.floor((nx + hw) / t).astype(np.int64)
            right_of = np.ceil((nx - hw) / t).astype(np.int64)
            nx = np.where(hit & (self.vx > 0), left_of * t - hw, nx)
            nx = np.where(hit & (self.vx < 0), right_of * t + hw, nx)
            hit_right = hit & (self.vx > 0)
            hit_left = hit & (self.vx < 0)
            self.vx[hit] = 0.0
            sgn = np.zeros(self.n, dtype=np.int8)
            sgn[hit_right] = 1
            sgn[hit_left] = -1
            self.wall_dir[hit] = sgn[hit]
            if np.any(hit & (self.dash_t > 0)):
                self.dash_t[hit & (self.dash_t > 0)] = 0.0
        self.x = nx
        # wall probe when airborne and not moving horizontally
        idle = (~self.on_ground) & (np.abs(self.vx) < 1e-3)
        if np.any(idle):
            l = self._probe_y(self.x - hw - 1.0, self.y)
            r = self._probe_y(self.x + hw + 1.0, self.y)
            self.wall_dir[idle] = np.where(r[idle] == SOLID, 1, np.where(l[idle] == SOLID, -1, 0)).astype(np.int8)

    def _integrate_y(self, dt: float):
        p = self.p
        hw = p.half_w
        hh = p.half_h
        self.on_ground[:] = False
        ny = self.y + self.vy * dt
        edge = ny + np.sign(self.vy) * hh
        probes = self._probe_cells(
            np.stack([self.x - hw, self.x, self.x + hw], axis=1),
            np.stack([edge, edge, edge], axis=1),
        )
        hit = np.any(probes == SOLID, axis=1)
        moving = np.abs(self.vy) > 1e-6
        hit = hit & moving
        if np.any(hit):
            t = p.tile
            top_of = np.floor((ny + hh) / t).astype(np.int64)
            bot_of = np.ceil((ny - hh) / t).astype(np.int64)
            falling = hit & (self.vy > 0)
            rising = hit & (self.vy < 0)
            ny = np.where(falling, top_of * t - hh, ny)
            ny = np.where(rising, bot_of * t + hh, ny)
            self.vy[hit] = 0.0
            self.on_ground[falling] = True
            self.can_djump[falling] = True
            self.can_dash[falling] = True
            if np.any(falling & (self.dash_t > 0)):
                self.dash_t[falling & (self.dash_t > 0)] = 0.0
        self.y = ny

    def _check_cells(self):
        """Spike death, portal escape, void death."""
        p = self.p
        corners_x = np.stack([self.x - p.half_w, self.x + p.half_w, self.x, self.x], axis=1)
        corners_y = np.stack([self.y - p.half_h, self.y - p.half_h, self.y, self.y + p.half_h], axis=1)
        cells = self._probe_cells(corners_x, corners_y)          # (N, 4)
        self.died |= np.any(cells == SPIKE, axis=1)
        self.escaped |= np.any(cells == PORTAL, axis=1)
        self.died |= self.y > (self.h - 1) * p.tile + 80.0

    # ----------------------------------------------------------------- helpers
    def aabb_overlap(self, other: "OriPhysics", i0: slice, i1: slice) -> np.ndarray:
        """Pairwise AABB overlap between agent sets i0 (self) and i1 (other)."""
        p = self.p
        dx = np.abs(self.x[i0] - other.x[i1])
        dy = np.abs(self.y[i0] - other.y[i1])
        return (dx < 2 * p.half_w) & (dy < 2 * p.half_h)

    def get_state(self, idx: slice) -> dict:
        return dict(
            x=self.x[idx].copy(), y=self.y[idx].copy(),
            vx=self.vx[idx].copy(), vy=self.vy[idx].copy(),
            on_ground=self.on_ground[idx].copy(),
            wall_dir=self.wall_dir[idx].copy().astype(np.float32),
            coyote=self.coyote[idx].copy(), buffer=self.buffer[idx].copy(),
            can_djump=self.can_djump[idx].copy(), can_dash=self.can_dash[idx].copy(),
            dash_t=self.dash_t[idx].copy(), bash_cd=self.bash_cd[idx].copy(),
            facing=self.facing[idx].copy(),
        )

    def set_portals(self, portals: np.ndarray):
        """(N, 2) pixel positions of each agent's portal goal."""
        self.portals[:] = portals
