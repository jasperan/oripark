"""Procedural tile arenas for the Ori tag game.

Path-first generation: lay a canonical chain of platforms from the evader
spawn to the portal, then decorate with climb towers, spike pits, dash
gaps and bash orbs. A coarse reachability BFS gates generation, so the
adversarial level generator can only propose levels that are (generously)
solvable by the evader's movement kit.

Tile codes: 0 empty, 1 solid, 2 spike, 3 orb, 4 portal.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import MoveParams
from .physics import EMPTY, SOLID, SPIKE, ORB, PORTAL


@dataclass
class Arena:
    grid: np.ndarray
    ev_spawn: tuple            # pixel center
    ch_spawn: tuple
    portal: tuple
    orbs: list                 # pixel centers
    params: np.ndarray = field(default_factory=lambda: np.full(6, 0.5))


class ArenaGenerator:
    """Parametrized arena generator. params: 6 normalized values in [0, 1]:

    [0] gap_scale      horizontal gap sizes (2..5 tiles)
    [1] climb_scale    tower heights (4..9 tiles) + climb frequency
    [2] spike_prob     probability of spike pits under deep gaps
    [3] orb_count      number of bash orbs (0..5)
    [4] wander         vertical zig-zag variance / platform length
    [5] dash_gap_prob  wide same-level gaps requiring dash / full-speed jump
    """

    def __init__(self, p: MoveParams, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        self.W, self.H = p.arena_w, p.arena_h

    def generate(self, params: np.ndarray, seed: int | None = None) -> Arena:
        rng = np.random.default_rng(seed if seed is not None else int(self.rng.integers(2**31)))
        for _ in range(40):
            a = self._try(rng, params)
            if a is not None:
                return a
        return self._fallback(params)

    # ------------------------------------------------------------- generation
    def _try(self, rng: np.random.Generator, params: np.ndarray) -> Arena | None:
        p = self.p
        W, H = self.W, self.H
        gap_scale, climb_scale, spike_prob, orb_count, wander, dash_gap_prob = np.clip(params, 0, 1)

        grid = np.zeros((H, W), dtype=np.int32)
        grid[0, :] = SOLID
        grid[-1, :] = SOLID
        grid[:, 0] = SOLID
        grid[:, -1] = SOLID
        ground_top = H - 4
        grid[ground_top:, :] = SOLID

        def carve(x0: int, y0: int, w: int):
            x0 = max(1, min(x0, W - w - 1))
            y0 = min(max(1, y0), ground_top)
            w = max(3, min(w, W - x0 - 2))
            grid[y0, x0:x0 + w] = SOLID

        platforms = []          # (x0, y0, w)
        transitions = []        # (gap_x0, gap_x1, y_top) for spike pass
        x, y = 3, ground_top
        carve(x, y, 6)
        platforms.append((x, y, 6))

        n_seg = int(rng.integers(7, 10))
        for s in range(n_seg):
            climb_ratio = 0.28 + 0.30 * climb_scale
            pw = platforms[-1][2]
            r = rng.random()
            if r < climb_ratio and y > 6:
                # climb tower: gap -> wall -> gap -> higher platform
                dh = int(rng.integers(4, 5 + round(5 * climb_scale)))
                gap1 = int(rng.integers(1, 3))
                gap2 = int(rng.integers(1, 4))
                tw = 2
                tx = x + pw + gap1
                if tx + tw + gap2 + 4 >= W - 2:
                    break
                ny = y - dh
                if ny < 3:
                    ny = 3
                grid[ny:y, tx:tx + tw] = SOLID          # tower column
                nx = tx + tw + gap2
                w = int(rng.integers(5, 6 + round(4 * wander)))
                carve(nx, ny, w)
                platforms.append((nx, ny, w))
                transitions.append((x + pw, nx, y))
                x, y = nx, ny
            elif r < climb_ratio + 0.5 * dash_gap_prob:
                gap = int(rng.integers(6, 8))
                nx = x + pw + gap
                if nx + 5 >= W - 2:
                    break
                w = int(rng.integers(5, 9))
                carve(nx, y, w)
                platforms.append((nx, y, w))
                transitions.append((x + pw, nx, y))
                x, y = nx, y
            else:
                if rng.random() < 0.45:
                    dy = int(rng.integers(1, 4))
                    gap = int(rng.integers(3, 4 + round(2 * gap_scale)))
                else:
                    dy = 0
                    gap = int(rng.integers(2, 3 + round(3 * gap_scale)))
                nx = x + pw + gap
                if nx + 5 >= W - 2:
                    break
                ny = min(y + dy, ground_top)
                w = int(rng.integers(5, 10))
                carve(nx, ny, w)
                platforms.append((nx, ny, w))
                transitions.append((x + pw, nx, y))
                x, y = nx, ny

        # extend final platform toward the portal
        fx, fy, fw = platforms[-1]
        want = min(W - 8 - fx, 12)
        if fw < want:
            carve(fx, fy, want)
            platforms[-1] = (fx, fy, want)

        # spike pits: under gaps with >= 3 tiles of depth, with probability
        for gx0, gx1, y_top in transitions:
            if gx1 - gx0 < 3:
                continue
            depth = ground_top - y_top
            if depth < 3:
                continue
            if rng.random() >= spike_prob:
                continue
            pit_floor = ground_top - 1
            grid[pit_floor, gx0 + 1:gx1] = SPIKE

        # bash orbs
        orbs = []
        target = int(round(orb_count * 5))
        cands = []
        for gx0, gx1, y_top in transitions:      # over wide gaps
            if gx1 - gx0 >= 4:
                cands.append(((gx0 + gx1) // 2, y_top - 3))
        for pl in platforms:                      # near tower/edges
            cands.append((pl[0] + pl[2] - 1, pl[1] - 2))
        rng.shuffle(cands)
        for cx, cy in cands:
            if len(orbs) >= target:
                break
            if 1 <= cx < W - 1 and 1 <= cy < H - 1 and grid[cy, cx] == EMPTY:
                grid[cy, cx] = ORB
                orbs.append(((cx + 0.5) * p.tile, (cy + 0.5) * p.tile))

        # spawns / portal
        sx, sy, sw = platforms[0]
        ev_spawn = ((sx + sw / 2) * p.tile, (sy - 0.5) * p.tile)
        ex0, ey0, ew0 = platforms[-1]
        portal_cell = (min(ex0 + ew0 - 2, W - 2), ey0 - 1)
        grid[portal_cell[1], portal_cell[0]] = PORTAL
        portal = ((portal_cell[0] + 0.5) * p.tile, (portal_cell[1] + 0.5) * p.tile)

        mid = int(self.W * 0.55)
        ch_pl = min(platforms[1:], key=lambda pl: abs(pl[0] + pl[2] / 2 - mid)) \
            if len(platforms) > 2 else platforms[-1]
        cx0, cy0, cw0 = ch_pl
        ch_spawn = ((cx0 + cw0 / 2) * p.tile, (cy0 - 0.5) * p.tile)

        # reachability gate + level-interest gate (>=2 elevated platforms)
        start = (sx + sw // 2, sy - 1)
        goal = portal_cell
        elevated = sum(1 for pl in platforms if pl[1] < ground_top - 2)
        if not self._reachable(grid, start, goal) or elevated < 2:
            return None
        return Arena(grid, ev_spawn, ch_spawn, portal, orbs, np.asarray(params, dtype=np.float32))

    # --------------------------------------------------------------- reachability
    def _reachable(self, grid: np.ndarray, start: tuple, goal: tuple) -> bool:
        W, H = self.W, self.H
        empty = (grid == EMPTY) | (grid == PORTAL)
        below = np.zeros((H, W), dtype=bool)
        below[:-1, :] = grid[1:, :] == SOLID   # cell below is solid
        land = empty & below
        land[goal[1], goal[0]] = True
        if not land[start[1], start[0]]:
            return False

        seen = np.zeros((H, W), dtype=bool)
        seen[start[1], start[0]] = True
        dq = deque([start])

        def offer(nx: int, ny: int):
            if 0 <= nx < W and 0 <= ny < H and land[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                dq.append((nx, ny))

        while dq:
            cx, cy = dq.popleft()
            # fall / walk off ledges
            for dx in (-1, 0, 1):
                nx = cx + dx
                for dy in range(1, 31):
                    ny = cy + dy
                    if not (0 <= nx < W and 0 <= ny < H):
                        break
                    if grid[ny, nx] == SOLID:
                        break
                    offer(nx, ny)
            # single jump
            for dx in range(-5, 6):
                for dy in range(-3, 2):
                    offer(cx + dx, cy + dy)
            # double jump
            for dx in range(-6, 7):
                for dy in range(-6, 2):
                    offer(cx + dx, cy + dy)
            # dash
            for dx in range(-8, 9):
                for dy in range(-4, 5):
                    offer(cx + dx, cy + dy)
            # wall climb chains: wall within 2 columns, up to 18 tiles, with
            # continuous wall-hugging (multi wall-jump chains need no landing)
            for sx in (-2, -1, 1, 2):
                wx = cx + sx
                if not (0 <= wx < W):
                    continue
                top = cy - 1
                while top >= 0 and grid[top, wx] == SOLID:
                    top -= 1
                top += 1
                if top >= cy or cy - top > 18:
                    continue
                offer(cx + sx, top - 1)          # stand on the wall top
                for dy in range(1, cy - top):    # ledges while hugging the wall
                    offer(cx, cy - dy)
                    offer(cx + sx, cy - dy)
        return seen[goal[1], goal[0]]

    # ------------------------------------------------------------------ fallback
    def _fallback(self, params: np.ndarray) -> Arena:
        """Trivially solvable arena if stochastic generation fails."""
        p = self.p
        W, H = self.W, self.H
        grid = np.zeros((H, W), dtype=np.int32)
        grid[0, :] = SOLID
        grid[-1, :] = SOLID
        grid[:, 0] = SOLID
        grid[:, -1] = SOLID
        ground_top = H - 4
        grid[ground_top:, :] = SOLID
        grid[ground_top - 8:ground_top, W // 2 - 1:W // 2 + 1] = SOLID  # small tower
        orbs = []
        portal_cell = (W - 6, ground_top - 1)
        grid[portal_cell[1], portal_cell[0]] = PORTAL
        return Arena(
            grid,
            (3.5 * p.tile, (ground_top - 0.5) * p.tile),
            ((W - 14) * p.tile, (ground_top - 0.5) * p.tile),
            ((portal_cell[0] + 0.5) * p.tile, (portal_cell[1] + 0.5) * p.tile),
            orbs,
            np.asarray(params, dtype=np.float32),
        )


def render_map(grid: np.ndarray, agents: dict | None = None) -> str:
    """ASCII map for debugging / terminal replay."""
    chars = {0: " ", 1: "#", 2: "^", 3: "o", 4: "P"}
    lines = []
    for yy in range(grid.shape[0]):
        row = ""
        for xx in range(grid.shape[1]):
            row += chars.get(int(grid[yy, xx]), "?")
        lines.append(row)
    if agents:
        for name, (x, y) in agents.items():
            lines[int(y // 32)][:0]  # no-op placeholder
    return "\n".join(lines)
