"""Scripted expert evader: BFS waypoint follower with the full Ori kit.

Used for two things:
  1. a sanity check that an arena is traversable by a competent agent
  2. behavior-cloning demonstrations: collect (obs, action) pairs from
     scripted rollouts that *escape*, then BC-pretrain the RL policy.

The controller is greedy: it follows a BFS path over "land" cells from its
spawn to the portal (same move model as the arena generator's reachability
gate), steering with the full kit — run, jump, double jump, wall jumps,
dashes, and bash launches.

IMPORTANT: the env consumes jump/dash/bash on rising edges, so the
controller PULSES those inputs (1 for one step, then 0), never holds them.
"""
from __future__ import annotations

import numpy as np
from collections import deque

from .physics import EMPTY, SOLID, SPIKE, ORB, PORTAL

MOVE_DX = np.array([0, -1, 1, 0, 0, -1, 1, -1, 1], dtype=np.float32)
MOVE_DY = np.array([0, 0, 0, -1, 1, -1, -1, 1, 1], dtype=np.float32)
AIM_DX = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.float32)
AIM_DY = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.float32)

RIGHT, LEFT = 2, 1
IDLE = 0
UP, DOWN = 3, 4
UP_RIGHT, UP_LEFT, DOWN_RIGHT, DOWN_LEFT = 7, 5, 8, 6


def bfs_path(grid: np.ndarray, start: tuple, goal: tuple):
    """BFS over land cells (empty with solid below) with the same move
    offers as ArenaGenerator._reachable. Returns a list of (x, y) cells
    from start to goal, or None if unreachable."""
    W, H = grid.shape[1], grid.shape[0]
    empty = (grid == EMPTY) | (grid == PORTAL) | (grid == ORB)
    below = np.zeros((H, W), dtype=bool)
    below[:-1, :] = grid[1:, :] == SOLID
    land = empty & below
    land[goal[1], goal[0]] = True
    if not land[start[1], start[0]]:
        return None

    parent = {}
    seen = np.zeros((H, W), dtype=bool)
    seen[start[1], start[0]] = True
    dq = deque([start])

    def offer(nx: int, ny: int, frm):
        if 0 <= nx < W and 0 <= ny < H and land[ny, nx] and not seen[ny, nx]:
            seen[ny, nx] = True
            parent[(nx, ny)] = frm
            dq.append((nx, ny))

    while dq:
        cx, cy = dq.popleft()
        if (cx, cy) == goal:
            path = [(cx, cy)]
            while (cx, cy) in parent:
                cx, cy = parent[(cx, cy)]
                path.append((cx, cy))
            return path[::-1]
        for dx in (-1, 0, 1):
            nx = cx + dx
            for dy in range(1, 31):
                ny = cy + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    break
                if grid[ny, nx] == SOLID:
                    break
                offer(nx, ny, (cx, cy))
        for dx in range(-5, 6):
            for dy in range(-3, 2):
                offer(cx + dx, cy + dy, (cx, cy))
        for dx in range(-6, 7):
            for dy in range(-6, 2):
                offer(cx + dx, cy + dy, (cx, cy))
        for dx in range(-8, 9):
            for dy in range(-4, 5):
                offer(cx + dx, cy + dy, (cx, cy))
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
            offer(cx + sx, top - 1, (cx, cy))
            for dy in range(1, cy - top):
                offer(cx, cy - dy, (cx, cy))
                offer(cx + sx, cy - dy, (cx, cy))
    return None


class ScriptedEvader:
    """Single-env greedy controller with privileged grid access."""

    def __init__(self, p, ep, lookahead_px: float = 56.0):
        self.p = p
        self.ep = ep
        self.lookahead = lookahead_px
        self.path = None
        self.idx = 0
        self.dj_used = False
        self.stuck = 0
        self.last_cx = -1
        self.wall_pulse = 0          # cooldown between wall-jump presses
        self.jump_cd = 0             # jump pulse cooldown (edge-triggered env)
        self.climb_mode = False      # tall wall ahead: chain wall jumps
        self.dash_pending = False    # pulse dash (edge-triggered in env)
        self.bash_pending = False

    def reset(self, arena):
        p = self.p
        t = float(p.tile)
        sx = int(arena.ev_spawn[0] / t)
        sy = int(arena.ev_spawn[1] / t)
        gx = int(arena.portal[0] / t)
        gy = int(arena.portal[1] / t)
        while sy > 0 and arena.grid[sy, sx] == SOLID:   # move up out of solid
            sy -= 1
        self.path = bfs_path(arena.grid, (sx, sy), (gx, gy))
        self.idx = 0
        self.dj_used = False
        self.stuck = 0
        self.last_cx = -1
        self.wall_pulse = 0
        self.jump_cd = 0
        self.climb_mode = False
        self.dash_pending = False
        self.bash_pending = False

    # ------------------------------------------------------------------
    def act(self, arena, x, y, vx, vy, on_ground, wall_dir, can_djump,
            can_dash, dash_t, bash_cd, facing, chaser_x=None, chaser_y=None) -> np.ndarray:
        """Return [move, jump, dash, bash, aim].

        chaser_x/chaser_y: optional pursuer position for flee behavior."""
        p = self.p
        t = float(p.tile)
        W, H = p.arena_w, p.arena_h
        g = arena.grid
        if self.path is None:
            return np.array([RIGHT, 1, 0, 0, 0], dtype=np.int32)  # fallback: run+hop

        cx = int(x / t)
        cy = int(y / t)

        # --- advance waypoint: skip any path cell whose left edge we have
        # reached (the agent is at-or-past it), so the target is always ahead
        while (self.idx + 1 < len(self.path) and
               self.path[self.idx][0] * t <= x):
            self.idx += 1
        wpx, wpy = self.path[self.idx]
        tx = (wpx + 0.5) * t
        ty = (wpy + 0.5) * t
        ddx = tx - x
        ddy = ty - y

        # stuck detection
        if cx == self.last_cx:
            self.stuck += 1
        else:
            self.stuck = 0
            self.last_cx = cx
            if on_ground:
                self.dj_used = False
                self.wall_pulse = 0

        # progress is rightward; only steer left when stuck ON THE GROUND
        dir_x = 1 if ddx > 8 else 0
        if self.stuck > 40 and on_ground and ddx < -8:
            dir_x = -1
        move = RIGHT if dir_x > 0 else (LEFT if dir_x < 0 else IDLE)

        def tile(px, py):
            if 0 <= px < W and 0 <= py < H:
                return g[py, px]
            return SOLID

        wall_ahead = dir_x != 0 and (
            tile(cx + dir_x, cy) in (SOLID, SPIKE) or
            tile(cx + dir_x, cy - 1) in (SOLID, SPIKE))
        tall_wall = dir_x != 0 and (
            tile(cx + dir_x, cy - 2) in (SOLID, SPIKE) and
            tile(cx + dir_x, cy - 3) in (SOLID, SPIKE))
        gap_ahead = dir_x != 0 and \
            tile(cx + dir_x, cy + 1) == EMPTY and tile(cx + dir_x, cy + 2) == EMPTY

        jump = 0
        dash = 0
        bash = 0
        aim = 0

        # --- climb mode: tall wall ahead — pulse jumps to hop up next to it,
        # then the airborne wall rule chains wall jumps up the face
        tall = wall_ahead and tall_wall
        if on_ground:
            if tall:
                self.climb_mode = True
            elif not (dir_x != 0 and wall_ahead):
                self.climb_mode = False

        # --- jump requests. The env consumes jump on the RISING EDGE only,
        # so every rule below must PULSE (one press per cooldown), never hold.
        want = None
        if on_ground:
            if tall or wall_ahead or gap_ahead:
                want = "hop"
            elif ddy < -48 and abs(ddx) < 140:
                want = "hop"
            elif self.stuck > 24:
                want = "unstuck"
        if (not on_ground) and wall_dir != 0 and (self.climb_mode or ddy < -10):
            want = "climb"
        if (not on_ground) and self.stuck > 30 and can_djump and not self.dj_used:
            want = "djump"
        if (not on_ground) and vy >= -50 and can_djump and not self.dj_used \
                and ddy < -70:
            want = "djump"
        cd = {"hop": 5, "climb": 7, "unstuck": 4, "djump": 60}[want] \
            if want else 0
        if on_ground and self.jump_cd > 10:
            self.jump_cd = 10        # landing resets any long cooldown
        self.jump_cd = max(0, self.jump_cd - 1)
        if want is not None and self.jump_cd <= 0:
            jump = 1
            self.jump_cd = cd
            if want == "djump":
                self.dj_used = True

        # --- dash toward a FAR waypoint with a clear corridor (pulse).
        # Conservative: 3-height corridor check, never into a wall, never
        # when the waypoint is near (dash would overshoot it).
        final_cell = self.idx >= len(self.path) - 1
        flee = False
        if chaser_x is not None and chaser_y is not None:
            ch_d = np.hypot(chaser_x - x, chaser_y - y)
            # chaser closing in from behind/below: run+burst away
            flee = ch_d < 320 and chaser_x < x + 40
        if can_dash and dash_t <= 0 and not self.dash_pending and not wall_ahead:
            clear = True
            for k in range(1, 6):
                for h in (-1, 0, 1):
                    if tile(cx + dir_x * k, cy + h) in (SOLID, SPIKE):
                        clear = False
                        break
                if not clear:
                    break
            # portal commit: on the final cell, dash in from up to 260 px;
            # flee: burst toward the next waypoint when "pursued" (dash must
            # land at/near the target, never overshoot it)
            if clear and (ddx > 200 or (final_cell and -80 < ddx < 260) or
                          (flee and 120 < ddx < 380)):
                self.dash_pending = True
        if self.dash_pending:
            dash = 1
            self.dash_pending = False
            move = RIGHT
            if ddy < -160:
                move = UP_RIGHT if dir_x >= 0 else UP_LEFT
            elif ddx < -150:
                move = LEFT
            else:
                move = RIGHT

        # --- bash: nearest orb in range whose launch pushes us toward target
        if bash_cd <= 0 and not self.bash_pending:
            best = None
            best_d = p.bash_range + 1.0
            for k in range(min(len(arena.orbs), 8)):
                ox, oy = arena.orbs[k]
                d = np.hypot(ox - x, oy - y)
                if d <= p.bash_range and d < best_d:
                    best_d = d
                    best = (ox, oy)
            if best is not None:
                ox, oy = best
                best_aim = -1
                best_score = -1.0
                for a in range(8):
                    score = (AIM_DX[a] * (tx - ox) + AIM_DY[a] * (ty - oy)) / 160.0
                    if score > best_score:
                        best_score = score
                        best_aim = a
                if best_score > 0.6:              # orb on the way to target
                    aim = best_aim
                    self.bash_pending = True
        if self.bash_pending:
            bash = 1
            self.bash_pending = False

        return np.array([move, jump, dash, bash, aim], dtype=np.int32)


class ScriptedChaser:
    """Simple pursuer: run right, jump obstacles, dash when far. Used as the
    demo opponent so BC rollouts contain real chaser pressure."""

    def __init__(self, p, ep):
        self.p = p
        self.ep = ep
        self.jump_cd = 0
        self.dash_pending = False
        self.stuck = 0
        self.last_cx = -1

    def reset(self, arena):
        self.jump_cd = 0
        self.dash_pending = False
        self.stuck = 0
        self.last_cx = -1

    def act(self, arena, x, y, vx, vy, on_ground, wall_dir, can_dash, dash_t,
            facing, evader_x) -> np.ndarray:
        """[move, jump, dash]"""
        p = self.p
        t = float(p.tile)
        W, H = p.arena_w, p.arena_h
        g = arena.grid
        cx = int(x / t)
        cy = int(y / t)
        ddx = evader_x - x
        dir_x = 1 if ddx > 8 else 0
        move = RIGHT if dir_x > 0 else IDLE

        def tile(px, py):
            if 0 <= px < W and 0 <= py < H:
                return g[py, px]
            return SOLID

        wall_ahead = dir_x != 0 and (
            tile(cx + dir_x, cy) in (SOLID, SPIKE) or
            tile(cx + dir_x, cy - 1) in (SOLID, SPIKE))
        gap_ahead = dir_x != 0 and \
            tile(cx + dir_x, cy + 1) == EMPTY and tile(cx + dir_x, cy + 2) == EMPTY

        jump = 0
        dash = 0
        want = None
        if on_ground:
            if wall_ahead:
                want = "hop"
            elif self.stuck > 40:
                want = "unstuck"
        if cx == self.last_cx:
            self.stuck += 1
        else:
            self.stuck = 0
            self.last_cx = cx
        cd = {"hop": 5, "unstuck": 4}.get(want, 0)
        self.jump_cd = max(0, self.jump_cd - 1)
        if want is not None and self.jump_cd <= 0:
            jump = 1
            self.jump_cd = cd
        # trailing pressure: dash only when the evader is FAR ahead
        if can_dash and dash_t <= 0 and not self.dash_pending and ddx > 420:
            clear = True
            for k in range(1, 6):
                for h in (-1, 0, 1):
                    if tile(cx + dir_x * k, cy + h) in (SOLID, SPIKE):
                        clear = False
                        break
                if not clear:
                    break
            if clear:
                self.dash_pending = True
        if self.dash_pending:
            dash = 1
            self.dash_pending = False
            move = RIGHT
        return np.array([move, jump, dash], dtype=np.int32)


def collect_demos(env, scripted, n_episodes: int, max_steps: int,
                  require_escape: bool = True, seed: int = 0,
                  fake_chaser_dx: float = 0.0):
    """Step `env` (role='evader') with the scripted controller and collect
    (obs, act) pairs from escaping episodes. The env must already have its
    opponent configured (ghost, random, frozen policy, or scripted chaser).

    fake_chaser_dx: if set, the evader's flee logic sees a chaser hovering
    `dx` px behind it (teaches dash-away without real catch risk)."""
    N = env.n_envs
    obs_buf, act_buf = [], []
    n_esc = 0
    n_done = 0
    total = 0
    total_waves = 0

    obs = env.reset()
    ep_steps = np.zeros(N, dtype=np.int32)
    ep_act = [[] for _ in range(N)]
    ep_obs = [[] for _ in range(N)]
    pending = [None] * N

    def flush(i):
        nonlocal n_esc, n_done, total
        if require_escape and pending[i]["outcome"] != "escaped":
            ep_obs[i], ep_act[i] = [], []
            return
        obs_buf.append(np.stack(ep_obs[i]))
        act_buf.append(np.stack(ep_act[i]))
        total += len(ep_act[i])
        n_esc += 1
        n_done += 1
        ep_obs[i], ep_act[i] = [], []

    while n_done < n_episodes and total_waves < 30000:
        acts = []
        for i in range(N):
            ar = env.arenas[i]
            sp = scripted[i]
            if len(ep_obs[i]) == 0:
                sp.reset(ar)
            phys = env.phys
            if fake_chaser_dx:
                # teach flee-dash behavior without real catch risk: a chaser
                # hovering just behind the evader triggers the dash-away rule
                cx = phys.x[i] + fake_chaser_dx
                cy = phys.y[i]
            else:
                cx = phys.x[N + i]
                cy = phys.y[N + i]
            acts.append(sp.act(ar, phys.x[i], phys.y[i], phys.vx[i], phys.vy[i],
                               phys.on_ground[i], phys.wall_dir[i],
                               phys.can_djump[i], phys.can_dash[i],
                               phys.dash_t[i], phys.bash_cd[i], phys.facing[i],
                               chaser_x=cx, chaser_y=cy))
        acts = np.stack(acts)
        total_waves += 1
        obs, _, done, infos = env.step(acts)
        for i in range(N):
            ep_obs[i].append(obs[i])
            ep_act[i].append(acts[i])
            ep_steps[i] += 1
            if done[i] or ep_steps[i] >= max_steps:
                if done[i] and "episode" in infos[i]:
                    pending[i] = infos[i]
                    flush(i)
                else:
                    ep_obs[i], ep_act[i] = [], []
                ep_steps[i] = 0
    return obs_buf, act_buf, {"episodes": n_done, "pairs": total,
                              "waves": total_waves}


if __name__ == "__main__":
    # quick self-test: scripted traversal escape rate on fixed arenas
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
    from oripark.config import MoveParams, EnvParams
    from oripark.arena import ArenaGenerator
    from oripark.env import OriArenaVecEnv

    mp, ep = MoveParams(), EnvParams()
    rng = np.random.default_rng(1)
    gen = ArenaGenerator(mp, rng, chaser_frac=ep.chaser_spawn_frac)

    class Fixed:
        def __init__(self, arenas):
            self.arenas = arenas
        def sample(self, n):
            return [self.arenas[k % len(self.arenas)] for k in range(n)]

    arenas = [gen.generate(np.full(6, 0.5), seed=1000 + k) for k in range(24)]
    env = OriArenaVecEnv("evader", 16, Fixed(arenas), mp, ep, opponent=None, seed=0)
    scripts = [ScriptedEvader(mp, ep) for _ in range(16)]
    _, _, stats = collect_demos(env, scripts, n_episodes=96, max_steps=ep.max_steps)
    env.close()
    print(f"scripted traversal: {stats['episodes']} escaped episodes, "
          f"{stats['pairs']} (obs,act) pairs collected "
          f"({stats['waves']} waves)")
