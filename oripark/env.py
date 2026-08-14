"""Vectorized gymnasium environment for the Ori tag game.

One `OriArenaVecEnv` simulates N arenas, each containing two agents
sharing Ori-style physics:
  * evader  — full Ori kit (run, jump, double jump, wall jump, dash, bash)
  * chaser  — same kit minus bash; its goal is to catch the evader.

Implements stable-baselines3's `VecEnv` protocol directly (no subprocesses)
because the physics is numpy-vectorized across the batch. Runs in "evader"
or "chaser" role: the learner's actions come from SB3, the opponent's from
a frozen policy (or random).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from .config import EnvParams, MoveParams
from .physics import OriPhysics, SOLID, SPIKE, ORB, PORTAL

EV_ACT = spaces.MultiDiscrete([9, 2, 2, 2, 8])   # move, jump, dash, bash, aim
CH_ACT = spaces.MultiDiscrete([9, 2, 2])         # move, jump, dash


def _scalars(phys: OriPhysics, si, oi, p: MoveParams, ep: EnvParams,
             t: np.ndarray, extra: dict | None = None) -> np.ndarray:
    """Normalized scalar features for a role view. Returns (N, K)."""
    t_ = float(p.tile)
    W, H = p.arena_w, p.arena_h
    xs, ys = phys.x[si], phys.y[si]
    oxs, oys = phys.x[oi], phys.y[oi]
    ovx, ovy = phys.vx[oi], phys.vy[oi]

    orb_dx = np.zeros_like(xs); orb_dy = np.zeros_like(xs); orb_d = np.full_like(xs, 9.0)
    orbs_s = phys.orbs[si]                    # (N, 8, 2)
    for k in range(8):
        ox = orbs_s[:, k, 0]; oy = orbs_s[:, k, 1]
        ok = ~np.isnan(ox)
        d = np.hypot(ox - xs, oy - ys)
        closer = ok & (d < orb_d)
        orb_dx = np.where(closer, (ox - xs) / W, orb_dx)
        orb_dy = np.where(closer, (oy - ys) / H, orb_dy)
        orb_d = np.where(closer, d, orb_d)

    ptx = phys.portals[si, 0]; pty = phys.portals[si, 1]
    portal_d = np.hypot(ptx - xs, pty - ys)

    feats = [
        xs / (W * t_), ys / (H * t_),
        phys.vx[si] / p.max_run, phys.vy[si] / 1200.0,
        phys.on_ground[si].astype(np.float32),
        phys.wall_dir[si].astype(np.float32),
        phys.coyote[si] / p.coyote_time, phys.buffer[si] / p.jump_buffer,
        phys.can_djump[si].astype(np.float32), phys.can_dash[si].astype(np.float32),
        phys.dash_t[si] / p.dash_time, phys.bash_cd[si] / p.bash_cooldown,
        (oxs - xs) / W, (oys - ys) / H,
        ovx / p.max_run, ovy / 1200.0,
        orb_dx, orb_dy, np.clip(orb_d / p.bash_range, 0, 9),
        (ptx - xs) / W, (pty - ys) / H, portal_d / (W * t_),
        t / ep.max_steps, phys.facing[si],
    ]
    if extra:
        feats += [extra[k] for k in extra]
    return np.stack(feats, axis=1).astype(np.float32)


def _patch(phys: OriPhysics, si, oi, ep: EnvParams) -> np.ndarray:
    """Local patch_w x patch_h tile patch centered on self; opponent marked 5."""
    p = phys.p
    t = float(p.tile)
    N = len(np.arange(phys.n)[si])
    px, py = phys.x[si], phys.y[si]
    cx = np.clip((px / t).astype(np.int64), 0, p.arena_w - 1)
    cy = np.clip((py / t).astype(np.int64), 0, p.arena_h - 1)
    ox = np.clip((phys.x[oi] / t).astype(np.int64), 0, p.arena_w - 1)
    oy = np.clip((phys.y[oi] / t).astype(np.int64), 0, p.arena_h - 1)

    half_w = ep.patch_w // 2
    half_h = ep.patch_h // 2
    xs = cx[:, None] + np.arange(-half_w, half_w + 1)[None, :]
    ys = cy[:, None] + np.arange(-half_h, half_h + 1)[None, :]
    xs_c = np.clip(xs, 0, p.arena_w - 1)
    ys_c = np.clip(ys, 0, p.arena_h - 1)
    flat = ys_c[..., None] * p.arena_w + xs_c[:, None, :]
    g = phys.grids[si].reshape(N, -1)
    vals = g[np.arange(N)[:, None, None], flat]

    remap = np.zeros(6, dtype=np.float32)
    remap[SOLID] = 1.0
    remap[SPIKE] = 2.0
    remap[ORB] = 3.0
    remap[PORTAL] = 4.0
    out = remap[vals]
    dr = oy - (cy - half_h)
    dc = ox - (cx - half_w)
    ok = (dr >= 0) & (dr < ep.patch_h) & (dc >= 0) & (dc < ep.patch_w)
    r = np.clip(dr, 0, ep.patch_h - 1)
    c = np.clip(dc, 0, ep.patch_w - 1)
    out[np.arange(N), r, c] = np.where(ok, 5.0, out[np.arange(N), r, c])
    return out.reshape(N, -1)


class OriArenaVecEnv(VecEnv):
    """Vectorized tag environment. role: 'evader' or 'chaser' (learner side)."""

    metadata = {"render.modes": []}

    def __init__(self, role: str, n_envs: int, sampler, move: MoveParams,
                 ep: EnvParams, opponent=None, seed: int = 0):
        assert role in ("evader", "chaser")
        self.role = role
        self.n_envs = n_envs
        self.sampler = sampler
        self.p = move
        self.ep = ep
        self.opponent = opponent
        self.rng = np.random.default_rng(seed)

        self.obs_dim = 24 + ep.patch_w * ep.patch_h + (4 if role == "chaser" else 0)
        obs_space = spaces.Box(low=-5.0, high=5.0, shape=(self.obs_dim,), dtype=np.float32)
        act_space = EV_ACT if role == "evader" else CH_ACT
        super().__init__(n_envs, obs_space, act_space)

        self.arenas = [None] * n_envs
        grids = np.zeros((2 * n_envs, move.arena_h, move.arena_w), dtype=np.int32)
        self.phys = OriPhysics(2 * n_envs, grids, move, self.rng)
        self.t = np.zeros(n_envs, dtype=np.float32)

        # agility counters per env: dash, walljump, djump, bash, airtime, maxspeed
        self.ev_agg = np.zeros((n_envs, 6), dtype=np.int32)
        self.ch_agg = np.zeros((n_envs, 6), dtype=np.int32)
        self.prev_full = np.zeros((2 * n_envs, 5), dtype=np.int32)
        self.last_infos = [{} for _ in range(n_envs)]
        self.pending = []            # episode summaries awaiting drain

        self._fresh_all()

    # ------------------------------------------------------------- VecEnv API
    def reset(self):
        self._fresh_all()
        return self._obs()

    def step_async(self, actions):
        self._pending_actions = np.asarray(actions).reshape(self.n_envs, -1).astype(np.int32)

    def step_wait(self):
        return self._step(self._pending_actions)

    def close(self):
        pass

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in range(self.n_envs)]

    def get_attr(self, attr_name, indices=None):
        idx = np.arange(self.n_envs) if indices is None else indices
        return [getattr(self, attr_name) for _ in idx]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self, method_name)(*method_args, **method_kwargs) for _ in range(self.n_envs)]

    # ----------------------------------------------------------------- internals
    def _fresh_all(self):
        arenas = self.sampler.sample(self.n_envs)
        for i, a in enumerate(arenas):
            self._load_arena(i, a)
        self.t[:] = 0
        self.ev_agg[:] = 0
        self.ch_agg[:] = 0
        self.prev_full[:] = 0

    def _load_arena(self, i: int, arena):
        self.arenas[i] = arena
        g = arena.grid
        ei, ci = i, self.n_envs + i          # contiguous layout: evaders 0..N-1, chasers N..2N-1
        self.phys.grids[ei] = g
        self.phys.grids[ci] = g
        self.phys.orbs[ei] = self._orbs_for(arena)
        self.phys.orbs[ci] = self.phys.orbs[ei]
        self.phys.portals[ei] = arena.portal
        self.phys.portals[ci] = arena.portal
        self.phys.place(np.array([arena.ev_spawn[0], arena.ch_spawn[0]]),
                        np.array([arena.ev_spawn[1], arena.ch_spawn[1]]),
                        ground=True, idx=[ei, ci])

    @staticmethod
    def _orbs_for(arena) -> np.ndarray:
        out = np.full((8, 2), np.nan, dtype=np.float32)
        for k, (ox, oy) in enumerate(arena.orbs[:8]):
            out[k, 0], out[k, 1] = ox, oy
        return out

    def _obs(self) -> np.ndarray:
        N = self.n_envs
        ev, ch = slice(0, N), slice(N, 2 * N)
        if self.role == "evader":
            scal = _scalars(self.phys, ev, ch, self.p, self.ep, self.t)
        else:
            extra = {
                "ev_on_ground": self.phys.on_ground[ev].astype(np.float32),
                "ev_wall_dir": self.phys.wall_dir[ev].astype(np.float32),
                "ev_can_djump": self.phys.can_djump[ev].astype(np.float32),
                "ev_can_dash": self.phys.can_dash[ev].astype(np.float32),
            }
            scal = _scalars(self.phys, ch, ev, self.p, self.ep, self.t, extra=extra)
        pat = _patch(self.phys, ev if self.role == "evader" else ch,
                     ch if self.role == "evader" else ev, self.ep)
        return np.concatenate([scal, pat], axis=1).astype(np.float32)

    def _obs_both(self) -> dict:
        N = self.n_envs
        ev, ch = slice(0, N), slice(N, 2 * N)
        ev_scal = _scalars(self.phys, ev, ch, self.p, self.ep, self.t)
        ch_scal = _scalars(self.phys, ch, ev, self.p, self.ep, self.t, extra={
            "ev_on_ground": self.phys.on_ground[ev].astype(np.float32),
            "ev_wall_dir": self.phys.wall_dir[ev].astype(np.float32),
            "ev_can_djump": self.phys.can_djump[ev].astype(np.float32),
            "ev_can_dash": self.phys.can_dash[ev].astype(np.float32),
        })
        ev_pat = _patch(self.phys, ev, ch, self.ep)
        ch_pat = _patch(self.phys, ch, ev, self.ep)
        return {
            "evader": np.concatenate([ev_scal, ev_pat], axis=1).astype(np.float32),
            "chaser": np.concatenate([ch_scal, ch_pat], axis=1).astype(np.float32),
        }

    def _step(self, learner_actions: np.ndarray):
        N = self.n_envs
        p = self.p
        ep = self.ep
        ev, ch = slice(0, N), slice(N, 2 * N)

        obs_both = self._obs_both() if self.opponent is not None else None
        if self.opponent is not None:
            opp_obs = obs_both["chaser" if self.role == "evader" else "evader"]
            opp_acts = np.asarray(self.opponent(opp_obs)).reshape(N, -1).astype(np.int32)
        else:
            if self.role == "evader":            # opponent = chaser (3 dims)
                opp_acts = self.rng.integers([9, 2, 2], size=(N, 3)).astype(np.int32)
            else:                                # opponent = evader (5 dims)
                opp_acts = self.rng.integers([9, 2, 2, 2, 8], size=(N, 5)).astype(np.int32)

        full = np.zeros((2 * N, 5), dtype=np.int32)
        if self.role == "evader":
            full[ev] = learner_actions       # learner evader: 5 dims
            full[ch, :3] = opp_acts          # opponent chaser: 3 dims
        else:
            full[ev] = opp_acts              # opponent evader: 5 dims
            full[ch, :3] = learner_actions   # learner chaser: 3 dims

        prev = self.prev_full
        # --- input edges
        jump_p = (full[:, 1] == 1) & (prev[:, 1] == 0)
        jump_r = (full[:, 1] == 0) & (prev[:, 1] == 1)
        dash_p = (full[:, 2] == 1) & (prev[:, 2] == 0)
        bash_p = (full[:, 3] == 1) & (prev[:, 3] == 0)

        # --- pre-step state for usage detection
        ev_dj_b = self.phys.can_djump[ev].copy()
        ch_dj_b = self.phys.can_djump[ch].copy()
        ev_dash_b = self.phys.dash_t[ev].copy()
        ch_dash_b = self.phys.dash_t[ch].copy()
        ev_wg_b = self.phys.wall_grace[ev].copy()
        ch_wg_b = self.phys.wall_grace[ch].copy()
        ev_bc_b = self.phys.bash_cd[ev].copy()
        dist_before = np.hypot(self.phys.x[ev] - self.phys.x[ch], self.phys.y[ev] - self.phys.y[ch])
        portal_before = np.hypot(self.phys.portals[ev, 0] - self.phys.x[ev],
                                 self.phys.portals[ev, 1] - self.phys.y[ev])

        self.phys.press_jump(jump_p)
        self.phys.release_jump(jump_r)
        self.phys.press_dash(dash_p)
        self.phys.press_bash(bash_p)
        self.phys.step(full)
        self.prev_full[:] = full
        self.t += 1

        # --- usage this step (state transitions)
        ev_dash_u = (self.phys.dash_t[ev] > 0) & (ev_dash_b <= 0)
        ch_dash_u = (self.phys.dash_t[ch] > 0) & (ch_dash_b <= 0)
        ev_wj = (self.phys.wall_grace[ev] > 0) & (ev_wg_b <= 0)
        ch_wj = (self.phys.wall_grace[ch] > 0) & (ch_wg_b <= 0)
        ev_dj = ev_dj_b & (~self.phys.can_djump[ev])
        ch_dj = ch_dj_b & (~self.phys.can_djump[ch])
        ev_bs = (self.phys.bash_cd[ev] > 0) & (ev_bc_b <= 0)
        ev_used = ev_dash_u | ev_wj | ev_dj | ev_bs
        ch_used = ch_dash_u | ch_wj | ch_dj

        # --- outcomes
        caught = self.phys.aabb_overlap(self.phys, ev, ch)
        died_ev = self.phys.died[ev] & ~caught
        died_ch = self.phys.died[ch] & ~caught
        esc = self.phys.escaped[ev] & ~caught & ~died_ev & ~died_ch
        timeout = self.t >= ep.max_steps
        done = caught | died_ev | died_ch | esc | timeout

        dist_after = np.hypot(self.phys.x[ev] - self.phys.x[ch], self.phys.y[ev] - self.phys.y[ch])
        d_gain = dist_after - dist_before
        portal_after = np.hypot(self.phys.portals[ev, 0] - self.phys.x[ev],
                                self.phys.portals[ev, 1] - self.phys.y[ev])
        p_gain = portal_before - portal_after

        # --- rewards
        prox = np.clip((200.0 - dist_after) / 200.0, 0, 1)
        r_ev = np.full(N, ep.r_time, dtype=np.float32)
        r_ch = np.full(N, -ep.r_time, dtype=np.float32)
        r_ev += ep.r_dist_gain * np.clip(d_gain / 50.0, -1, 1)
        r_ev += ep.r_portal_progress * np.clip(p_gain / 100.0, -1, 1)
        r_ch += ep.r_dist_gain * np.clip(-d_gain / 50.0, -1, 1)
        r_ev -= ep.r_proximity * prox
        r_ch += ep.r_proximity * prox
        r_ev += ep.r_agility * ev_used.astype(np.float32)
        r_ch += ep.r_agility * ch_used.astype(np.float32)
        r_ev = np.where(caught, ep.r_caught, r_ev)
        r_ch = np.where(caught, -ep.r_caught, r_ch)
        r_ev = np.where(esc, ep.r_portal, r_ev)
        r_ch = np.where(esc, -ep.r_portal, r_ch)
        r_ev = np.where(died_ev, ep.r_hazard, r_ev)
        r_ch = np.where(died_ev, 0.5, r_ch)
        r_ev = np.where(died_ch, 0.5, r_ev)
        r_ch = np.where(died_ch, ep.r_hazard, r_ch)
        r_ev = np.where(timeout, ep.r_timeout, r_ev)
        r_ch = np.where(timeout, -ep.r_timeout, r_ch)

        # --- agility counters
        self.ev_agg[:, 0] += ev_dash_u.astype(np.int32)
        self.ev_agg[:, 1] += ev_wj.astype(np.int32)
        self.ev_agg[:, 2] += ev_dj.astype(np.int32)
        self.ev_agg[:, 3] += ev_bs.astype(np.int32)
        self.ev_agg[:, 4] += (~self.phys.on_ground[ev]).astype(np.int32)
        self.ev_agg[:, 5] = np.maximum(self.ev_agg[:, 5], np.abs(self.phys.vx[ev]).astype(np.int32))
        self.ch_agg[:, 0] += ch_dash_u.astype(np.int32)
        self.ch_agg[:, 1] += ch_wj.astype(np.int32)
        self.ch_agg[:, 2] += ch_dj.astype(np.int32)
        self.ch_agg[:, 4] += (~self.phys.on_ground[ch]).astype(np.int32)
        self.ch_agg[:, 5] = np.maximum(self.ch_agg[:, 5], np.abs(self.phys.vx[ch]).astype(np.int32))

        # --- episode bookkeeping
        infos = [{} for _ in range(N)]
        for i in range(N):
            if done[i]:
                out = ("caught" if caught[i] else "escaped" if esc[i] else
                       "ev_hazard" if died_ev[i] else "ch_hazard" if died_ch[i] else "timeout")
                ev_win = out in ("escaped", "ch_hazard", "timeout")
                infos[i] = {
                    "episode": {"r": float(r_ev[i] if self.role == "evader" else r_ch[i]),
                                "l": int(self.t[i])},
                    "outcome": out,
                    "ev_win": bool(ev_win),
                    "arena_params": self.arenas[i].params.copy(),
                    "ev_agility": self.ev_agg[i].copy(),
                    "ch_agility": self.ch_agg[i].copy(),
                }
                self._reset_one(i)
                self.pending.append(infos[i])
        self.last_infos = infos
        obs = self._obs()
        r = r_ev if self.role == "evader" else r_ch
        return obs, r.astype(np.float32), done, infos

    def drain(self) -> list:
        """Return and clear accumulated episode summaries."""
        out, self.pending = self.pending, []
        return out

    def _reset_one(self, i: int):
        arena = self.sampler.sample(1)[0]
        self._load_arena(i, arena)
        self.t[i] = 0
        self.ev_agg[i] = 0
        self.ch_agg[i] = 0
        self.prev_full[i] = 0
        self.prev_full[self.n_envs + i] = 0
