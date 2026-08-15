"""Replay recording + rendering for the Ori tag game.

Runs an episode with fixed policies (deterministic), records positions and
actions, then renders headless-pygame frames into a GIF (or ASCII frames
for the terminal). `--mode before` uses the untrained pool-0 snapshot so
you can watch a random agent; `--mode after` uses the trained policy.
"""
from __future__ import annotations

import os

import numpy as np

from .arena import Arena
from .config import EnvParams, MoveParams
from .env import OriArenaVecEnv
from .selfplay import frozen_policy


class SingleArenaSampler:
    def __init__(self, arena: Arena):
        self.arena = arena

    def sample(self, n: int):
        return [self.arena for _ in range(n)]

    def mean_params(self) -> np.ndarray:
        return self.arena.params


class ReplayRecorder:
    def __init__(self, arena: Arena, ev_pol, ch_pol, mp: MoveParams, ep: EnvParams,
                 deterministic: bool = True, seed: int = 0):
        self.mp, self.ep = mp, ep
        self.det = deterministic
        sampler = SingleArenaSampler(arena)
        self.env = OriArenaVecEnv("evader", 1, sampler, mp, ep,
                                  opponent=frozen_policy(ch_pol, deterministic), seed=seed)
        self.ev_pol = ev_pol

    def run(self, max_steps: int = 1500) -> tuple[list, str]:
        obs = self.env.reset()
        frames = []
        outcome = "timeout"
        ph = self.env.phys
        for t in range(max_steps):
            a = self.ev_pol.predict(obs, deterministic=self.det)[0]
            obs, _, done, infos = self.env.step(a)
            frames.append({
                "t": t,
                "ev": (float(ph.x[0]), float(ph.y[0])),
                "ch": (float(ph.x[1]), float(ph.y[1])),
                "ev_act": self.env.prev_full[0].copy(),
                "ch_act": self.env.prev_full[1].copy(),
                "ev_agg": self.env.ev_agg[0].copy(),
                "ch_agg": self.env.ch_agg[0].copy(),
            })
            if done[0]:
                outcome = infos[0].get("outcome", "timeout")
                break
        return frames, outcome

    def close(self):
        self.env.close()


# ------------------------------------------------------------------ rendering
# Painterly Ori-style scene: warm god-light sky fading to deep teal, two
# parallax foliage layers, translucent god rays, grass-topped soil terrain
# with per-tile texture, glowing orbs/portal, luminous spirit agents, drifting
# fireflies, soft vignette, and a subtle HUD.

_PAL = {
    "sky_top": (252, 224, 158), "sky_mid": (138, 200, 190),
    "sky_low": (38, 96, 108), "sky_bot": (10, 26, 40),
    "foliage_far": (12, 30, 46), "foliage_near": (8, 20, 32),
    "ray": (255, 236, 190),
    "soil": (84, 66, 55), "soil_dark": (56, 44, 38),
    "grass": (88, 150, 88), "grass_hi": (150, 205, 130),
    "spike": (232, 40, 120), "spike_hi": (255, 130, 180),
    "orb": (255, 214, 110), "orb_hi": (255, 244, 200),
    "portal": (120, 230, 255), "portal_hi": (235, 252, 255),
    "evader": (238, 250, 255), "evader_glow": (170, 228, 255),
    "chaser": (92, 22, 34), "chaser_glow": (255, 70, 96),
    "firefly": (255, 240, 190),
}


def _hash2(x: int, y: int, salt: int = 0) -> float:
    """Deterministic [0,1) hash for per-tile variation."""
    h = (x * 374761393 + y * 668265263 + salt * 1274126177) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFFFF) / 0xFFFFFFFF


def _glow_sprite(radius: int, color, peak: int = 200) -> "pygame.Surface":
    """Radial alpha-gradient glow sprite (per-pixel alpha)."""
    import pygame
    r = max(2, int(radius))
    surf = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
    steps = 22
    for k in range(steps, 0, -1):
        f = k / steps
        a = int(peak * f * f)
        pygame.draw.circle(surf, (*color, a), (r, r), int(r * f))
    return surf


class _Scene:
    """Precomputed painterly scene for one arena; renders frames cheaply."""

    def __init__(self, arena, mp: MoveParams, seed: int = 0):
        import pygame
        self.mp = mp
        self.t = t = float(mp.tile)
        self.W = mp.arena_w * mp.tile
        self.H = mp.arena_h * mp.tile
        self.arena = arena
        # glow sprites (needed by terrain drawing)
        self.g_ev = _glow_sprite(26, _PAL["evader_glow"])
        self.g_ch = _glow_sprite(24, _PAL["chaser_glow"])
        self.g_orb = _glow_sprite(30, _PAL["orb"])
        self.g_portal = _glow_sprite(44, _PAL["portal"])
        self.g_fly = _glow_sprite(7, _PAL["firefly"])
        self.static = pygame.Surface((self.W, self.H))
        self._draw_background()
        self._draw_terrain(arena.grid)
        # fireflies: seeded positions, phase for drift + pulse
        rng = np.random.default_rng(seed * 7919 + 13)
        n = 46
        self.ff = np.stack([
            rng.uniform(0, self.W, n),
            rng.uniform(self.H * 0.08, self.H * 0.9, n),
            rng.uniform(0, 6.28, n),       # phase
            rng.uniform(0.6, 1.4, n),       # speed
        ], axis=1)
        # vignette overlay
        self.vig = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        self._draw_vignette()
        # hud font
        self.font = pygame.font.SysFont("dejavusansmono", 22)

    # ------------------------------------------------------------ background
    def _draw_background(self):
        import pygame
        W, H, P = self.W, self.H, _PAL
        # --- sky gradient (warm light above, deep teal below)
        stops = [(0.0, P["sky_top"]), (0.30, P["sky_mid"]),
                 (0.62, P["sky_low"]), (1.0, P["sky_bot"])]
        for y in range(H):
            f = y / H
            i = 1
            while f > stops[i][0]:
                i += 1
            (y0, c0), (y1, c1) = stops[i - 1], stops[i]
            k = 0.0 if y1 == y0 else (f - y0) / (y1 - y0)
            col = tuple(int(c0[j] + (c1[j] - c0[j]) * k) for j in range(3))
            pygame.draw.line(self.static, col, (0, y), (W, y))
        # --- god rays: soft warm wedges from the top
        rays = pygame.Surface((W, H), pygame.SRCALPHA)
        cx = W * 0.5
        for i, (ox, span, a) in enumerate([(-0.35, 0.16, 26), (0.05, 0.22, 34),
                                           (0.45, 0.12, 22), (0.25, 0.09, 18)]):
            px = cx + ox * W
            base = int(40 + (i % 3) * 90)
            pygame.draw.polygon(rays, (*P["ray"], a),
                                [(px - span * W * 0.5, 0), (px + span * W * 0.5, 0),
                                 (px + span * W * 1.6, H), (px - span * W * 1.6, H)])
        self.static.blit(rays, (0, 0))
        # --- parallax foliage: two dark painterly layers
        for layer, (col, nblob, salt) in enumerate([
                (P["foliage_far"], 26, 11), (P["foliage_near"], 18, 29)]):
            rng = np.random.default_rng(1000 * (layer + 1) + salt)
            lay = pygame.Surface((W, H), pygame.SRCALPHA)
            for _ in range(nblob):
                bx = rng.uniform(-50, W + 50)
                by = rng.uniform(-30, H * (0.55 if layer == 0 else 0.4))
                br = rng.uniform(60, 160)
                for _ in range(7):
                    pygame.draw.circle(lay, (*col, 255),
                                       (int(bx + rng.uniform(-br, br) * 0.4),
                                        int(by + rng.uniform(-br, br) * 0.4)),
                                       int(rng.uniform(br * 0.35, br * 0.8)))
            self.static.blit(lay, (0, 0))

    def _draw_vignette(self):
        import pygame
        W, H = self.W, self.H
        m = min(W, H)
        for y in range(0, H, 4):
            for x in range(0, W, 4):
                dx = min(x, W - x) / (W / 2)
                dy = min(y, H - y) / (H / 2)
                d = min(1.0, (dx * dx + dy * dy))
                a = int(70 * max(0.0, d - 0.55) / 0.45)
                if a > 0:
                    pygame.draw.rect(self.vig, (2, 6, 12, a), (x, y, 4, 4))

    # --------------------------------------------------------------- terrain
    def _draw_terrain(self, grid: np.ndarray):
        import pygame
        t, P = int(self.t), _PAL
        H, W = grid.shape
        for y in range(H):
            for x in range(W):
                v = int(grid[y, x])
                px, py = x * t, y * t
                if v == 1:
                    self._tile_solid(px, py, x, y, grid)
                elif v == 2:
                    self._tile_spike(px, py, x, y)
                elif v == 3:
                    self.static.blit(self.g_orb, (px + t // 2 - 30, py + t // 2 - 30))
                    pygame.draw.circle(self.static, P["orb"], (px + t // 2, py + t // 2), 14)
                    pygame.draw.circle(self.static, P["orb_hi"], (px + t // 2 - 4, py + t // 2 - 4), 6)
                elif v == 4:
                    self._tile_portal(px, py, x, y)

    def _tile_solid(self, px, py, x, y, grid):
        import pygame
        t, P = int(self.t), _PAL
        h = _hash2(x, y, 7)
        base = tuple(int(c * (0.90 + 0.18 * h)) for c in P["soil"])
        pygame.draw.rect(self.static, base, (px, py, t, t))
        # grass cap if the tile above is not solid
        above_empty = y == 0 or int(grid[y - 1, x]) != 1
        if above_empty:
            pygame.draw.rect(self.static, P["grass"], (px, py, t, 7))
            pygame.draw.rect(self.static, P["grass_hi"], (px, py, t, 2))
            if h > 0.55:                       # occasional grass tuft
                tuft = pygame.draw.polygon(
                    self.static, P["grass"],
                    [(px + 4 + 18 * (h - 0.55), py), (px + 6 + 18 * (h - 0.55), py - 5),
                     (px + 9 + 18 * (h - 0.55), py)])
        # warm light on the top-left edge, cool shadow bottom-right
        pygame.draw.rect(self.static, tuple(int(c * 1.25) for c in base), (px, py, t, 2))
        pygame.draw.rect(self.static, tuple(int(c * 1.12) for c in base), (px, py, 2, t))
        pygame.draw.rect(self.static, P["soil_dark"], (px, py + t - 3, t, 3))
        pygame.draw.rect(self.static, P["soil_dark"], (px + t - 2, py, 2, t))
        # texture specks
        if h > 0.3:
            for k in range(3):
                sx = px + int(_hash2(x, y, k + 1) * (t - 6)) + 2
                sy = py + 9 + int(_hash2(x, y, k + 9) * (t - 16))
                pygame.draw.circle(self.static, P["soil_dark"], (sx, sy), 1)

    def _tile_spike(self, px, py, x, y):
        import pygame
        t, P = int(self.t), _PAL
        self.static.blit(self.g_orb, (px + t // 2 - 24, py + t // 2 - 24))
        for k, (x0, x1, x2, y0) in enumerate([
                (2, 10, 6, 26), (10, 20, 15, 30), (19, 30, 24, 24)]):
            pygame.draw.polygon(self.static, P["spike"],
                                [(px + x0, py + t - 4), (px + x2, py + y0), (px + x1, py + t - 4)])
        pygame.draw.polygon(self.static, P["spike_hi"],
                            [(px + 8, py + t - 8), (px + 14, py + 6), (px + 21, py + t - 8)])

    def _tile_portal(self, px, py, x, y):
        import pygame
        t, P = int(self.t), _PAL
        # rising light beam
        beam = pygame.Surface((t, self.H - py), pygame.SRCALPHA)
        for k in range(10):
            a = int(26 * (1 - k / 10))
            pygame.draw.rect(beam, (*P["portal"], a),
                             (k, 0, t - 2 * k, self.H - py))
        self.static.blit(beam, (px, py))
        self.static.blit(self.g_portal, (px + t // 2 - 44, py - 30))
        pygame.draw.rect(self.static, P["portal"], (px + 2, py - 14, t - 4, t + 14))
        pygame.draw.rect(self.static, P["portal_hi"], (px + 8, py - 14, t - 16, t - 10))

    # ------------------------------------------------------------------ frame
    def render(self, surf, frames, i, label: str):
        import pygame
        P = _PAL
        # trails: ribbon of fading glow sprites (cheap blits)
        for k in range(max(0, i - 22), i):
            f = (i - k) / 23.0
            fr = frames[k]
            for pos, g in ((fr["ev"], self.g_ev), (fr["ch"], self.g_ch)):
                s = g.copy()
                s.set_alpha(int(120 * f))
                surf.blit(s, (int(pos[0]) - 26, int(pos[1]) - 26))
        ev = frames[i]["ev"]
        ch = frames[i]["ch"]
        ex, ey = int(ev[0]), int(ev[1])
        cx, cy = int(ch[0]), int(ch[1])
        # bodies: glow halo + core + facing eye
        surf.blit(self.g_ev, (ex - 26, ey - 26))
        surf.blit(self.g_ch, (cx - 24, cy - 24))
        pygame.draw.circle(surf, P["evader"], (ex, ey), 11)
        pygame.draw.circle(surf, (255, 255, 255), (ex - 3, ey - 3), 5)
        face = frames[i]["ev_act"][0]
        fdx = 6 if face == 2 else -6 if face == 1 else 0
        pygame.draw.circle(surf, (40, 60, 80), (ex + fdx, ey - 3), 2)
        pygame.draw.circle(surf, P["chaser"], (cx, cy), 11)
        pygame.draw.circle(surf, P["chaser_glow"], (cx - 3, cy - 3), 5)
        pygame.draw.circle(surf, (255, 120, 140), (cx + fdx, cy - 3), 2)
        # fireflies (single sprite, alpha pulse, drift with time)
        t = frames[i]["t"]
        for fx, fy, ph, sp in self.ff:
            yy = fy + 10 * np.sin(t * 0.02 * sp + ph)
            xx = fx + 6 * np.sin(t * 0.013 * sp + ph * 1.7)
            a = int(120 + 110 * np.sin(t * 0.05 * sp + ph))
            if a <= 0:
                continue
            self.g_fly.set_alpha(min(255, a))
            surf.blit(self.g_fly, (int(xx) - 7, int(yy) - 7))
        self.g_fly.set_alpha(255)
        # vignette + HUD
        surf.blit(self.vig, (0, 0))
        d = float(np.hypot(ev[0] - ch[0], ev[1] - ch[1]))
        agg = frames[i]["ev_agg"]
        hud = pygame.Surface((self.W, 34), pygame.SRCALPHA)
        hud.fill((8, 16, 28, 150))
        surf.blit(hud, (0, 0))
        txt = (f"{label}   t={frames[i]['t']:4d}   dist={d:5.0f}   "
               f"dash={int(agg[0])} wj={int(agg[1])} dj={int(agg[2])} bash={int(agg[3])}")
        surf.blit(self.font.render(txt, True, (230, 245, 255)), (12, 6))


def _draw_terrain(surf, grid: np.ndarray, mp: MoveParams):
    """Back-compat stub: flat background used by older callers."""
    t = mp.tile
    surf.fill((16, 30, 38))
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            v = int(grid[y, x])
            px, py = x * t, y * t
            if v == 1:
                pygame.draw.rect(surf, (178, 137, 104), (px, py, t, t))
                pygame.draw.rect(surf, (216, 178, 140), (px, py, t, 6))
            elif v == 4:
                pygame.draw.rect(surf, (76, 201, 240), (px, py - 30, t, t + 30))


def _draw_frame(surf, frames, i, label: str, mp: MoveParams):
    """Back-compat stub for callers using the old flat API."""
    import pygame
    ev = frames[i]["ev"]
    ch = frames[i]["ch"]
    pygame.draw.circle(surf, (224, 251, 252), (int(ev[0]), int(ev[1])), 12)
    pygame.draw.circle(surf, (230, 57, 70), (int(ch[0]), int(ch[1])), 12)
    font = pygame.font.SysFont("dejavusansmono", 22)
    surf.blit(font.render(label, True, (230, 245, 255)), (10, 8))


def make_gif(frames: list, arena: Arena, mp: MoveParams, out_path: str,
             label: str = "", scale: float = 0.5, fps: int = 30, step: int = 2) -> str:
    """Render frames headlessly with pygame and assemble a GIF with PIL."""
    import pygame
    from PIL import Image
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    t = mp.tile
    W, H = mp.arena_w * t, mp.arena_h * t
    scene = _Scene(arena, mp, seed=arena.seed if hasattr(arena, "seed") else 0)

    sel = frames[::step]
    imgs = []
    for i, fr in enumerate(sel):
        surf = scene.static.copy()
        scene.render(surf, sel, i, label)
        arr = pygame.image.tostring(surf, "RGB")
        img = Image.frombytes("RGB", (W, H), arr)
        if scale != 1.0:
            img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
        imgs.append(img.convert("P", palette=Image.ADAPTIVE, colors=160))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    pygame.quit()
    return out_path


def ascii_frames(frames: list, arena: Arena, mp: MoveParams, n_frames: int = 24) -> str:
    """Compact ASCII replay for the terminal."""
    t = mp.tile
    sel = frames[::max(1, len(frames) // n_frames)][:n_frames]
    lines = []
    chars = {0: ".", 1: "#", 2: "^", 3: "o", 4: "P"}
    for fr in sel:
        g = arena.grid.copy()
        ex, ey = int(fr["ev"][0] // t), int(fr["ev"][1] // t)
        cx, cy = int(fr["ch"][0] // t), int(fr["ch"][1] // t)
        g[ey, ex] = 5
        g[cy, cx] = 6
        rows = []
        for y in range(0, arena.grid.shape[0], 2):
            row = ""
            for x in range(arena.grid.shape[1]):
                v = int(g[y, x])
                row += "E" if v == 5 else "C" if v == 6 else chars.get(v, "?")
            rows.append(row)
        lines.append(f"t={fr['t']:4d} " + " | ".join(rows))
    return "\n".join(lines)
