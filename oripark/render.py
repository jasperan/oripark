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
        if frames:
            frames[-1]["terminal"] = outcome
        return frames, outcome

    def close(self):
        self.env.close()


# ------------------------------------------------------------------ rendering
_PAL = {
    "sky_top": (13, 30, 49), "sky_mid": (46, 102, 119),
    "sky_low": (27, 65, 82), "sky_bot": (8, 20, 34),
    "canopy_far": (26, 60, 74), "canopy_near": (9, 28, 41),
    "rock": (49, 68, 76), "rock_dark": (29, 45, 55), "rock_deep": (18, 31, 43),
    "moss": (78, 151, 144), "moss_hi": (156, 221, 191), "moss_lo": (45, 104, 107),
    "spike": (226, 44, 116), "spike_hi": (255, 150, 196), "spike_lo": (128, 18, 62),
    "orb": (255, 206, 96), "orb_hi": (255, 248, 214), "orb_ring": (188, 132, 46),
    "portal": (118, 226, 255), "portal_hi": (238, 252, 255), "portal_deep": (36, 108, 168),
    "evader": (236, 250, 255), "evader_core": (255, 255, 255), "evader_glow": (168, 226, 255),
    "chaser": (120, 22, 40), "chaser_core": (255, 110, 130), "chaser_glow": (255, 62, 92),
    "firefly": (255, 240, 190), "hud": (8, 16, 28), "hud_text": (226, 242, 255),
    "hud_dim": (150, 176, 200), "good": (128, 230, 168), "bad": (255, 122, 138),
    "ray": (165, 223, 235),
}

OUTCOME_LABEL = {
    "escaped": "ESCAPED", "caught": "CAUGHT", "timeout": "TIMED OUT",
    "ev_hazard": "HIT SPIKES", "ch_hazard": "CHASER FELL",
}


def _hash2(x: int, y: int, salt: int = 0) -> float:
    """Deterministic [0,1) hash for per-tile variation."""
    h = (x * 374761393 + y * 668265263 + salt * 1274126177) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFFFF) / 0xFFFFFFFF


def _glow_sprite(radius: int, color, peak: int = 200, steps: int = 20) -> "pygame.Surface":
    """Radial alpha-gradient glow sprite (per-pixel alpha)."""
    import pygame
    r = max(2, int(radius))
    surf = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
    for k in range(steps, 0, -1):
        f = k / steps
        a = int(peak * (1.0 - f) ** 2)
        pygame.draw.circle(surf, (*color, a), (r, r), int(r * f))
    return surf


def _blur_layer(size, draw_fn, radius: float):
    """Draw with PIL, blur it, return the PIL layer — the cheap way to get
    atmospheric depth that pygame cannot do natively."""
    from PIL import Image, ImageDraw, ImageFilter
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(img, "RGBA"))
    return img.filter(ImageFilter.GaussianBlur(radius))


def _fade_alpha(img, top: float, bottom: float):
    """Multiply the layer's alpha by a vertical ramp (top -> bottom)."""
    import numpy as np
    from PIL import Image
    a = np.array(img)
    h = a.shape[0]
    ramp = np.linspace(top, bottom, h)[:, None]
    a[..., 3] = np.clip(a[..., 3] * ramp, 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def _pil_to_pygame(img):
    import pygame
    return pygame.image.fromstring(img.tobytes(), img.size, img.mode)


class _Scene:
    """Precomputed painterly scene for one arena; renders frames cheaply."""

    def __init__(self, arena, mp: MoveParams, seed: int = 0):
        import pygame
        self.mp = mp
        self.t = t = float(mp.tile)
        self.W = mp.arena_w * mp.tile
        self.H = mp.arena_h * mp.tile
        self.arena = arena
        self.g_ev = _glow_sprite(38, _PAL["evader_glow"], peak=225)
        self.g_ch = _glow_sprite(38, _PAL["chaser_glow"], peak=240)
        self.g_orb = _glow_sprite(34, _PAL["orb"], peak=170)
        self.g_portal = _glow_sprite(64, _PAL["portal"], peak=190)
        self.g_fly = _glow_sprite(6, _PAL["firefly"], peak=150)
        self.g_spike = _glow_sprite(20, _PAL["spike"], peak=90)
        self.static = pygame.Surface((self.W, self.H))
        self._draw_background()
        self._draw_terrain(arena.grid)
        rng = np.random.default_rng(seed * 7919 + 13)
        n = 40
        self.ff = np.stack([
            rng.uniform(0, self.W, n), rng.uniform(self.H * 0.10, self.H * 0.92, n),
            rng.uniform(0, 6.28, n), rng.uniform(0.6, 1.4, n),
        ], axis=1)
        self.vig = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        self._draw_vignette()
        # portal progress reference: spawn -> portal distance
        sx, sy = arena.ev_spawn
        px, py = arena.portal
        self.spawn_d = float(np.hypot(px - sx, py - sy))
        try:
            self.font = pygame.font.SysFont("dejavusans", 28, bold=True)
            self.font_sm = pygame.font.SysFont("dejavusansmono", 21)
            self.font_big = pygame.font.SysFont("dejavusans", 54, bold=True)
        except Exception:                               # pragma: no cover
            self.font = self.font_sm = self.font_big = pygame.font.Font(None, 24)

    # ------------------------------------------------------------ background
    def _draw_background(self):
        import pygame
        W, H, P = self.W, self.H, _PAL
        stops = [(0.0, P["sky_top"]), (0.30, P["sky_mid"]),
                 (0.64, P["sky_low"]), (1.0, P["sky_bot"])]
        for y in range(H):
            f = y / H
            i = 1
            while i < len(stops) - 1 and f > stops[i][0]:
                i += 1
            (y0, c0), (y1, c1) = stops[i - 1], stops[i]
            k = 0.0 if y1 == y0 else (f - y0) / (y1 - y0)
            col = tuple(int(c0[j] + (c1[j] - c0[j]) * k) for j in range(3))
            pygame.draw.line(self.static, col, (0, y), (W, y))
        # A diffused moon anchors the depth; it never changes collision geometry.
        moon_x, moon_y = int(W * .70), int(H * .24)
        halo = _glow_sprite(int(H * .21), (152, 218, 232), peak=100, steps=48)
        self.static.blit(halo, (moon_x - halo.get_width() // 2,
                               moon_y - halo.get_height() // 2))
        pygame.draw.circle(self.static, (183, 222, 221), (moon_x, moon_y), int(H * .048))
        pygame.draw.circle(self.static, (208, 236, 225), (moon_x - 8, moon_y - 9),
                           int(H * .032))
        # god rays: blurred cool wedges fading out toward the floor (they used
        # to be hard-edged triangles that read as flat geometry, not light)
        def rays(d):
            cx = W * 0.5
            for ox, span, a in [(-0.35, 0.16, 30), (0.05, 0.22, 40),
                                (0.45, 0.12, 26), (0.25, 0.09, 20)]:
                px = cx + ox * W
                d.polygon([(px - span * W * 0.5, 0), (px + span * W * 0.5, 0),
                           (px + span * W * 1.6, H), (px - span * W * 1.6, H)],
                          fill=(*P["ray"], a))
        self.static.blit(_pil_to_pygame(_fade_alpha(_blur_layer((W, H), rays, 26),
                                                    1.0, 0.15)), (0, 0))
        # distant floodlit rock forms in the mid band: parallax depth without
        # competing with the play space (smooth sum-of-sines silhouettes — the
        # first attempt used random triangles and read as a sawtooth range)
        def hills(d):
            for yb, amp, col, a, f0, ph in [
                    (H * 0.62, H * 0.10, P["canopy_far"], 110, 1.7, 0.0),
                    (H * 0.72, H * 0.08, P["canopy_near"], 140, 2.6, 2.1)]:
                xs = np.linspace(-60.0, W + 60.0, 260)
                ys = yb - amp * (0.62 * np.sin(xs / W * f0 * 2 * np.pi + ph)
                                 + 0.38 * np.sin(xs / W * f0 * 3.7 * np.pi + ph * 1.7)
                                 + 0.22 * np.sin(xs / W * f0 * 7.1 * np.pi + ph * 0.6))
                d.polygon(list(zip(xs, ys)) + [(W + 60, H), (-60, H)],
                          fill=(*col, a))
        self.static.blit(_pil_to_pygame(_blur_layer((W, H), hills, 11)), (0, 0))
        # Distant trunks and their branching silhouettes sit behind every platform.
        def woodland(d):
            rng = np.random.default_rng(107)
            for depth, count in ((0, 19), (1, 11)):
                for x in np.linspace(-40, W + 40, count):
                    x += rng.uniform(-35, 35)
                    base = H * rng.uniform(.64, .94)
                    top = H * rng.uniform(.08, .3)
                    lean = rng.uniform(-80, 80)
                    width = int(rng.uniform(8, 17) * (depth + 1))
                    color = (19, 49, 65, 55 + depth * 35)
                    d.polygon([(x - width, base), (x + width, base),
                               (x + lean + 4, top), (x + lean, top)], fill=color)
                    for branch in range(3):
                        y = top + (base - top) * (.16 + branch * .17)
                        direction = -1 if branch % 2 else 1
                        d.line([(x + lean * .6, y + 80),
                                (x + direction * 70, y),
                                (x + direction * 105, y - 75)], fill=color,
                               width=max(2, width // 3))
        self.static.blit(_pil_to_pygame(_blur_layer((W, H), woodland, 4)), (0, 0))
        # canopy: ONE organic mass hanging from the top edge with vines, built
        # blurred so it reads as out-of-focus foliage. The old version was two
        # layers of near-black circles over the whole arena and looked like
        # storm clouds sitting on top of the platforms.
        band = H * 0.17
        edge_x = np.linspace(0, W, 72)
        _rng = np.random.default_rng(4242)
        edge_y = (band * (0.52 + 0.48 * np.sin(edge_x / W * 5.4 + 1.1))
                  + _rng.uniform(-8, 8, edge_x.size))

        def canopy(d):
            d.polygon(list(zip(edge_x, edge_y)) + [(W, 0), (0, 0)],
                      fill=(*P["canopy_near"], 255))
            for x, e in zip(edge_x[::2], edge_y[::2]):
                r = _rng.uniform(20, 52)
                d.ellipse([x - r * 1.35, e - r * 1.7, x + r * 1.35, e + r * 0.6],
                          fill=(*P["canopy_near"], 255))
        self.static.blit(_pil_to_pygame(_blur_layer((W, H), canopy, 7)), (0, 0))

        def leaves(d):
            for x, e in zip(edge_x[::3], edge_y[::3]):
                for _ in range(2):
                    lx = x + _rng.uniform(-30, 30)
                    ly = e + _rng.uniform(-4, 18)
                    lr = _rng.uniform(6, 14)
                    d.ellipse([lx - lr, ly - lr * 0.55, lx + lr, ly + lr * 0.55],
                              fill=(*P["canopy_far"], 190))
                if _rng.random() < 0.22:                 # hanging vine
                    vx = x + _rng.uniform(-18, 18)
                    vlen = _rng.uniform(26, 110)
                    d.line([(vx, e - 6), (vx + _rng.uniform(-10, 10), e + vlen)],
                           fill=(*P["canopy_far"], 120), width=1)
                    for kk in range(3):                  # sparse leaves down the vine
                        ly = e + vlen * (0.35 + 0.25 * kk)
                        d.ellipse([vx - 4, ly - 2, vx + 4, ly + 2],
                                  fill=(*P["canopy_far"], 120))
        self.static.blit(_pil_to_pygame(_blur_layer((W, H), leaves, 2.2)), (0, 0))

    def _draw_vignette(self):
        import pygame
        W, H = self.W, self.H
        for y in range(0, H, 8):
            for x in range(0, W, 8):
                dx = min(x, W - x) / (W / 2)
                dy = min(y, H - y) / (H / 2)
                d = min(1.0, dx * dx + dy * dy)
                a = int(58 * max(0.0, d - 0.62) / 0.38)
                if a > 0:
                    pygame.draw.rect(self.vig, (2, 6, 12, a), (x, y, 8, 8))

    # --------------------------------------------------------------- terrain
    def _draw_terrain(self, grid: np.ndarray):
        """Rock masses rather than per-tile bricks: fill horizontal runs, then
        cap the exposed top with moss. Reads as terrain, not as a grid."""
        import pygame
        t, P = int(self.t), _PAL
        H, W = grid.shape
        solid = (grid == 1)
        surface = solid & np.vstack([np.ones((1, W), dtype=bool), ~solid[:-1]])
        # 0) soft drop shadows under overhangs: the single biggest readability
        # win — without them the platforms read as flat slabs painted on the fog
        def shadows(d):
            for y in range(H):
                x = 0
                while x < W:
                    if not solid[y, x]:
                        x += 1
                        continue
                    x0 = x
                    while x < W and solid[y, x]:
                        x += 1
                    if y + 1 < H and not solid[y + 1, x0]:
                        d.rectangle([x0 * t, y * t + t - 2, (x0 + x - x0) * t,
                                     y * t + t + 9], fill=(3, 9, 15, 105))
        self.static.blit(_pil_to_pygame(_blur_layer((self.W, self.H), shadows, 7)),
                         (0, 0))
        # 1) body: merged horizontal runs with SUBTLE per-run tint variation
        # (hashing on the row made vertical walls look like brick banding) and
        # long sedimentary strata lines instead of noisy per-tile speckle
        for y in range(H):
            x = 0
            while x < W:
                if not solid[y, x]:
                    x += 1
                    continue
                x0 = x
                while x < W and solid[y, x]:
                    x += 1
                run = x - x0
                h = _hash2(x0, y // 6, 3)
                base = tuple(int(c * (0.95 + 0.10 * h)) for c in P["rock"])
                pygame.draw.rect(self.static, base, (x0 * t, y * t, run * t, t))
                # a darker seam every few rows reads as strata
                if _hash2(x0 // 3, y, 5) > 0.72:
                    pygame.draw.rect(self.static, P["rock_dark"],
                                     (x0 * t, y * t + t - 3, run * t, 3))
                depth = 0
                yy = y - 1
                while yy >= 0 and solid[yy, x0]:
                    depth += 1
                    yy -= 1
                if depth > 1:
                    a = min(150, 26 * depth)
                    shade = pygame.Surface((run * t, t), pygame.SRCALPHA)
                    shade.fill((*P["rock_dark"], a))
                    self.static.blit(shade, (x0 * t, y * t))
        # 2) moss cap on exposed tops (+ a thin lit rim)
        for y in range(H):
            x = 0
            while x < W:
                if not surface[y, x]:
                    x += 1
                    continue
                x0 = x
                while x < W and surface[y, x]:
                    x += 1
                run = x - x0
                h = _hash2(x0, y, 11)
                cap = 7 + int(3 * h)
                pygame.draw.rect(self.static, P["moss"], (x0 * t, y * t, run * t, cap))
                pygame.draw.rect(self.static, P["moss_hi"], (x0 * t, y * t, run * t, 2))
                pygame.draw.line(self.static, P["moss_lo"], (x0 * t, y * t + cap),
                                 ((x0 + run) * t, y * t + cap))
                # tufts along the cap
                for k in range(run):
                    if _hash2(x0 + k, y, 17) > 0.55:
                        gx = (x0 + k) * t + int(_hash2(x0 + k, y, 23) * (t - 8)) + 3
                        pygame.draw.polygon(self.static, P["moss"],
                                            [(gx, y * t + 1), (gx + 2, y * t - 6), (gx + 5, y * t + 1)])
                # left/right rock lips
                pygame.draw.rect(self.static, P["rock_deep"], (x0 * t, y * t + cap, 2, t - cap))
                pygame.draw.rect(self.static, P["rock_deep"],
                                 ((x0 + run) * t - 2, y * t + cap, 2, t - cap))
        # 3) hazards, orbs, portal
        for y in range(H):
            for x in range(W):
                v = int(grid[y, x])
                if v == 2:
                    self._tile_spike(x * t, y * t)
                elif v == 3:
                    self._tile_orb(x * t, y * t)
                elif v == 4:
                    self._tile_portal(x * t, y * t)

    def _tile_spike(self, px, py):
        import pygame
        t, P = int(self.t), _PAL
        # base plate so the hazard reads even at small scale
        pygame.draw.rect(self.static, P["spike_lo"], (px, py + t - 9, t, 9))
        pygame.draw.rect(self.static, (58, 10, 28), (px, py + t - 3, t, 3))
        for k, (x0, x1, x2) in enumerate([(1, 9, 5), (10, 20, 15), (20, 30, 25)]):
            pygame.draw.polygon(self.static, P["spike"],
                                [(px + x0, py + t - 8), (px + x2, py + 3), (px + x1, py + t - 8)])
            pygame.draw.line(self.static, P["spike_hi"],
                             (px + x2, py + 5), (px + x2 - 3, py + t - 10), 2)

    def _tile_orb(self, px, py):
        import pygame
        t, P = int(self.t), _PAL
        c = (px + t // 2, py + t // 2)
        self.static.blit(self.g_orb, (c[0] - 34, c[1] - 34))
        pygame.draw.circle(self.static, P["orb_ring"], c, 15)
        pygame.draw.circle(self.static, P["orb"], c, 12)
        pygame.draw.circle(self.static, P["orb_hi"], (c[0] - 4, c[1] - 4), 5)
        pygame.draw.arc(self.static, P["orb_hi"], (c[0] - 17, c[1] - 17, 34, 34),
                        0.6, 2.4, 2)

    def _tile_portal(self, px, py):
        import pygame
        t, P = int(self.t), _PAL
        cx = px + t // 2
        cy = py + t // 2
        # light column with a soft halo, built blurred so it reads as glow
        h = max(1, py)

        def beam(d):
            w0 = t * 1.6
            for k in range(16):
                a = int(34 * (1 - k / 16))
                s = max(1.0, w0 * (1 - k / 16))
                d.rectangle([cx - s, 0, cx + s, h], fill=(*P["portal"], a))
        self.static.blit(_pil_to_pygame(_fade_alpha(
            _blur_layer((self.W, self.H), beam, 12), 0.35, 1.0)), (0, 0))
        self.static.blit(self.g_portal, (cx - 64, cy - 64))
        # a gate, not a bubble: dark throat, bright rim, orbiting arcs
        pygame.draw.circle(self.static, P["portal_deep"], (cx, cy), 27)
        pygame.draw.circle(self.static, (18, 40, 66), (cx, cy), 20)
        pygame.draw.circle(self.static, P["portal"], (cx, cy), 24, 3)
        pygame.draw.circle(self.static, P["portal_hi"], (cx, cy), 16, 2)
        for a0 in (0.2, 1.8, 3.4, 5.0):
            pygame.draw.arc(self.static, P["portal_hi"],
                            (cx - 20, cy - 20, 40, 40), a0, a0 + 1.1, 3)
        pygame.draw.circle(self.static, P["portal_hi"], (cx, cy), 5)
        pygame.draw.circle(self.static, (255, 255, 255), (cx - 2, cy - 2), 2)

    # ------------------------------------------------------------------ frame
    def render(self, surf, frames, i, label: str):
        import pygame
        P = _PAL
        # comet trails: few, small, quickly fading (was 22 full-size rings)
        for k in range(max(0, i - 12), i):
            f = (i - k) / 13.0
            fr = frames[k]
            for pos, col in ((fr["ev"], P["evader_glow"]), (fr["ch"], P["chaser_glow"])):
                r = int(3 + 6 * f)
                s = pygame.Surface((2 * r + 2, 2 * r + 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*col, int(170 * f)), (r + 1, r + 1), r)
                surf.blit(s, (int(pos[0]) - r - 1, int(pos[1]) - r - 1))
        ev, ch = frames[i]["ev"], frames[i]["ch"]
        ex, ey, cx, cy = int(ev[0]), int(ev[1]), int(ch[0]), int(ch[1])
        face = int(frames[i]["ev_act"][0])
        fdx = 5 if face == 2 else -5 if face == 1 else 0
        # chaser first (threat sits behind the hero)
        surf.blit(self.g_ch, (cx - 38, cy - 38))
        pygame.draw.circle(surf, (18, 4, 10), (cx, cy), 16)
        pygame.draw.circle(surf, P["chaser"], (cx, cy), 14)
        pygame.draw.circle(surf, P["chaser_core"], (cx + fdx, cy - 4), 6)
        pygame.draw.circle(surf, (255, 40, 70), (cx + fdx, cy - 4), 3)
        # evader: glow, body, core, facing eye — with a dark rim so it stays
        # readable against both the bright sky and the dark rock
        surf.blit(self.g_ev, (ex - 38, ey - 38))
        pygame.draw.ellipse(surf, (28, 52, 74), (ex - 13, ey - 17, 26, 34))
        pygame.draw.ellipse(surf, P["evader_glow"], (ex - 11, ey - 15, 22, 30))
        pygame.draw.ellipse(surf, P["evader"], (ex - 8, ey - 12, 16, 24))
        pygame.draw.circle(surf, P["evader_core"], (ex, ey - 4), 6)
        pygame.draw.circle(surf, (44, 70, 92), (ex + fdx, ey - 5), 3)
        # fireflies
        t = frames[i]["t"]
        for fx, fy, ph, sp in self.ff:
            yy = fy + 10 * np.sin(t * 0.02 * sp + ph)
            xx = fx + 6 * np.sin(t * 0.013 * sp + ph * 1.7)
            a = int(120 + 110 * np.sin(t * 0.05 * sp + ph))
            if a <= 0:
                continue
            self.g_fly.set_alpha(min(255, a))
            surf.blit(self.g_fly, (int(xx) - 6, int(yy) - 6))
        self.g_fly.set_alpha(255)
        surf.blit(self.vig, (0, 0))
        self._hud(surf, frames, i, label, (ex, ey), (cx, cy))
        term = frames[i].get("terminal")
        if term:
            self._banner(surf, term, frames[i]["t"])

    def _hud(self, surf, frames, i, label, ev, ch):
        import pygame
        P = _PAL
        pad, h = 24, 104
        panel = pygame.Surface((self.W, h), pygame.SRCALPHA)
        panel.fill((*P["hud"], 226))
        pygame.draw.line(panel, (66, 111, 127, 220), (pad, h - 2), (self.W - pad, h - 2), 1)
        surf.blit(panel, (0, 0))
        surf.blit(self.font_sm.render("ORI / FIELD OBSERVATORY", True, P["portal"]), (pad, 12))
        label_image = self.font.render(label, True, P["hud_text"])
        max_label_width = int(self.W * .62) - pad * 2
        if label_image.get_width() > max_label_width:
            ratio = max_label_width / label_image.get_width()
            label_image = pygame.transform.smoothscale(label_image,
                (max_label_width, max(1, int(label_image.get_height() * ratio))))
        surf.blit(label_image, (pad, 40))
        d = float(np.hypot(ev[0] - ch[0], ev[1] - ch[1]))
        agg = frames[i]["ev_agg"]
        stats = (f"STEP {frames[i]['t']:04d}  /  GAP {d:.0f}px  /  "
                 f"DASH {int(agg[0])}  WALL {int(agg[1])}  "
                 f"JUMP {int(agg[2])}  BASH {int(agg[3])}")
        surf.blit(self.font_sm.render(stats, True, P["hud_dim"]), (pad, 76))
        # portal progress bar
        px, py = float(self.arena.portal[0]), float(self.arena.portal[1])
        prog = 1.0 - float(np.hypot(px - ev[0], py - ev[1])) / max(self.spawn_d, 1.0)
        prog = min(1.0, max(0.0, prog))
        bw, bh = int(self.W * 0.30), 8
        bx, by = self.W - bw - pad, 57
        pygame.draw.rect(surf, (20, 38, 54), (bx, by, bw, bh), border_radius=6)
        pygame.draw.rect(surf, P["portal"], (bx, by, int(bw * prog), bh), border_radius=6)
        pygame.draw.rect(surf, (70, 110, 140), (bx, by, bw, bh), 1, border_radius=6)
        surf.blit(self.font_sm.render(f"PORTAL PROXIMITY / {prog:.0%}", True, P["hud_dim"]), (bx, 21))
        for fraction in (.25, .5, .75):
            tick_x = bx + int(bw * fraction)
            pygame.draw.line(surf, P["hud"], (tick_x, by), (tick_x, by + bh), 2)

    def _banner(self, surf, outcome, t):
        import pygame
        col = _PAL["good"] if outcome in ("escaped", "ch_hazard") else _PAL["bad"]
        txt = OUTCOME_LABEL.get(outcome, outcome.upper())
        y = self.H // 2 - 76
        band = pygame.Surface((self.W, 152), pygame.SRCALPHA)
        band.fill((6, 18, 28, 222))
        surf.blit(band, (0, y))
        img = self.font_big.render(txt, True, col)
        surf.blit(img, (self.W // 2 - img.get_width() // 2, y + 28))
        detail = self.font_sm.render(f"REPLAY COMPLETE  /  {t} steps  /  {t / self.mp.step_hz:.1f}s", True, _PAL["hud_dim"])
        surf.blit(detail, (self.W // 2 - detail.get_width() // 2, y + 103))


def _draw_terrain(surf, grid: np.ndarray, mp: MoveParams):
    """Back-compat stub: flat background used by older callers."""
    import pygame
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
    if not sel:
        raise ValueError("no frames to render")
    if frames[-1].get("terminal") and sel[-1] is not frames[-1]:
        sel.append(frames[-1])
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
