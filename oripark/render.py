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
def _draw_terrain(surf, grid: np.ndarray, mp: MoveParams):
    import pygame
    t = mp.tile
    bg = (16, 30, 38)
    surf.fill(bg)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            v = int(grid[y, x])
            px, py = x * t, y * t
            if v == 1:
                pygame.draw.rect(surf, (178, 137, 104), (px, py, t, t))
                pygame.draw.rect(surf, (216, 178, 140), (px, py, t, 6))
                pygame.draw.rect(surf, (120, 88, 66), (px, py + t - 4, t, 4))
            elif v == 2:
                pygame.draw.polygon(surf, (247, 37, 133),
                                    [(px + 4, py + t), (px + 16, py + 6), (px + 28, py + t)])
                pygame.draw.polygon(surf, (255, 122, 178),
                                    [(px + 8, py + t), (px + 16, py + 12), (px + 24, py + t)])
            elif v == 3:
                pygame.draw.circle(surf, (255, 209, 102), (px + t // 2, py + t // 2), t // 2 + 4)
                pygame.draw.circle(surf, (255, 240, 180), (px + t // 2, py + t // 2), t // 3)
            elif v == 4:
                pygame.draw.rect(surf, (76, 201, 240), (px, py - 30, t, t + 30))
                pygame.draw.circle(surf, (200, 245, 255), (px + t // 2, py), 14)


def _draw_frame(surf, frames, i, label: str, mp: MoveParams):
    import pygame
    t = mp.tile
    # trails
    for k in range(max(1, i - 24), i + 1):
        fr = frames[k]
        f = 1.0 - (i - k) / 25.0
        r = int(6 + 4 * f)
        pygame.draw.circle(surf, (76, 201, 240, 255), (int(fr["ev"][0]), int(fr["ev"][1])), r)
        pygame.draw.circle(surf, (247, 37, 133, 255), (int(fr["ch"][0]), int(fr["ch"][1])), r)
    # bodies
    ev = frames[i]["ev"]; ch = frames[i]["ch"]
    pygame.draw.circle(surf, (224, 251, 252), (int(ev[0]), int(ev[1])), 12)
    pygame.draw.circle(surf, (230, 57, 70), (int(ch[0]), int(ch[1])), 12)
    pygame.draw.circle(surf, (255, 122, 178), (int(ch[0]), int(ch[1])), 6)
    # HUD
    font = pygame.font.SysFont("dejavusansmono", 22)
    d = np.hypot(ev[0] - ch[0], ev[1] - ch[1])
    agg = frames[i]["ev_agg"]
    txt = (f"{label}  t={frames[i]['t']:4d}  dist={d:5.0f}  "
           f"dash={int(agg[0])} wj={int(agg[1])} dj={int(agg[2])} bash={int(agg[3])}")
    surf.blit(font.render(txt, True, (230, 245, 255)), (10, 8))


def make_gif(frames: list, arena: Arena, mp: MoveParams, out_path: str,
             label: str = "", scale: float = 0.5, fps: int = 30, step: int = 2) -> str:
    """Render frames headlessly with pygame and assemble a GIF with PIL."""
    import pygame
    from PIL import Image
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    t = mp.tile
    W, H = mp.arena_w * t, mp.arena_h * t
    static = pygame.Surface((W, H))
    _draw_terrain(static, arena.grid, mp)

    sel = frames[::step]
    imgs = []
    for i, fr in enumerate(sel):
        surf = static.copy()
        _draw_frame(surf, sel, i, label, mp)
        arr = pygame.image.tostring(surf, "RGB")
        img = Image.frombytes("RGB", (W, H), arr)
        if scale != 1.0:
            img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
        imgs.append(img.convert("P", palette=Image.ADAPTIVE, colors=128))
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
