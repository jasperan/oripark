"""Render an isolated, deterministic art-direction fixture (not a policy result)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pygame

from oripark.arena import ArenaGenerator
from oripark.config import MoveParams
from oripark.render import _Scene, _glow_sprite


def main():
    pygame.init()
    mp = MoveParams()
    arena = ArenaGenerator(mp, np.random.default_rng(42)).generate(
        np.array([.5, .65, .25, .6, .5, .35]), seed=4242
    )
    grid_before = arena.grid.copy()
    scene = _Scene(arena, mp, seed=42)
    frame = {"t": 0, "ev": arena.ev_spawn, "ch": arena.ch_spawn,
             "ev_act": np.array([2, 0, 0, 0, 0]), "ev_agg": np.zeros(4), "terminal": None}
    surface = scene.static.copy()
    scene.render(surface, [frame], 0, "MOONLIT WILDS / initial arena state")
    output = Path(__file__).resolve().parents[1] / "docs/media/moonlit-observatory.png"
    pygame.image.save(pygame.transform.smoothscale(surface, (1440, 960)), output)
    np.testing.assert_array_equal(arena.grid, grid_before)
    glow = _glow_sprite(32, (100, 200, 220))
    assert glow.get_at((32, 32)).a > glow.get_at((4, 32)).a
    for outcome in ("escaped", "caught", "timeout", "ev_hazard", "ch_hazard"):
        frame["terminal"] = outcome
        scene.render(scene.static.copy(), [frame], 0, "Outcome render check")
    pygame.quit()
    print(f"Captured {output}; verified immutable arena, radial glow, and all five outcomes")


if __name__ == "__main__":
    main()
