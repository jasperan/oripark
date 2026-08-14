"""Ori-park: an Ori-inspired movement physics engine + adversarial RL arena.

Package layout:
  physics.py     vectorized Ori-style movement (jump/djump/wall/dash/bash)
  arena.py       procedural tile arenas with reachability gating
  env.py         vectorized gymnasium VecEnv (evader vs chaser tag game)
  adversaries.py CEM terrain adversary + static samplers
  selfplay.py    self-play league: alternating PPO blocks vs snapshot pool
  metrics.py     Elo, agility metrics, JSONL + plots
  render.py      headless pygame replay -> GIF, matplotlib curves
"""
from .config import EnvParams, MoveParams, TrainParams
from .physics import OriPhysics
from .arena import Arena, ArenaGenerator
from .env import OriArenaVecEnv
from .adversaries import StaticSampler, TerrainAdversary

__version__ = "0.1.0"
