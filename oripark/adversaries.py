"""Adversarial level generator: a CEM policy over arena parameters.

This is the third "adversarial NN" of the system. It proposes level
parameters (gap size, tower height, spike probability, orb count, wander,
dash-gap frequency) and is updated so that the evader's win rate against
the current chaser stays near 50% — keeping the curriculum just past the
edge of the agents' current skill. Uses a tiny cross-entropy-method policy
(gradient-free, robust for low-dim continuous control).
"""
from __future__ import annotations

import numpy as np

from .arena import Arena, ArenaGenerator


class TerrainAdversary:
    def __init__(self, gen: ArenaGenerator, rng: np.random.Generator,
                 pop: int = 8, elites: int = 4, sigma0: float = 0.30,
                 target_wr: float = 0.50, lo: float = 0.0, hi: float = 1.0):
        self.gen = gen
        self.rng = rng
        self.dim = 6
        self.pop = pop
        self.elites = elites
        self.target_wr = target_wr
        self.lo, self.hi = lo, hi
        self.mu = np.full(self.dim, 0.5, dtype=np.float32)
        self.sigma = np.full(self.dim, sigma0, dtype=np.float32)
        self.history = []

    # ------------------------------------------------------------------ sample
    def sample(self, n: int) -> list[Arena]:
        """Draw n arenas from the current parameter distribution."""
        p = np.clip(self.mu[None, :] + self.sigma[None, :] * self.rng.standard_normal((n, self.dim)),
                    self.lo, self.hi)
        out = []
        for row in p:
            out.append(self.gen.generate(row, seed=int(self.rng.integers(2**31))))
        return out

    def mean_params(self) -> np.ndarray:
        return self.mu.copy()

    def fixed_seed_sample(self, n: int, seed0: int = 1000) -> list[Arena]:
        """Deterministic arenas at the mean params (fair evaluation)."""
        out = []
        for k in range(n):
            out.append(self.gen.generate(self.mu, seed=seed0 + k))
        return out

    # ------------------------------------------------------------------ update
    def update(self, eval_fn) -> dict:
        """eval_fn(cands: (pop, 6)) -> evader win rates (pop,)."""
        cands = np.clip(self.mu[None, :] + self.sigma[None, :] * self.rng.standard_normal((self.pop, self.dim)),
                        self.lo, self.hi)
        wrs = np.asarray(eval_fn(cands), dtype=np.float64)
        loss = np.abs(wrs - self.target_wr)
        idx = np.argsort(loss)[:self.elites]
        elites = cands[idx]
        new_mu = elites.mean(axis=0)
        new_sigma = 0.8 * self.sigma + 0.2 * elites.std(axis=0) + 0.02
        self.mu = new_mu.astype(np.float32)
        self.sigma = np.clip(new_sigma, 0.05, 1.0).astype(np.float32)
        rec = {
            "mu": self.mu.copy(), "sigma": self.sigma.copy(),
            "wr_mean": float(wrs.mean()), "wr_std": float(wrs.std()),
            "elite_wr": float(wrs[idx].mean()),
        }
        self.history.append(rec)
        return rec


class StaticSampler:
    """Fixed-parameter sampler (demos, evals, sanity checks)."""

    def __init__(self, gen: ArenaGenerator, params: np.ndarray, rng: np.random.Generator):
        self.gen = gen
        self.params = np.asarray(params, dtype=np.float32)
        self.rng = rng

    def sample(self, n: int) -> list[Arena]:
        return [self.gen.generate(self.params, seed=int(self.rng.integers(2**31))) for _ in range(n)]

    def mean_params(self) -> np.ndarray:
        return self.params
