"""Metrics, Elo ratings, and results persistence for self-play training."""
from __future__ import annotations

import json
import os

import numpy as np


class Elo:
    """Two-player Elo with K-factor updates (draws count as 0.5)."""

    def __init__(self, k: float = 32.0, init: float = 1200.0):
        self.k = k
        self.rating = init

    @staticmethod
    def expected(r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))

    def update(self, r_opp: float, score: float) -> float:
        """score: 1 win, 0.5 draw, 0 loss. Returns new rating."""
        e = self.expected(self.rating, r_opp)
        self.rating += self.k * (score - e)
        return self.rating


class Metrics:
    """Collects per-block aggregates and writes JSONL + plots."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.rows = []
        self.episodes = []          # per-episode raw logs

    def add_block(self, rec: dict):
        self.rows.append(rec)
        with open(os.path.join(self.out_dir, "blocks.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def add_episode(self, rec: dict):
        self.episodes.append(self._json_safe(rec))

    @staticmethod
    def _json_safe(o):
        if isinstance(o, dict):
            return {k: Metrics._json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [Metrics._json_safe(v) for v in o]
        if isinstance(o, np.ndarray):
            return Metrics._json_safe(o.tolist())
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    def save_episodes(self):
        with open(os.path.join(self.out_dir, "episodes.jsonl"), "a") as f:
            for e in self.episodes:
                f.write(json.dumps(e) + "\n")
        self.episodes = []

    def summarize_episodes(self, eps: list[dict]) -> dict:
        if not eps:
            return {}
        ev_a = np.stack([e["ev_agility"] for e in eps])
        ch_a = np.stack([e["ch_agility"] for e in eps])
        lens = np.array([e["episode"]["l"] for e in eps])
        outs = [e["outcome"] for e in eps]
        return {
            "n": len(eps),
            "win_rate": float(np.mean([e["ev_win"] for e in eps])),
            "caught": outs.count("caught"),
            "escaped": outs.count("escaped"),
            "ev_hazard": outs.count("ev_hazard"),
            "ch_hazard": outs.count("ch_hazard"),
            "timeout": outs.count("timeout"),
            "avg_len": float(lens.mean()),
            "ev_dashes": float(ev_a[:, 0].mean()),
            "ev_walljumps": float(ev_a[:, 1].mean()),
            "ev_djumps": float(ev_a[:, 2].mean()),
            "ev_bashes": float(ev_a[:, 3].mean()),
            "ev_airtime": float(ev_a[:, 4].mean()),
            "ev_maxspeed": float(ev_a[:, 5].max()),
            "ch_dashes": float(ch_a[:, 0].mean()),
            "ch_walljumps": float(ch_a[:, 1].mean()),
            "ch_djumps": float(ch_a[:, 2].mean()),
            "ch_airtime": float(ch_a[:, 4].mean()),
        }

    def plot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.rows:
            return
        rows = self.rows
        b = [r["block"] for r in rows]
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        axes[0, 0].plot(b, [r["elo_evader"] for r in rows], label="evader", color="#4cc9f0")
        axes[0, 0].plot(b, [r["elo_chaser"] for r in rows], label="chaser", color="#f72585")
        axes[0, 0].set_title("Elo ratings (self-play league)")
        axes[0, 0].legend()
        axes[0, 1].plot(b, [r["eval_ev_win_rate"] for r in rows], color="#4cc9f0")
        axes[0, 1].set_title("Eval: evader win rate vs latest chaser")
        axes[0, 1].set_ylim(0, 1)
        axes[0, 2].plot(b, [r["tr_ev_len"] for r in rows], color="#f4a261")
        axes[0, 2].set_title("Mean episode length during training (steps)")
        axes[1, 0].plot(b, [r["tr_ev_dashes"] for r in rows], label="dash", color="#e9c46a")
        axes[1, 0].plot(b, [r["tr_ev_walljumps"] for r in rows], label="walljump", color="#90e0ef")
        axes[1, 0].plot(b, [r["tr_ev_djumps"] for r in rows], label="djump", color="#b5e48c")
        axes[1, 0].plot(b, [r["tr_ev_bashes"] for r in rows], label="bash", color="#ffb4a2")
        axes[1, 0].set_title("Evader agility usage (per episode)")
        axes[1, 0].legend()
        axes[1, 1].plot(b, [r["tr_ev_airtime"] for r in rows], color="#c77dff")
        axes[1, 1].set_title("Evader airtime (steps airborne / ep)")
        axes[1, 1].plot(b, [r["tr_ch_airtime"] for r in rows], color="#f72585", alpha=0.5)
        axes[1, 2].plot(b, [r["adv_wr"] for r in rows], color="#2ec4b6")
        axes[1, 2].set_title("Terrain adversary: evader win rate under its levels")
        axes[1, 2].set_ylim(0, 1)
        for ax in axes.flat:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "curves.png"), dpi=110)
        plt.close(fig)
