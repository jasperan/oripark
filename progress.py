#!/usr/bin/env python3
"""Progress curve: evaluate evader checkpoints on a FIXED arena set.

Shares the arena set and the (frozen, deterministic) chaser across every
checkpoint, so the only thing that changes is the evader policy — the
cleanest "improvement over time" measurement. Produces a table, a plot,
and a markdown report.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from oripark.arena import Arena, ArenaGenerator  # noqa: E402
from oripark.config import EnvParams, MoveParams, TrainParams  # noqa: E402
from oripark.env import OriArenaVecEnv  # noqa: E402
from oripark.selfplay import frozen_policy, make_ppo  # noqa: E402


def load(path, role, tp, mp, ep):
    rng = np.random.default_rng(0)
    gen = ArenaGenerator(mp, rng, chaser_frac=ep.chaser_spawn_frac)
    env = OriArenaVecEnv(role, 1, StaticSamplerShim(gen), mp, ep, seed=0)
    p = PPO.load(path, env=env, device="cpu") if path.endswith(".zip") else None
    if p is None:
        p = make_ppo(env, tp, 0, "cpu", side=role)
        p.policy.load_state_dict(torch.load(path, map_location="cpu"))
    env.close()
    return p


class StaticSamplerShim:
    def __init__(self, gen):
        self.gen = gen

    def sample(self, n):
        return [self.gen.generate(np.full(6, 0.5), seed=0) for _ in range(n)]


class FixedArenaSampler:
    """Replays the SAME arenas for every evaluation (fixed seeds)."""

    def __init__(self, gen, mu, sigma, arenas):
        self.gen, self.mu, self.sigma = gen, mu, sigma
        self.arenas = arenas

    def sample(self, n):
        return [self.arenas[k % len(self.arenas)] for k in range(n)]


def run_eval(ev_pol, ch_pol, sampler, mp, ep, n_envs, seed=7) -> dict:
    env = OriArenaVecEnv("evader", n_envs, sampler, mp, ep,
                         opponent=frozen_policy(ch_pol, deterministic=True), seed=seed)
    obs = env.reset()
    esc = caught = tm = 0
    zones, lens = [], []
    for _ in range(ep.max_steps):
        a = ev_pol.predict(obs, deterministic=False)[0]
        obs, _, done, infos = env.step(a)
        for info in infos:
            if "episode" in info:
                o = info["outcome"]
                zones.append(info["ev_max_zone"])
                lens.append(info["episode"]["l"])
                esc += o == "escaped"
                caught += o == "caught"
                tm += o == "timeout"
        if esc + caught + tm >= n_envs:
            break
    env.close()
    n = esc + caught + tm
    return {"esc": esc, "n": n, "caught": caught,
            "esc_rate": esc / max(n, 1),
            "zone": float(np.mean(zones)) if zones else 0.0,
            "length": float(np.mean(lens)) if lens else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/run10")
    ap.add_argument("--matches", type=int, default=24, help="matches per checkpoint (<=32)")
    ap.add_argument("--arena-seeds", type=int, default=24, help="fixed arena count")
    args = ap.parse_args()

    tp, mp, ep = TrainParams(), MoveParams(), EnvParams()
    rng = np.random.default_rng(1)
    gen = ArenaGenerator(mp, rng, chaser_frac=ep.chaser_spawn_frac)
    mu = np.load(os.path.join(args.run, "adv_mu.npy")) if os.path.exists(os.path.join(args.run, "adv_mu.npy")) else np.full(6, 0.5)
    sig = np.load(os.path.join(args.run, "adv_sigma.npy")) if os.path.exists(os.path.join(args.run, "adv_sigma.npy")) else np.full(6, 0.3)

    # fixed arena set sampled once from the final adversary distribution
    arng = np.random.default_rng(4242)
    arenas = [gen.generate(np.clip(mu + sig * arng.standard_normal(6), 0, 1), seed=1000 + k)
              for k in range(args.arena_seeds)]
    sampler = FixedArenaSampler(gen, mu, sig, arenas)

    ch = load(os.path.join(args.run, "chaser.zip"), "chaser", tp, mp, ep)

    # checkpoints: evader_init (true untrained baseline) + evader_b{block}.zip
    base_path = os.path.join(args.run, "evader_init.pt")
    if not os.path.exists(base_path):
        base_path = os.path.join(args.run, "pool_ev_0.pt")
    ckpts = [("init (untrained)", base_path, 0)]
    for p in sorted(glob.glob(os.path.join(args.run, "evader_b*.zip"))):
        m = re.search(r"evader_b(\d+)\.zip", p)
        ckpts.append((m.group(1), p, int(m.group(1))))
    ckpts.sort(key=lambda c: c[2])

    rows = []
    for label, path, block in ckpts:
        ev = load(path, "evader", tp, mp, ep)
        r = run_eval(ev.policy, ch.policy, sampler, mp, ep, n_envs=min(args.matches, 32))
        rows.append((block, label, r))
        print(f"block {block:>4}: esc={r['esc']}/{r['n']} ({r['esc_rate']:.0%}) "
              f"caught={r['caught']} zone={r['zone']:.1f} len={r['length']:.0f}", flush=True)

    # markdown report
    md = [f"# Progress curve (`{args.run}`)",
          "",
          f"Fixed arena set: {len(arenas)} arenas sampled from the final adversary distribution. "
          "Chaser frozen (deterministic). Evader plays stochastically.",
          "",
          "| block | escape rate | caught | avg zone (0-15) | avg length |",
          "|---|---|---|---|---|"]
    for block, label, r in rows:
        md.append(f"| {label} | {r['esc_rate']:.0%} | {r['caught']} | {r['zone']:.1f} | {r['length']:.0f} |")
    if len(rows) >= 2:
        b, t = rows[0][2], rows[-1][2]
        md += ["", f"**Total improvement (block {b} → {rows[-1][1]}):** "
                   f"escape {rows[0][2]['esc_rate']:.0%} → {rows[-1][2]['esc_rate']:.0%} "
                   f"(+{rows[-1][2]['esc_rate'] - rows[0][2]['esc_rate']:.0%}), "
                   f"caught {rows[0][2]['caught']} → {rows[-1][2]['caught']}, "
                   f"zone {rows[0][2]['zone']:.1f} → {rows[-1][2]['zone']:.1f}."]
    out_md = os.path.join(args.run, "progress.md")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blocks = [r[0] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(blocks, [r[2]["esc_rate"] for r in rows], "o-", color="#4cc9f0")
    ax[0, 0].set_title("Escape rate (portal reached)")
    ax[0, 0].set_ylim(0, 1)
    ax[0, 1].plot(blocks, [r[2]["caught"] for r in rows], "o-", color="#f72585")
    ax[0, 1].set_title("Times caught")
    ax[1, 0].plot(blocks, [r[2]["zone"] for r in rows], "o-", color="#2ec4b6")
    ax[1, 0].set_title("Avg traversal zone (0-15)")
    ax[1, 1].plot(blocks, [r[2]["length"] for r in rows], "o-", color="#e9c46a")
    ax[1, 1].set_title("Avg episode length (steps)")
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.suptitle("Evader improvement over self-play training (fixed arenas, frozen chaser)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.run, "progress.png"), dpi=110)
    plt.close(fig)
    print(f"\nwrote {out_md} and progress.png")


if __name__ == "__main__":
    main()
