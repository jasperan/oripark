#!/usr/bin/env python3
"""Head-to-head evaluation of the trained league, producing a markdown report."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from oripark.adversaries import StaticSampler  # noqa: E402
from oripark.arena import ArenaGenerator  # noqa: E402
from oripark.config import EnvParams, MoveParams, TrainParams  # noqa: E402
from oripark.env import OriArenaVecEnv  # noqa: E402
from oripark.selfplay import evaluate, make_ppo  # noqa: E402


def load(path, role, tp, mp, ep, device):
    from stable_baselines3 import PPO
    rng = np.random.default_rng(0)
    gen = ArenaGenerator(mp, rng)
    env = OriArenaVecEnv(role, 1, StaticSampler(gen, np.full(6, 0.5), rng), mp, ep, seed=0)
    if path.endswith(".zip"):
        ppo = PPO.load(path, env=env, device=device)
        env.close()
        return ppo
    ppo = make_ppo(env, tp, 0, device)
    ppo.policy.load_state_dict(torch.load(path, map_location=device))
    env.close()
    return ppo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/run1")
    ap.add_argument("--matches", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", action="store_true",
                    help="also evaluate the untrained pool_ev_0 baseline for improvement comparison")
    args = ap.parse_args()

    tp = TrainParams()
    mp = MoveParams()
    ep = EnvParams()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ev = load(os.path.join(args.run, "evader.zip"), "evader", tp, mp, ep, device)
    ch = load(os.path.join(args.run, "chaser.zip"), "chaser", tp, mp, ep, device)

    rng = np.random.default_rng(1)
    gen = ArenaGenerator(mp, rng)
    mu = np.load(os.path.join(args.run, "adv_mu.npy")) if os.path.exists(os.path.join(args.run, "adv_mu.npy")) else np.full(6, 0.5)
    sampler = StaticSampler(gen, mu, rng)

    res = evaluate(ev.policy, ch.policy, sampler, mp, ep,
                   n_envs=min(args.matches, 16), max_steps=ep.max_steps)
    n = res["wins"] + res["losses"] + res["draws"]

    md = f"""# Ori-park head-to-head report (`{args.run}`)

Arena params (terrain adversary mean): `{np.round(mu, 2).tolist()}`

## Trained evader vs trained chaser

| metric | value |
|---|---|
| matches | {n} |
| evader wins (escaped / chaser died) | {res["wins"]} |
| chaser wins (caught / evader died) | {res["losses"]} |
| timeouts (stalemate) | {res["draws"]} |
| **escape rate (portal reached)** | **{res["escaped"] / max(n, 1):.2%}** |
| catch rate | {res["caught"] / max(n, 1):.2%} |
| avg episode length | {res["avg_len"]:.0f} steps ({res["avg_len"] / 60:.1f} s) |
"""

    if args.baseline:
        base = load(os.path.join(args.run, "pool_ev_0.pt"), "evader", tp, mp, ep, device)
        rb = evaluate(base.policy, ch.policy, sampler, mp, ep,
                      n_envs=min(args.matches, 16), max_steps=ep.max_steps)
        nb = rb["wins"] + rb["losses"] + rb["draws"]
        esc_trained = res["escaped"] / max(n, 1)
        esc_base = rb["escaped"] / max(nb, 1)
        md += f"""## Improvement vs untrained baseline (same arenas, same chaser)

| policy | matches | escape rate | caught | avg length |
|---|---|---|---|---|
| untrained (pool_ev_0) | {nb} | {esc_base:.2%} | {rb["caught"]} | {rb["avg_len"]:.0f} steps |
| trained (evader.zip) | {n} | {esc_trained:.2%} | {res["caught"]} | {res["avg_len"]:.0f} steps |
| **improvement** | | **{esc_trained - esc_base:+.2%}** | **{res["caught"] - rb["caught"]:+.0f}** | **{res["avg_len"] - rb["avg_len"]:+.0f} steps** |
"""

    out = args.out or os.path.join(args.run, "report.md")
    with open(out, "w") as f:
        f.write(md)
    print(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
