#!/usr/bin/env python3
"""Head-to-head evaluation of the trained league, producing a markdown report."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from oripark.adversaries import StaticSampler, TerrainAdversary  # noqa: E402
from oripark.arena import Arena, ArenaGenerator  # noqa: E402
from oripark.config import EnvParams, MoveParams, TrainParams  # noqa: E402
from oripark.env import OriArenaVecEnv  # noqa: E402
from oripark.selfplay import frozen_policy, make_ppo  # noqa: E402


def load(path, role, tp, mp, ep, device):
    from stable_baselines3 import PPO
    rng = np.random.default_rng(0)
    gen = ArenaGenerator(mp, rng)
    env = OriArenaVecEnv(role, 1, StaticSampler(gen, np.full(6, 0.5), rng), mp, ep, seed=0)
    if path.endswith(".zip"):
        ppo = PPO.load(path, env=env, device=device)
        env.close()
        return ppo
    ppo = make_ppo(env, tp, 0, device, side=role)
    ppo.policy.load_state_dict(torch.load(path, map_location=device))
    env.close()
    return ppo


class GaussianSampler:
    """Samples arenas from the final terrain-adversary parameter distribution."""

    def __init__(self, gen: ArenaGenerator, mu: np.ndarray, sigma: np.ndarray,
                 rng: np.random.Generator, seed0: int = 1000):
        self.gen, self.mu, self.sigma = gen, mu, sigma
        self.rng, self.seed0 = rng, seed0

    def sample(self, n: int) -> list[Arena]:
        out = []
        for k in range(n):
            p = np.clip(self.mu + self.sigma * self.rng.standard_normal(6), 0, 1)
            out.append(self.gen.generate(p, seed=self.seed0 + k))
        return out

    def mean_params(self) -> np.ndarray:
        return self.mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/run1")
    ap.add_argument("--matches", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--deterministic", action="store_true",
                    help="eval with deterministic evader (stochastic is the default honest measure)")
    ap.add_argument("--baseline", action="store_true",
                    help="also evaluate the untrained pool_ev_0 baseline for improvement comparison")
    args = ap.parse_args()

    tp = TrainParams()
    mp = MoveParams()
    ep = EnvParams()
    device = "cpu"

    ev = load(os.path.join(args.run, "evader.zip"), "evader", tp, mp, ep, device)
    ch = load(os.path.join(args.run, "chaser.zip"), "chaser", tp, mp, ep, device)

    rng = np.random.default_rng(1)
    gen = ArenaGenerator(mp, rng, chaser_frac=ep.chaser_spawn_frac)
    mu = np.load(os.path.join(args.run, "adv_mu.npy")) if os.path.exists(os.path.join(args.run, "adv_mu.npy")) else np.full(6, 0.5)
    sig = np.load(os.path.join(args.run, "adv_sigma.npy")) if os.path.exists(os.path.join(args.run, "adv_sigma.npy")) else np.full(6, 0.3)
    sampler = GaussianSampler(gen, mu, sig, rng)

    n_envs = min(args.matches, 32)
    ev_det = args.deterministic
    res = run_eval(ev.policy, ch.policy, sampler, mp, ep, n_envs, ev_det=ev_det)
    n = res["esc"] + res["caught"] + res["tm"]

    md = f"""# Ori-park head-to-head report (`{args.run}`)

Eval protocol: evader plays **{('deterministic' if ev_det else 'stochastic')}**, chaser deterministic, arenas sampled from the final terrain-adversary distribution (mu `{np.round(mu, 2).tolist()}`, sigma `{np.round(sig, 2).tolist()}`).

## Trained evader vs trained chaser

| metric | value |
|---|---|
| matches | {n} |
| **escape rate (portal reached)** | **{res["esc"] / max(n, 1):.2%}** |
| caught rate | {res["caught"] / max(n, 1):.2%} |
| timeouts (stalemate) | {res["tm"]} |
| avg traversal zone (0-15) | {res["zone"]:.1f} |
| avg episode length | {res["length"]:.0f} steps ({res["length"] / 60:.1f} s) |
"""

    if args.baseline:
        base_path = os.path.join(args.run, "evader_init.pt")
        if not os.path.exists(base_path):
            base_path = os.path.join(args.run, "pool_ev_0.pt")
        base = load(base_path, "evader", tp, mp, ep, device)
        rb = run_eval(base.policy, ch.policy, sampler, mp, ep, n_envs)
        nb = rb["esc"] + rb["caught"] + rb["tm"]
        md += f"""## Improvement vs untrained baseline (same arenas, same chaser)

| policy | matches | escape rate | caught | avg zone | avg length |
|---|---|---|---|---|---|
| untrained (pool_ev_0) | {nb} | {rb["esc"] / max(nb, 1):.2%} | {rb["caught"]} | {rb["zone"]:.1f} | {rb["length"]:.0f} steps |
| trained (evader.zip) | {n} | {res["esc"] / max(n, 1):.2%} | {res["caught"]} | {res["zone"]:.1f} | {res["length"]:.0f} steps |
| **improvement** | | **+{(res["esc"] / max(n, 1) - rb["esc"] / max(nb, 1)):.2%}** | **{res["caught"] - rb["caught"]:+d}** | **{res["zone"] - rb["zone"]:+.1f}** | **{res["length"] - rb["length"]:+.0f} steps** |
"""

    out = args.out or os.path.join(args.run, "report.md")
    with open(out, "w") as f:
        f.write(md)
    print(md)
    print(f"wrote {out}")


def run_eval(ev_pol, ch_pol, sampler, mp, ep, n_envs, ev_det: bool = False, seed: int = 7) -> dict:
    """Stochastic evader vs deterministic chaser on a shared arena sampler."""
    env = OriArenaVecEnv("evader", n_envs, sampler, mp, ep,
                         opponent=frozen_policy(ch_pol, deterministic=True), seed=seed)
    obs = env.reset()
    esc = caught = tm = 0
    zones, lens = [], []
    for _ in range(ep.max_steps):
        a = ev_pol.predict(obs, deterministic=ev_det)[0]
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
    return {"esc": esc, "caught": caught, "tm": tm,
            "zone": float(np.mean(zones)) if zones else 0.0,
            "length": float(np.mean(lens)) if lens else 0.0}


if __name__ == "__main__":
    main()
