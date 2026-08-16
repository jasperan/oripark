#!/usr/bin/env python3
"""Select the best evader checkpoint by escape rate on the fixed eval set.

Self-play is non-stationary (the chaser adapts), so the final block is not
necessarily the strongest evader. This evaluates every saved checkpoint on
the SAME fixed arena set vs the frozen final chaser and copies the best to
evader.zip (and reports the runner-up).
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from oripark.arena import ArenaGenerator  # noqa: E402
from oripark.config import (EnvParams, MoveParams, TrainParams,
                             load_run_params)  # noqa: E402
from oripark.env import OriArenaVecEnv  # noqa: E402
from oripark.selfplay import frozen_policy, make_ppo, set_nets_from_run  # noqa: E402


def load(path, role, tp, mp, ep):
    rng = np.random.default_rng(0)
    gen = ArenaGenerator(mp, rng, chaser_frac=ep.chaser_spawn_frac)
    env = OriArenaVecEnv(role, 1, FixedSampler(gen), mp, ep, seed=0)
    p = PPO.load(path, env=env, device="cpu") if path.endswith(".zip") else None
    if p is None:
        p = make_ppo(env, tp, 0, "cpu", side=role)
        p.policy.load_state_dict(torch.load(path, map_location="cpu"))
    env.close()
    return p


import torch  # noqa: E402


class FixedSampler:
    def __init__(self, gen):
        self.gen = gen

    def sample(self, n):
        return [self.gen.generate(np.full(6, 0.5), seed=0) for _ in range(n)]


def eval_on(ev_pol, ch_pol, arenas, mp, ep, n_envs, seed=7):
    from oripark.selfplay import frozen_policy

    class Fix:
        def __init__(self, arenas):
            self.arenas = arenas

        def sample(self, n):
            return [self.arenas[k % len(self.arenas)] for k in range(n)]

    env = OriArenaVecEnv("evader", n_envs, Fix(arenas), mp, ep,
                         opponent=frozen_policy(ch_pol, deterministic=True), seed=seed)
    obs = env.reset()
    esc = caught = 0
    n_done = 0
    for _ in range(ep.max_steps):
        a = ev_pol.predict(obs, deterministic=False)[0]
        obs, _, done, infos = env.step(a)
        for info in infos:
            if "episode" in info:
                o = info["outcome"]
                n_done += 1
                esc += o == "escaped"
                caught += o == "caught"
        if n_done >= n_envs:
            break
    env.close()
    return esc / max(n_done, 1), esc, caught, n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/run13")
    ap.add_argument("--matches", type=int, default=24)
    ap.add_argument("--arena-seeds", type=int, default=24)
    ap.add_argument("--matches-final", type=int, default=40,
                    help="matches for the chosen winner (tighter estimate)")
    args = ap.parse_args()

    tp, mp, ep = load_run_params(args.run)
    set_nets_from_run(tp, args.run)
    rng = np.random.default_rng(1)
    gen = ArenaGenerator(mp, rng, chaser_frac=ep.chaser_spawn_frac)
    mu = np.load(os.path.join(args.run, "adv_mu.npy")) if os.path.exists(
        os.path.join(args.run, "adv_mu.npy")) else np.full(6, 0.5)
    sig = np.load(os.path.join(args.run, "adv_sigma.npy")) if os.path.exists(
        os.path.join(args.run, "adv_sigma.npy")) else np.full(6, 0.3)
    arng = np.random.default_rng(4242)
    arenas = [gen.generate(np.clip(mu + sig * arng.standard_normal(6), 0, 1), seed=1000 + k)
              for k in range(args.arena_seeds)]

    ch = load(os.path.join(args.run, "chaser.zip"), "chaser", tp, mp, ep)

    ckpts = []
    for p in sorted(glob.glob(os.path.join(args.run, "evader_b*.zip"))):
        m = re.search(r"evader_b(\d+)\.zip", p)
        ckpts.append((int(m.group(1)), p))
    # include the final block
    for p in (os.path.join(args.run, "evader.zip"),):
        if os.path.exists(p) and all(c[1] != p for c in ckpts):
            ckpts.append((10 ** 9, p))
    ckpts.sort(key=lambda c: c[0])

    print(f"evaluating {len(ckpts)} checkpoints ({args.matches} matches each) vs frozen chaser...")
    results = []
    for block, path in ckpts:
        ev = load(path, "evader", tp, mp, ep)
        rate, esc, caught, n = eval_on(ev.policy, ch.policy, arenas, mp, ep,
                                       n_envs=min(args.matches, 32))
        results.append((block, rate, esc, caught, n))
        print(f"  block {block:>4}: escape {rate:.0%} ({esc}/{n}) caught {caught}", flush=True)

    results.sort(key=lambda r: -r[1])
    best_block, best_rate, best_esc, best_caught, best_n = results[0]
    best_path = [p for b, p in ckpts if b == best_block][0]
    out = os.path.join(args.run, "evader.zip")
    import shutil
    shutil.copy(best_path, out)
    print(f"\nBEST: block {best_block} ({best_rate:.0%} escape, {best_esc}/{best_n}, "
          f"caught {best_caught}) -> {out}")

    # tighter estimate on the winner
    ev = load(best_path, "evader", tp, mp, ep)
    rate, esc, caught, n = eval_on(ev.policy, ch.policy, arenas, mp, ep,
                                   n_envs=min(args.matches_final, 32))
    print(f"winner re-eval ({args.matches_final} matches): escape {rate:.0%} ({esc}/{n}), "
          f"caught {caught}")
    with open(os.path.join(args.run, "selected.txt"), "w") as f:
        f.write(f"best checkpoint: block {best_block}\n"
                f"escape rate (24 matches): {best_rate:.0%} ({best_esc}/{best_n})\n"
                f"final re-eval: {rate:.0%} ({esc}/{n}), caught {caught}\n")


if __name__ == "__main__":
    main()
