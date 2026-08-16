#!/usr/bin/env python3
"""Train the Ori tag-game self-play league.

Runs alternating PPO blocks: evader (full Ori kit) vs a snapshot from the
chaser league pool, then chaser vs a snapshot from the evader pool, with a
CEM terrain adversary keeping the curriculum at the edge of skill.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from oripark.config import (EnvParams, MoveParams, TrainParams,
                             save_run_params)  # noqa: E402
from oripark.selfplay import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Ori tag self-play training")
    ap.add_argument("--blocks", type=int, default=None, help="number of self-play blocks")
    ap.add_argument("--steps", type=int, default=None, help="timesteps per agent per block")
    ap.add_argument("--workers", type=int, default=None, help="parallel envs")
    ap.add_argument("--out", type=str, default=None, help="output directory")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default=None, help="cpu or cuda (default: cuda if available)")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test config")
    ap.add_argument("--evader-net", type=str, default=None, help="e.g. 384,384")
    ap.add_argument("--evader-lr", type=float, default=None)
    ap.add_argument("--gap-force-prob", type=float, default=None,
                    help="fraction of training arenas with gap_scale forced high")
    ap.add_argument("--gap-force", type=float, default=None,
                    help="forced gap_scale value for sprinkled arenas")
    args = ap.parse_args()

    tp = TrainParams()
    if args.quick:
        tp.blocks = 6
        tp.n_envs = 8
        tp.block_steps = 2048
        tp.eval_matches = 8
        tp.eval_ep_len = 480
        tp.adv_pop = 4
        tp.adv_matches = 2
        tp.pool_size = 4
        tp.out_dir = "results/quick"
    if args.blocks:
        tp.blocks = args.blocks
    if args.steps:
        tp.block_steps = args.steps
    if args.workers:
        tp.n_envs = args.workers
    if args.seed is not None:
        tp.seed = args.seed
    if args.out:
        tp.out_dir = args.out
    if args.evader_net:
        tp.evader_net = [int(x) for x in args.evader_net.split(",")]
    if args.evader_lr:
        tp.evader_lr = args.evader_lr
    if args.gap_force_prob is not None:
        tp.gap_force_prob = args.gap_force_prob
    if args.gap_force is not None:
        tp.gap_force = args.gap_force

    mp = MoveParams()
    ep = EnvParams()
    save_run_params(tp.out_dir, tp, mp, ep)   # persist the exact run config
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"== ori-park self-play | device={device} | blocks={tp.blocks} "
          f"steps/block={tp.block_steps} envs={tp.n_envs} out={tp.out_dir} ==", flush=True)
    run(tp, mp, ep, tp.out_dir, device=device)


if __name__ == "__main__":
    main()
