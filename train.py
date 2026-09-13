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
    ap.add_argument("--chaser-net", type=str, default=None,
                    help="comma-separated hidden widths for the chaser")
    ap.add_argument("--evader-net", type=str, default=None, help="e.g. 384,384")
    ap.add_argument("--evader-lr", type=float, default=None)
    ap.add_argument("--init-evader", type=str, default=None,
                    help="warm-start the evader from a state dict (.pt) or any "
                         "SB3 checkpoint (.zip), e.g. results/runN/evader.zip. "
                         "PPO then refines an already-competent policy instead "
                         "of a random init that escapes 0%% of hard arenas.")
    ap.add_argument("--r-hazard", type=float, default=None,
                    help="override the evader's spike-death reward. It defaults "
                         "to the SAME value as r_caught (-8), so the learner is "
                         "indifferent between dying on spikes and being caught "
                         "and has no reason to be spike-safe; learned evaders "
                         "die on spikes 4-5x as often as the scripted expert")
    ap.add_argument("--r-hazard-ch", type=float, default=None,
                    help="override the CHASER's self-impalement reward (the "
                         "suicide rate is the learned chaser's worst defect)")
    ap.add_argument("--init-chaser", type=str, default=None,
                    help="warm-start the chaser from a state-dict checkpoint "
                         "(e.g. a DAgger distillation of the scripted pursuer, "
                         "which unlike the learned chasers never kills itself "
                         "on spikes)")
    ap.add_argument("--bc-epochs", type=int, default=None,
                    help="behaviour-cloning epochs at startup (0 = skip; "
                         "defaults to 0 when --init-evader is given)")
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
    # warm starts must match the checkpoint's architecture: run21 is 384-wide
    # while the default is 256, and a mismatch only surfaces as a torch
    # "size mismatch" traceback mid-run
    def _net_from_ckpt(path):
        from stable_baselines3 import PPO as _PPO
        sd = _PPO.load(path, device="cpu").policy.state_dict()
        ws = [v.shape for k, v in sd.items()
              if k.startswith("mlp_extractor.policy_net.") and k.endswith(".weight")]
        if not ws:
            return None
        # every Linear in mlp_extractor.policy_net is a hidden layer (the
        # action head lives in action_net), so count them all: run21 has two
        # 384-wide layers and reports [384, 384]
        return [int(ws[0][0])] * len(ws)

    if args.init_evader and args.init_evader.endswith(".zip") and not args.evader_net:
        net = _net_from_ckpt(args.init_evader)
        if net:
            tp.evader_net = net
            print(f"[cfg] evader net {net} (from {args.init_evader})", flush=True)
    if args.init_chaser and args.init_chaser.endswith(".zip") and not getattr(args, "chaser_net", None):
        net = _net_from_ckpt(args.init_chaser)
        if net:
            tp.chaser_net = net
            print(f"[cfg] chaser net {net} (from {args.init_chaser})", flush=True)
    if args.chaser_net:
        tp.chaser_net = [int(x) for x in args.chaser_net.split(",")]
    if args.evader_lr:
        tp.evader_lr = args.evader_lr
    if args.init_chaser:
        tp.init_chaser = args.init_chaser
    if args.init_evader:
        tp.init_evader = args.init_evader
        if args.bc_epochs is None:
            tp.bc_epochs = 0        # the warm start replaces BC pretraining
    if args.gap_force_prob is not None:
        tp.gap_force_prob = args.gap_force_prob
    if args.gap_force is not None:
        tp.gap_force = args.gap_force

    mp = MoveParams()
    ep = EnvParams()
    if args.r_hazard is not None:
        ep.r_hazard = args.r_hazard
        print(f"[cfg] evader hazard-death reward {ep.r_hazard}", flush=True)
    if args.r_hazard_ch is not None:
        ep.r_hazard_ch = args.r_hazard_ch
        print(f"[cfg] chaser self-impalement reward {ep.r_hazard_ch}", flush=True)
    save_run_params(tp.out_dir, tp, mp, ep)   # persist the exact run config
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"== ori-park self-play | device={device} | blocks={tp.blocks} "
          f"steps/block={tp.block_steps} envs={tp.n_envs} out={tp.out_dir} ==", flush=True)
    run(tp, mp, ep, tp.out_dir, device=device)


if __name__ == "__main__":
    main()
