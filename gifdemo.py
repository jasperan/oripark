#!/usr/bin/env python3
"""Hero demo GIFs: scan arena seeds for a stochastic escape and record frames.

The trained-evader clip IS the scan (same env, same arena, same RNG), so the
recorded frames are exactly the escaping run. The untrained clip is a fresh
seeded replay of the untrained policy on the same arena.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from oripark.arena import ArenaGenerator  # noqa: E402
from oripark.config import (EnvParams, MoveParams, TrainParams,
                             load_run_params)  # noqa: E402
from oripark.env import OriArenaVecEnv  # noqa: E402
from oripark.selfplay import frozen_policy, make_ppo, set_nets_from_run  # noqa: E402
from oripark.render import ReplayRecorder, make_gif  # noqa: E402


def load(path, role, tp, mp, ep, mu):
    from stable_baselines3 import PPO
    rng = np.random.default_rng(0)
    gen = ArenaGenerator(mp, rng, chaser_frac=ep.chaser_spawn_frac)
    env = OriArenaVecEnv(role, 1, Dummy(gen, mu), mp, ep, seed=0)
    if path.endswith(".zip"):
        p = PPO.load(path, env=env, device="cpu")
    else:
        p = make_ppo(env, tp, 0, "cpu", side=role)
        p.policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    env.close()
    return p, gen


class Dummy:
    def __init__(self, gen, mu):
        self.gen, self.mu = gen, mu

    def sample(self, n):
        return [self.gen.generate(self.mu, seed=0) for _ in range(n)]

    def mean_params(self):
        return self.mu


class One:
    def __init__(self, a):
        self.a = a

    def sample(self, n):
        return [self.a]


def record(arena, ev_pol, ch_pol, mp, ep, max_steps):
    """Record a full stochastic replay; returns (frames, outcome, length).

    Thin wrapper over oripark.render.ReplayRecorder so the GIF tools and the
    renderer cannot drift apart. (This file used to carry its own copy of the
    loop, which appended the frame AFTER the env's auto-reset — every clip
    therefore ended on the respawned agent instead of the escape.)
    """
    rec = ReplayRecorder(arena, ev_pol, ch_pol, mp, ep, deterministic=False, seed=7)
    frames, outcome = rec.run(max_steps)
    rec.close()
    return frames, outcome, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/run14")
    ap.add_argument("--out", default="results/hero.gif")
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--seed0", type=int, default=4242)
    ap.add_argument("--scan", type=int, default=120)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--params", type=float, default=None,
                    help="uniform arena params (e.g. 0.7) instead of adv_mu")
    args = ap.parse_args()

    tp, mp, ep = load_run_params(args.run)
    set_nets_from_run(tp, args.run)
    mu = (np.full(6, args.params) if args.params is not None
          else np.load(os.path.join(args.run, "adv_mu.npy")) if os.path.exists(
        os.path.join(args.run, "adv_mu.npy")) else np.full(6, 0.5))

    ch, gen = load(os.path.join(args.run, "chaser.zip"), "chaser", tp, mp, ep, mu)
    ev, _ = load(os.path.join(args.run, "evader.zip"), "evader", tp, mp, ep, mu)
    # the "before" policy: the true pre-training checkpoint, not pool_ev_0.pt
    # (which is an early TRAINED pool snapshot -- see oripark/baselines.py)
    from oripark.baselines import find_baseline
    base_path, base_label = find_baseline(args.run)
    if base_path is None:
        base_path, base_label = os.path.join(args.run, "evader_init.pt"), "init"
    base, _ = load(base_path, "evader", tp, mp, ep, mu)
    print(f"before-panel policy: {base_label} ({base_path})", flush=True)

    # ---- scan candidates with the trained evader, recording frames
    chosen_frames = None
    for k in range(args.scan):
        torch.manual_seed(1000 + k)
        np.random.seed(1000 + k)
        a = gen.generate(mu, seed=args.seed0 + k)
        fr, out, ln = record(a, ev.policy, ch.policy, mp, ep, args.max_steps)
        print(f"  seed {args.seed0 + k}: {out} (len {ln})", flush=True)
        if out == "escaped":
            chosen_frames = (a, fr, out, ln, 1000 + k)
            break
    if chosen_frames is None:
        print("no escape found; using first arena")
        torch.manual_seed(1000)
        np.random.seed(1000)
        a = gen.generate(mu, seed=args.seed0)
        fr, out, ln = record(a, ev.policy, ch.policy, mp, ep, args.max_steps)
        chosen_frames = (a, fr, out, ln, 1000)
    a, tr_fr, tr_out, tr_ln, seed_rng = chosen_frames
    print(f"hero arena seed {args.seed0 + (seed_rng - 1000)}: trained {tr_out} in {tr_ln} steps")

    # ---- untrained replay on the same arena (fresh, separate RNG)
    torch.manual_seed(2000)
    np.random.seed(2000)
    un_fr, un_out, un_ln = record(a, base.policy, ch.policy, mp, ep, args.max_steps)
    print(f"before: {un_out} (len {un_ln})")

    stem = os.path.splitext(args.out)[0]
    p1 = f"{stem}_untrained_{un_out}.gif"
    p2 = f"{stem}_trained_{tr_out}.gif"
    make_gif(un_fr, a, mp, p1, label=f"BEFORE ({base_label})", scale=args.scale,
             fps=args.fps)
    make_gif(tr_fr, a, mp, p2, label="AFTER (trained)", scale=args.scale, fps=args.fps)
    print(f"wrote {p1}\nwrote {p2}")

    # side-by-side
    from PIL import Image
    from demo import iter_frames
    imgs = []
    for p, fr in ((p1, un_fr), (p2, tr_fr)):
        im = Image.open(p)
        imgs.append(list(iter_frames(im)))
    h = max(len(x) for x in imgs)
    out_frames = []
    for k in range(h):
        left = imgs[0][min(k, len(imgs[0]) - 1)]
        right = imgs[1][min(k, len(imgs[1]) - 1)]
        canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        out_frames.append(canvas)
    sbs = f"{stem}_sidebyside.gif"
    out_frames[0].save(sbs, save_all=True, append_images=out_frames[1:],
                       duration=1000 // args.fps, loop=0)
    print(f"wrote {sbs}")


if __name__ == "__main__":
    main()
