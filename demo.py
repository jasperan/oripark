#!/usr/bin/env python3
"""Render demo replays of trained (or untrained) policies as GIFs.

Usage:
  python demo.py --run results/run1 --mode both --arena-seed 42 --out demo.gif
  python demo.py --run results/quick --mode after --ascii
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from oripark.adversaries import StaticSampler  # noqa: E402
from oripark.arena import ArenaGenerator  # noqa: E402
from oripark.config import EnvParams, MoveParams, TrainParams  # noqa: E402
from oripark.render import ReplayRecorder, ascii_frames, make_gif  # noqa: E402
from oripark.selfplay import make_ppo  # noqa: E402


def load_policy(pt_path: str, role: str, tp: TrainParams, mp: MoveParams,
                ep: EnvParams, device: str):
    from stable_baselines3 import PPO
    from oripark.adversaries import StaticSampler
    from oripark.arena import ArenaGenerator
    from oripark.env import OriArenaVecEnv
    rng = np.random.default_rng(0)
    gen = ArenaGenerator(mp, rng)
    env = OriArenaVecEnv(role, 1, StaticSampler(gen, np.full(6, 0.5), rng), mp, ep, seed=0)
    if pt_path.endswith(".zip"):
        ppo = PPO.load(pt_path, env=env, device=device)
        env.close()
        return ppo
    ppo = make_ppo(env, tp, 0, device)
    sd = torch.load(pt_path, map_location=device)
    ppo.policy.load_state_dict(sd)
    env.close()
    return ppo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/run1")
    ap.add_argument("--mode", choices=["before", "after", "both"], default="both")
    ap.add_argument("--arena-seed", type=int, default=4242)
    ap.add_argument("--out", default="demo.gif")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--ascii", action="store_true", help="also print an ASCII replay")
    args = ap.parse_args()

    tp = TrainParams()
    mp = MoveParams()
    ep = EnvParams()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(0)
    gen = ArenaGenerator(mp, rng)
    if os.path.exists(os.path.join(args.run, "adv_mu.npy")):
        mu = np.load(os.path.join(args.run, "adv_mu.npy"))
    else:
        mu = np.full(6, 0.5)
    arena = gen.generate(mu, seed=args.arena_seed)

    # load policies
    ch_path = (os.path.join(args.run, "chaser.zip")
               if os.path.exists(os.path.join(args.run, "chaser.zip"))
               else os.path.join(args.run, "pool_ch_0.pt"))
    ch_after = load_policy(ch_path, "chaser", tp, mp, ep, device)
    ev_before = load_policy(os.path.join(args.run, "pool_ev_0.pt"), "evader", tp, mp, ep, device)
    ev_after = load_policy(os.path.join(args.run, "evader.zip"), "evader", tp, mp, ep, device)

    clips = []
    if args.mode in ("before", "both"):
        rec = ReplayRecorder(arena, ev_before, ch_after, mp, ep, seed=7)
        fr, out = rec.run(args.max_steps)
        rec.close()
        print(f"before: outcome={out} len={len(fr)}")
        clips.append((fr, out, "UNTRAINED (block 0)"))
        if args.ascii:
            print(ascii_frames(fr, arena, mp))
    if args.mode in ("after", "both"):
        rec = ReplayRecorder(arena, ev_after, ch_after, mp, ep, seed=7)
        fr, out = rec.run(args.max_steps)
        rec.close()
        print(f"after : outcome={out} len={len(fr)}")
        clips.append((fr, out, "TRAINED (final)"))
        if args.ascii:
            print(ascii_frames(fr, arena, mp))

    stem = os.path.splitext(args.out)[0]
    paths = []
    for fr, out, label in clips:
        p = make_gif(fr, arena, mp, f"{stem}_{label.split()[0].lower()}_{out}.gif",
                     label=label, scale=args.scale, fps=args.fps)
        paths.append(p)
        print(f"wrote {p}")
    if len(clips) == 2:
        # vertical side-by-side comparison GIF
        from PIL import Image
        imgs = []
        for fr, out, label in clips:
            sel = fr[::2]
            # reuse make_gif temp frames
            p = f"/tmp/ori_frame_{label.split()[0].lower()}.gif"
            make_gif(fr, arena, mp, p, label=label, scale=args.scale, fps=args.fps)
            im = Image.open(p)
            imgs.append(list(iter_frames(im)))
        h = max(len(a) for a in imgs)
        out_frames = []
        for k in range(h):
            a = imgs[0][min(k, len(imgs[0]) - 1)]
            b = imgs[1][min(k, len(imgs[1]) - 1)]
            canvas = Image.new("RGB", (a.width * 2, a.height))
            canvas.paste(a, (0, 0))
            canvas.paste(b, (a.width, 0))
            out_frames.append(canvas)
        out_frames[0].save(stem + "_sidebyside.gif", save_all=True,
                           append_images=out_frames[1:], duration=int(1000 / args.fps), loop=0)
        print(f"wrote {stem}_sidebyside.gif")


def iter_frames(im):
    try:
        i = 0
        while True:
            im.seek(i)
            yield im.convert("RGB")
            i += 1
    except EOFError:
        pass


if __name__ == "__main__":
    main()
