#!/usr/bin/env python3
"""The definitive skill metric: escape rate on hard arenas (mixed 0.7/0.85
uniform terrain params), stochastic evader, vs a FROZEN chaser that can be
deterministic (argmax) or stochastic (action sampling).

A deterministic frozen chaser is partly exploitable by chaos — a random
policy escapes ~37% of hard arenas just by being unpredictable. Evaluating
against a STOCHASTIC frozen chaser closes that hole: the chaser's own
noise removes the predictability the evader could otherwise lean on.

Usage:
  python hardarena.py --run results/run17 --matches 150 --chaser det
  python hardarena.py --run results/run17 --matches 150 --chaser stoch
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
from oripark.config import EnvParams, MoveParams, TrainParams  # noqa: E402
from oripark.env import OriArenaVecEnv  # noqa: E402
from oripark.selfplay import frozen_policy, make_ppo, set_nets_from_run  # noqa: E402


def load(path, role, tp, mp, ep):
    rng = np.random.default_rng(0)
    gen = ArenaGenerator(mp, rng, chaser_frac=ep.chaser_spawn_frac)
    env = OriArenaVecEnv(role, 1, FixedSampler(gen), mp, ep, seed=0)
    if path.endswith(".zip"):
        p = PPO.load(path, env=env, device="cpu")
    else:
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


class HardSampler:
    """Mixed uniform-hard arenas: 2/3 at params 0.7, 1/3 at 0.85, filtered
    by the scripted-expert oracle (an arena only counts if a competent
    player can actually win it) so the test measures skill, not luck of
    reachability."""

    def __init__(self, gen, n, mp, ep, seed0=5000, max_cand=600,
                 mix="mixed"):
        self.gen, self.n, self.mp, self.ep = gen, n, mp, ep
        self.arenas = []
        cands, params = [], []
        k = 0
        while len(cands) < max_cand:
            if mix == "mixed":
                p = 0.85 if (k % 3 == 2) else 0.7
            elif mix == "07":
                p = 0.7
            elif mix == "085":
                p = 0.85
            else:
                raise ValueError(mix)
            a = gen.generate(np.full(6, p), seed=seed0 + k)
            if a is not None:
                cands.append(a)
                params.append(p)
            k += 1
        self.arenas = expert_winnable(cands, mp, ep)
        print(f"  oracle: {len(self.arenas)}/{len(cands)} candidate arenas "
              f"winnable by the scripted expert", flush=True)
        if len(self.arenas) < n:
            raise SystemExit(f"only {len(self.arenas)} winnable arenas; "
                             f"lower --matches or raise max_cand")

    def sample(self, n):
        return [self.arenas[k % len(self.arenas)] for k in range(n)]


def expert_winnable(arenas, mp, ep, max_steps=None, batch=16):
    """Return the arenas the scripted evader escapes with NO chaser — the
    oracle for 'hard but fair'."""
    from oripark.scripted import ScriptedEvader
    max_steps = max_steps or ep.max_steps
    win = np.zeros(len(arenas), dtype=bool)
    for s in range(0, len(arenas), batch):
        chunk = arenas[s:s + batch]

        class Fix:
            def sample(self, m):
                return [chunk[k % len(chunk)] for k in range(m)]

        env = OriArenaVecEnv("evader", len(chunk), Fix(), mp, ep,
                             opponent=None, seed=0)
        scripts = [ScriptedEvader(mp, ep) for _ in range(len(chunk))]
        obs = env.reset()
        steps = np.zeros(len(chunk), dtype=np.int32)
        done_map = np.zeros(len(chunk), dtype=bool)
        while not done_map.all():
            acts = []
            for i in range(len(chunk)):
                if done_map[i]:
                    acts.append(np.zeros(5, dtype=np.int32))
                    continue
                ar = env.arenas[i]
                sp = scripts[i]
                if steps[i] == 0:
                    sp.reset(ar)
                phys = env.phys
                acts.append(sp.act(ar, phys.x[i], phys.y[i], phys.vx[i],
                                   phys.vy[i], phys.on_ground[i],
                                   phys.wall_dir[i], phys.can_djump[i],
                                   phys.can_dash[i], phys.dash_t[i],
                                   phys.bash_cd[i], phys.facing[i],
                                   chaser_x=phys.x[i], chaser_y=phys.y[i]))
            acts = np.stack(acts)
            obs, _, done, infos = env.step(acts)
            steps += 1
            for i in range(len(chunk)):
                if done[i] and not done_map[i]:
                    done_map[i] = True
                    win[s + i] = ("episode" in infos[i] and
                                  infos[i]["outcome"] == "escaped")
                elif steps[i] >= max_steps and not done_map[i]:
                    done_map[i] = True
        env.close()
    return [a for a, w in zip(arenas, win) if w]


def escape_rate(ev_pol, ch_pol, sampler, mp, ep, n_matches, chaser_stoch: bool,
                seed: int = 7):
    n_envs = min(n_matches, 32)
    ch = frozen_policy(ch_pol, deterministic=not chaser_stoch)
    env = OriArenaVecEnv("evader", n_envs, sampler, mp, ep, opponent=ch, seed=seed)
    obs = env.reset()
    esc = caught = n_done = 0
    lens = []
    for _ in range(ep.max_steps):
        a = ev_pol.predict(obs, deterministic=False)[0]
        obs, _, done, infos = env.step(a)
        for info in infos:
            if "episode" in info:
                o = info["outcome"]
                n_done += 1
                esc += o == "escaped"
                caught += o == "caught"
                lens.append(info["episode"]["l"])
        if n_done >= n_matches:
            break
    env.close()
    return esc / max(n_done, 1), esc, caught, n_done - esc - caught, \
        float(np.mean(lens)) if lens else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/run17")
    ap.add_argument("--matches", type=int, default=150)
    ap.add_argument("--chaser", choices=["det", "stoch"], default="det",
                    help="frozen chaser mode: argmax (det) or action sampling (stoch)")
    ap.add_argument("--set", choices=["mixed", "07", "085"], default="mixed",
                    help="arena difficulty mix (mixed = 2/3 x 0.7 + 1/3 x 0.85)")
    ap.add_argument("--entrants", choices=["trio", "all"], default="trio",
                    help="trio = random/BC/trained; all = every checkpoint too")
    ap.add_argument("--max-checkpoints", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tp, mp, ep = TrainParams(), MoveParams(), EnvParams()
    set_nets_from_run(tp, args.run)
    rng = np.random.default_rng(1)
    gen = ArenaGenerator(mp, rng, chaser_frac=ep.chaser_spawn_frac)
    sampler = HardSampler(gen, args.matches, mp, ep, mix=args.set)

    ch = load(os.path.join(args.run, "chaser.zip"), "chaser", tp, mp, ep)

    ents = []
    p0 = os.path.join(args.run, "pool_ev_0.pt")
    if os.path.exists(p0):
        ents.append(("random", p0))
    bi = os.path.join(args.run, "evader_init.pt")
    if os.path.exists(bi):
        ents.append(("BC-pretrained", bi))
    if args.entrants == "all":
        cks = sorted(glob.glob(os.path.join(args.run, "evader_b*.zip")))
        if len(cks) > args.max_checkpoints:
            idx = np.linspace(0, len(cks) - 1, args.max_checkpoints).round().astype(int)
            cks = [cks[i] for i in sorted(set(idx))]
        for c in cks:
            b = re.search(r"evader_b(\d+)\.zip", c).group(1)
            ents.append((f"checkpoint b{b}", c))
    fin = os.path.join(args.run, "evader.zip")
    if os.path.exists(fin):
        ents.append(("trained (best)", fin))

    mode = "stochastic chaser" if args.chaser == "stoch" else "deterministic chaser"
    print(f"hard-arena skill test: {args.matches} matches, stochastic evader, {mode}\n")
    rows = []
    for name, path in ents:
        ev = load(path, "evader", tp, mp, ep)
        rate, esc, caught, tm, length = escape_rate(
            ev.policy, ch.policy, sampler, mp, ep, args.matches,
            chaser_stoch=(args.chaser == "stoch"))
        rows.append((name, rate, esc, caught, tm, length))
        print(f"  {name:<18} escape {rate:.1%} ({esc}/{args.matches}) "
              f"caught {caught} tm {tm} len {length:.0f}", flush=True)

    md = [f"# Hard-arena skill test (`{args.run}`, {mode})",
          "",
          f"{args.matches} matches, stochastic evader, mixed 0.7/0.85 arenas, "
          f"frozen chaser (`{args.chaser}`).",
          "",
          "| policy | escape rate | caught | timeout | avg len |",
          "|---|---:|---:|---:|---:|"]
    for name, rate, esc, caught, tm, length in sorted(rows, key=lambda r: -r[1]):
        md.append(f"| {name} | {rate:.1%} ({esc}/{args.matches}) | {caught} | {tm} | {length:.0f} |")
    md += ["", "Caveat: a deterministic frozen chaser is partly exploitable by chaos;",
           "the stochastic-chaser number is the honest one."]
    out = args.out or os.path.join(args.run, f"hardarena_{args.chaser}.md")
    with open(out, "w") as f:
        f.write("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
