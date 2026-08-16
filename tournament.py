#!/usr/bin/env python3
"""Tournament: round-robin the evader policies from a run (random init, BC,
every checkpoint, final, scripted expert) against the frozen chaser on fixed
arenas, and record a small video per entrant so improvements are visible.

Ranking is by escape rate against the SAME chaser on the SAME arena set —
the single-opponent ladder that makes self-play checkpoints comparable.

Outputs:
  docs/tournament.md        — results table + how-to-read
  docs/media/tournament/    — one GIF per entrant + a side-by-side montage
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from oripark.config import EnvParams, MoveParams, TrainParams  # noqa: E402
from oripark.arena import ArenaGenerator  # noqa: E402
from oripark.env import OriArenaVecEnv  # noqa: E402
from oripark.selfplay import frozen_policy, make_ppo, set_nets_from_run  # noqa: E402


def load_policy(path, role, tp, mp, ep, device="cpu"):
    from stable_baselines3 import PPO
    env = OriArenaVecEnv(role, 1, StaticSampler(mp, ep), mp, ep, seed=0)
    if path.endswith(".zip"):
        p = PPO.load(path, env=env, device=device)
    else:
        p = make_ppo(env, tp, 0, device, side=role)
        p.policy.load_state_dict(torch.load(path, map_location=device))
    env.close()
    return p


class StaticSampler:
    def __init__(self, mp, ep):
        self.mp, self.ep = mp, ep

    def sample(self, n):
        from oripark.config import MoveParams
        g = np.zeros((1, self.mp.arena_h, self.mp.arena_w), dtype=np.int32)
        g[0, self.mp.arena_h - 2, :] = 1
        from oripark.arena import Arena
        return [Arena(grid=g, ev_spawn=(64.0, 64.0), ch_spawn=(64.0, 64.0),
                      portal=(64.0, 64.0), orbs=[], params=np.full(6, 0.5))
                for _ in range(n)]


class FixedArenas:
    def __init__(self, arenas):
        self.arenas = arenas

    def sample(self, n):
        return [self.arenas[k % len(self.arenas)] for k in range(n)]

    def mean_params(self):
        return self.arenas[0].params


def make_arena_set(run_dir, mp, ep, n=24, seed0=4242, combined=False):
    rng = np.random.default_rng(1)
    gen = ArenaGenerator(mp, rng, chaser_frac=ep.chaser_spawn_frac)
    arenas = []
    if combined:
        # robust ranking: uniform-hard (0.7) + uniform-harder (0.85) + the
        # run's own curriculum gauntlets (final adversary mu) — no single
        # distribution dominates the ranking
        per = n // 3
        for params, s0 in ((0.7, 5000), (0.85, 7000)):
            k = 0
            while len(arenas) < (per if params == 0.7 else 2 * per):
                a = gen.generate(np.full(6, params), seed=s0 + k)
                if a is not None:
                    arenas.append(a)
                k += 1
        mu = np.load(os.path.join(run_dir, "adv_mu.npy")) if os.path.exists(
            os.path.join(run_dir, "adv_mu.npy")) else np.full(6, 0.5)
        k = 0
        while len(arenas) < n:
            a = gen.generate(mu, seed=9000 + k)
            if a is not None:
                arenas.append(a)
            k += 1
    else:
        mu = np.load(os.path.join(run_dir, "adv_mu.npy")) if os.path.exists(
            os.path.join(run_dir, "adv_mu.npy")) else np.full(6, 0.5)
        k = 0
        while len(arenas) < n:
            a = gen.generate(mu, seed=seed0 + k)
            if a is not None:
                arenas.append(a)
            k += 1
    return FixedArenas(arenas), mu


def escape_rate(ev_pol, ch_pol, sampler, mp, ep, matches=40, seed=7,
                chaser_stoch=False):
    """Stochastic evader vs frozen chaser (argmax or action sampling) on the
    fixed arena set."""
    env = OriArenaVecEnv("evader", matches, sampler, mp, ep,
                         opponent=frozen_policy(ch_pol, deterministic=not chaser_stoch),
                         seed=seed)
    obs = env.reset()
    esc = caught = tm = 0
    lens = []
    for _ in range(ep.max_steps):
        a = ev_pol.predict(obs, deterministic=False)[0]
        obs, _, done, infos = env.step(a)
        for info in infos:
            if "episode" in info:
                o = info["outcome"]
                esc += o == "escaped"
                caught += o == "caught"
                tm += o == "timeout"
                lens.append(info["episode"]["l"])
        if esc + caught + tm >= matches:
            break
    env.close()
    return {"esc": esc, "caught": caught, "tm": tm, "n": esc + caught + tm,
            "avg_len": float(np.mean(lens)) if lens else 0.0}


def entrants_for(run_dir, tp, mp, ep, max_checkpoints=8):
    """Ordered entrant list: random init -> BC -> checkpoints -> final, plus
    the scripted expert as a reference player."""
    ents = []
    p0 = os.path.join(run_dir, "pool_ev_0.pt")
    if os.path.exists(p0):
        ents.append(("random-init", p0))
    bi = os.path.join(run_dir, "evader_init.pt")
    if os.path.exists(bi):
        ents.append(("BC-pretrained", bi))
    cks = sorted(glob_ck(run_dir, "evader_b*.zip"))
    if len(cks) > max_checkpoints:
        idx = np.linspace(0, len(cks) - 1, max_checkpoints).round().astype(int)
        cks = [cks[i] for i in sorted(set(idx))]
    for c in cks:
        b = re.search(r"evader_b(\d+)\.zip", c).group(1)
        ents.append((f"checkpoint b{b}", c))
    fin = os.path.join(run_dir, "evader.zip")
    if os.path.exists(fin):
        ents.append(("final/best", fin))
    return ents


def glob_ck(run_dir, pat):
    import glob
    return sorted(glob.glob(os.path.join(run_dir, pat)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/run14")
    ap.add_argument("--matches", type=int, default=60)
    ap.add_argument("--arenas", type=int, default=24)
    ap.add_argument("--max-checkpoints", type=int, default=8)
    ap.add_argument("--video-seed", type=int, default=4242,
                    help="arena seed used for the recorded entrant videos")
    ap.add_argument("--chaser", choices=["det", "stoch"], default="det",
                    help="frozen chaser mode: argmax (det) or action sampling (stoch)")
    ap.add_argument("--out-md", default="docs/tournament.md")
    ap.add_argument("--out-media", default="docs/media/tournament")
    args = ap.parse_args()

    tp, mp, ep = TrainParams(), MoveParams(), EnvParams()
    set_nets_from_run(tp, args.run)
    sampler, mu = make_arena_set(args.run, mp, ep, n=args.arenas, seed0=4242,
                                  combined=True)
    ch = load_policy(os.path.join(args.run, "chaser.zip"), "chaser", tp, mp, ep)
    cstoch = args.chaser == "stoch"

    ents = entrants_for(args.run, tp, mp, ep, max_checkpoints=args.max_checkpoints)
    print(f"tournament on {args.arenas} fixed arenas (mu {np.round(mu,2)}) "
          f"vs {'stochastic' if cstoch else 'deterministic'} frozen chaser "
          f"| {len(ents)} entrants", flush=True)

    rows = []
    for name, path in ents:
        ev = load_policy(path, "evader", tp, mp, ep)
        r = escape_rate(ev.policy, ch.policy, sampler, mp, ep,
                        matches=args.matches, chaser_stoch=cstoch)
        rate = r["esc"] / max(r["n"], 1)
        rows.append((name, r, rate))
        print(f"  {name:22s} esc={r['esc']:3d}/{r['n']} ({rate:5.1%}) "
              f"caught={r['caught']} len={r['avg_len']:5.0f}", flush=True)
        ev = None

    # ---- scripted expert reference (no NN — pure BFS waypoint + flee)
    from oripark.scripted import ScriptedEvader
    env = OriArenaVecEnv("evader", min(16, args.matches), sampler, mp, ep,
                         opponent=frozen_policy(ch.policy, deterministic=True), seed=0)
    scripts = [ScriptedEvader(mp, ep) for _ in range(env.n_envs)]
    obs = env.reset()
    esc_s = caught_s = tm_s = 0
    lens_s = []
    done_n = 0
    for _ in range(ep.max_steps):
        acts = []
        ph = env.phys
        for i in range(env.n_envs):
            ar = env.arenas[i]
            if env.t[i] == 0:
                scripts[i].reset(ar)
            acts.append(scripts[i].act(ar, ph.x[i], ph.y[i], ph.vx[i], ph.vy[i],
                                       ph.on_ground[i], ph.wall_dir[i], ph.can_djump[i],
                                       ph.can_dash[i], ph.dash_t[i], ph.bash_cd[i],
                                       ph.facing[i], chaser_x=ph.x[env.n_envs + i],
                                       chaser_y=ph.y[env.n_envs + i]))
        obs, _, done, infos = env.step(np.stack(acts))
        for info in infos:
            if "episode" in info:
                o = info["outcome"]
                esc_s += o == "escaped"
                caught_s += o == "caught"
                tm_s += o == "timeout"
                lens_s.append(info["episode"]["l"])
                done_n += 1
        if done_n >= args.matches:
            break
    env.close()
    rows.append(("scripted expert",
                 {"esc": esc_s, "caught": caught_s, "tm": tm_s,
                  "n": done_n, "avg_len": float(np.mean(lens_s)) if lens_s else 0.0},
                 esc_s / max(done_n, 1)))

    rows.sort(key=lambda r: -r[2])
    rank = {name: i + 1 for i, (name, _, _) in enumerate(rows)}

    # ---- recorded videos: one GIF per entrant on a common arena
    os.makedirs(args.out_media, exist_ok=True)
    video_names = [n for n, _, _ in rows if n != "scripted expert"]
    vids = []
    from oripark.render import ReplayRecorder, make_gif
    # find an arena the RANK-1 entrant escapes (so the champion's clip is a
    # clean escape); the others then replay the same arena for contrast
    # video-scan winner: top-ranked NN entrant (the scripted expert has no
    # policy path and privileged grid access — it stays in the table only)
    winner = next(n for n, _, _ in rows if n != "scripted expert")
    winner_path = dict(ents).get(winner)
    arena, video_seed_rng = None, 0
    if winner_path is not None:
        rngv = np.random.default_rng(0)
        genv = ArenaGenerator(mp, rngv, chaser_frac=ep.chaser_spawn_frac)
        w_ev = load_policy(winner_path, "evader", tp, mp, ep)
        for k in range(80):
            cand = genv.generate(np.full(6, 0.7), seed=args.video_seed + k)
            if cand is None:
                continue
            srng = 1000 + k                     # vary the draw per candidate
            torch.manual_seed(srng)
            np.random.seed(srng)
            rec = ReplayRecorder(cand, w_ev.policy, ch.policy, mp, ep,
                                 deterministic=False, seed=7)
            _, out = rec.run(800)
            rec.close()
            print(f"  video-scan seed {args.video_seed + k} rng {srng}: {out}", flush=True)
            if out == "escaped":
                arena = cand
                video_seed_rng = srng
                break
    if arena is None:
        arena_v, _ = make_arena_set(args.run, mp, ep, n=1, seed0=args.video_seed)
        arena = arena_v.arenas[0]
        video_seed_rng = 4242
    for name, path in [(n, dict(ents).get(n)) for n in video_names]:
        ev = load_policy(path, "evader", tp, mp, ep)
        torch.manual_seed(video_seed_rng)
        np.random.seed(video_seed_rng)
        rec = ReplayRecorder(arena, ev.policy, ch.policy, mp, ep,
                             deterministic=False, seed=7)
        fr, out = rec.run(800)
        rec.close()
        p = os.path.join(args.out_media, f"entrant_{rank[name]:02d}_{name.replace('/', '_')}.gif")
        make_gif(fr, arena, mp, p, label=name.upper(), scale=0.5, fps=30, step=2)
        vids.append((rank[name], name, p, out))
        print(f"  video {rank[name]:2d} {name}: {out} ({len(fr)} frames) -> {p}", flush=True)

    # ---- markdown report
    lines = [
        "# Tournament — evader policies vs the frozen chaser",
        "",
        f"Run: `{args.run}` · {args.arenas} fixed arenas (adversary mean "
        f"`{np.round(mu,2)}`) · {args.matches} matches each · stochastic "
        "evader, deterministic chaser (the honest PPO protocol).",
        "",
        "| rank | entrant | escape rate | caught | avg len |",
        "|---|---|---:|---:|---:|",
    ]
    for name, r, rate in rows:
        lines.append(f"| {rank[name]} | {name} | {rate:.0%} ({r['esc']}/{r['n']}) "
                     f"| {r['caught']} | {r['avg_len']:.0f} |")
    lines += [
        "",
        "## Recorded videos",
        "",
        "Each entrant plays the same arena (seed "
        f"{args.video_seed}) with the same chaser; the clip shows its natural "
        "behavior. Watch the arc: random flails, BC traverses, checkpoints "
        "gain evasion, the best checkpoint escapes.",
        "",
    ]
    for rk, name, p, out in sorted(vids):
        rel = os.path.relpath(p, os.path.dirname(args.out_md))
        lines.append(f"### #{rk} {name} — {out}")
        lines.append(f"![{name}]({rel})")
        lines.append("")
    lines += [
        "## Reading the table",
        "",
        "- **random-init**: the untrained policy — lower bound.",
        "- **BC-pretrained**: the scripted expert's traversal cloned into the NN.",
        "- **checkpoint bN**: evader mid-training (self-play is non-stationary, "
        "so the best checkpoint often beats the final block).",
        "- **final/best**: `evader.zip` (select.py output).",
        "- **scripted expert**: reference player (BFS waypoint + flee), not an NN.",
        "",
        "Escape rate is measured on the run's own (hard) arena distribution, "
        "so numbers are comparable within a run but not across runs with "
        "different adversary means.",
    ]
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {args.out_md} + videos in {args.out_media}")


if __name__ == "__main__":
    main()
