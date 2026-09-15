#!/usr/bin/env python3
"""Post-mortem on spike deaths: was the fatal spike VISIBLE in the obs patch?

Motivation (E25): learned policies die on spikes 4-11x as often as the scripted
expert (42-112 per ~300 episodes vs ~10), and a terminal penalty made it worse.
Two very different root causes are possible:
  (a) perception - the policy cannot see the spike that kills it (patch too
      small for a 12 tiles/s fall), or
  (b) control/credit  - the spike was in view, the policy just mis-handled it.
This measures which, and reports the lead time (how many steps before death the
fatal cell first appeared inside the patch).
"""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from stable_baselines3 import PPO
from oripark.arena import ArenaGenerator
from oripark.config import EnvParams, MoveParams, load_run_params
from oripark.env import OriArenaVecEnv
from oripark.selfplay import frozen_policy, make_ppo, set_nets_from_run
from oripark.physics import SOLID, SPIKE
sys.path.insert(0, ".")
from hardarena import HardSampler  # noqa: E402


def load_ev(path, role, tp, mp, ep):
    rng = np.random.default_rng(0)
    gen = ArenaGenerator(mp, rng, chaser_frac=ep.chaser_spawn_frac)

    class Fix:
        def sample(self, n):
            return [gen.generate(np.full(6, 0.7), seed=0) for _ in range(n)]

    env = OriArenaVecEnv(role, 1, Fix(), mp, ep, seed=0)
    if path.endswith(".zip"):
        p = PPO.load(path, env=env, device="cpu")
    else:
        p = make_ppo(env, tp, 0, "cpu", side=role)
        p.policy.load_state_dict(torch_load(path))
    env.close()
    return p


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evader", required=True)
    ap.add_argument("--run", default="results/run21_fixed")
    ap.add_argument("--chaser-run", default="results/run21_fixed")
    ap.add_argument("--matches", type=int, default=300)
    ap.add_argument("--n-envs", type=int, default=24)
    ap.add_argument("--arena-seed0", type=int, default=9000)
    ap.add_argument("--history", type=int, default=24)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    tp, mp, ep = load_run_params(args.run)
    set_nets_from_run(tp, args.run)
    rng = np.random.default_rng(1)
    gen = ArenaGenerator(mp, rng, chaser_frac=ep.chaser_spawn_frac)
    sampler = HardSampler(gen, args.matches, mp, ep, mix="07", oracle="calm",
                          seed0=args.arena_seed0, max_cand=900)
    ctp, cmp_, cep = load_run_params(args.chaser_run)
    set_nets_from_run(ctp, args.chaser_run)
    cgen = ArenaGenerator(mp, np.random.default_rng(0), chaser_frac=cep.chaser_spawn_frac)

    class FixC:
        def sample(self, n):
            return [cgen.generate(np.full(6, 0.7), seed=0) for _ in range(n)]

    probe = OriArenaVecEnv("chaser", 1, FixC(), mp, cep, seed=0)
    ch = PPO.load(os.path.join(args.chaser_run, "chaser.zip"), env=probe, device="cpu")
    probe.close()

    ev = load_ev(args.evader, "evader", tp, mp, ep)
    N = args.n_envs
    env = OriArenaVecEnv("evader", N, sampler, mp, ep,
                         opponent=frozen_policy(ch.policy, deterministic=False),
                         seed=12345)
    obs = env.reset()
    t = float(mp.tile)
    hist = np.zeros((N, args.history, 2), dtype=np.int64)   # agent cell history
    n_done = 0
    deaths = []
    seen_all = 0
    while n_done < args.matches:
        hist[:, 1:] = hist[:, :-1]
        hist[:, 0, 0] = np.clip((env.phys.x[:N] / t).astype(np.int64), 0, mp.arena_w - 1)
        hist[:, 0, 1] = np.clip((env.phys.y[:N] / t).astype(np.int64), 0, mp.arena_h - 1)
        # the env RESPAWNS on termination, so the post-step position of a dead
        # agent is the spawn point: keep the pre-step state for the post-mortem
        pre_x = env.phys.x[:N].copy(); pre_y = env.phys.y[:N].copy()
        pre_vx = env.phys.vx[:N].copy(); pre_vy = env.phys.vy[:N].copy()
        a = ev.predict(obs, deterministic=False)[0]
        obs, _, done, infos = env.step(a)
        for i, info in enumerate(infos):
            if "episode" not in info:
                continue
            n_done += 1
            if info["outcome"] != "ev_hazard":
                continue
            g = env.arenas[i].grid
            # fatal cell = the spike tile nearest the death position
            x, y = float(pre_x[i]), float(pre_y[i])
            cx, cy = int(x / t), int(y / t)
            best, bd = None, 1e9
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    px, py = cx + dx, cy + dy
                    if 0 <= px < mp.arena_w and 0 <= py < mp.arena_h and g[py, px] == SPIKE:
                        d = (px - cx) ** 2 + (py - cy) ** 2
                        if d < bd:
                            bd, best = d, (px, py)
            if best is None:
                seen_all += 1
                continue
            fx, fy = best
            # LEAD TIME = how many contiguous steps the fatal spike had been
            # inside the obs window before the death. (Reporting the FIRST
            # visible step instead gives 0 for every death, which is trivially
            # true: at impact the agent is standing on the spike.)
            lead = -1
            for k in range(args.history):
                ax, ay = hist[i, k]
                vis = (ax - ep.patch_back <= fx <= ax + ep.patch_front and
                       ay - ep.patch_up <= fy <= ay + ep.patch_down)
                if not vis:
                    break
                lead = k
            if args.debug and len(deaths) < 6:
                cells = [tuple(hist[i, k]) for k in range(6)]
                print(f"  death#{len(deaths)} fatal=({fx},{fy}) agent_cells={cells} "
                      f"window=up{ep.patch_up}/down{ep.patch_down}/back{ep.patch_back}"
                      f"/front{ep.patch_front} lead={lead} vy={pre_vy[i]:+.0f}")
            deaths.append({
                "lead": lead,
                "vy": float(pre_vy[i]),
                "vx": float(pre_vx[i]),
                "dy_tiles": fy - cy,
            })
    leads = np.array([d["lead"] for d in deaths])
    vy = np.array([d["vy"] for d in deaths])
    dyt = np.array([d["dy_tiles"] for d in deaths])
    print(f"\n=== spike-death post-mortem: {os.path.basename(args.evader)} ===")
    print(f"deaths={len(deaths)} / {n_done} episodes ({100*len(deaths)/max(n_done,1):.0f}%)"
          f"  no spike found near death: {seen_all}")
    if len(deaths):
        vis = leads >= 0
        print(f"fatal spike inside the patch at the step of death: "
              f"{100*vis.mean():.0f}% of deaths")
        for k in (1, 2, 3, 5, 8, 12):
            print(f"  warned >= {k:2d} frames before impact: {100*(leads >= k).mean():5.1f}%")
        print(f"median warning: {np.median(leads):.0f} frames = "
              f"{1000*np.median(leads)/60:.0f} ms")
        fall_px = 1600/60.0
        print(f"reaction distance at terminal velocity: "
              f"{np.median(leads)*fall_px:.0f} px = {np.median(leads)*fall_px/32:.1f} tiles")
        print(f"spike below the death cell (tiles): mean {dyt.mean():+.1f}, "
              f"{100*(dyt > 0).mean():.0f}% below, {100*(dyt < 0).mean():.0f}% above")
        print(f"vertical speed at death: mean {vy.mean():+.0f} px/s "
              f"(falling in {100*(vy > 100).mean():.0f}% of deaths)")


if __name__ == "__main__":
    main()
