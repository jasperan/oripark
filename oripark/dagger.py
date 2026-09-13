"""DAgger distillation of the scripted expert into the evader policy.

Plain behaviour cloning on expert trajectories fails badly here: the scripted
controller is *privileged* (it plans with BFS over the tile grid and reads the
exact physics state), so the same observation maps to different actions
depending on hidden state, and the student's own small errors move it onto
states the demos never covered. Measured result of the old BC pipeline — a
teacher that escapes 56% of hard arenas distilled into a student that escapes
1-4% and never leaves the first third of the arena.

DAgger fixes the state distribution: roll out the *student*, ask the expert
what it would do in each of those states, and train on those labels. `beta`
mixes the expert back in (beta=1 is pure expert rollouts) which keeps the
rollouts from wandering too far early on.

Usage:
    python -m oripark.dagger --run results/run21_fixed --rounds 4 \
        --episodes 64 --out results/run22_init
writes <out>/evader_init.pt for `train.py --init-evader <out>/evader_init.pt`.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .arena import ArenaGenerator
from .bc import behavior_clone
from .config import load_run_params
from .env import OriArenaVecEnv
from .scripted import ScriptedEvader
from .selfplay import frozen_policy, make_ppo, set_nets_from_run


class StaticArenaSampler:
    """Sample arena params around a mean (same behaviour as training)."""

    def __init__(self, gen, mu, rng):
        self.gen, self.mu, self.rng = gen, np.asarray(mu, dtype=np.float64), rng

    def sample(self, n):
        out = []
        while len(out) < n:
            a = self.gen.generate(self.mu, seed=int(self.rng.integers(0, 2 ** 31 - 1)))
            if a is not None:
                out.append(a)
        return out

    def mean_params(self):
        return self.mu


def dagger_collect(env, scripts, policy, n_episodes: int, max_steps: int,
                   beta: float = 0.5, deterministic: bool = False,
                   device: str = "cpu", seed: int = 0, keep_failures: bool = True,
                   role: str = "evader"):
    """Roll the student out, label every visited state with the expert action.

    Returns (obs_list, act_list, stats). `beta` is the per-env probability of
    taking the expert's action instead of the student's.

    `role` picks which agent is the learner: evaders occupy indices 0..N-1 and
    chasers N..2N-1 in the vectorised env, while the *learner* is whichever
    slice the env was built for. So for role="chaser" the expert reads the
    chaser slice (N+i) and looks at the evader at i, and vice versa.
    """
    N = env.n_envs
    nd = 5 if role == "evader" else 3
    rng = np.random.default_rng(seed)
    obs_buf, act_buf = [], []
    ep_obs = [[] for _ in range(N)]
    ep_act = [[] for _ in range(N)]
    ep_steps = np.zeros(N, dtype=np.int32)
    outcomes: dict[str, int] = {}
    n_done = 0
    obs = env.reset()
    while n_done < n_episodes:
        student = (policy.predict(obs, deterministic=deterministic)[0]
                   if beta < 1.0 else None)
        take_expert = (np.ones(N, dtype=bool) if student is None
                       else rng.random(N) < beta)
        acts = np.zeros((N, nd), dtype=np.int64)
        phys = env.phys
        for i in range(N):
            sp = scripts[i]
            if ep_steps[i] == 0:
                sp.reset(env.arenas[i])
            # the label is the expert's action in the state the STUDENT is in
            if role == "evader":
                expert_a = sp.act(env.arenas[i], phys.x[i], phys.y[i], phys.vx[i],
                                  phys.vy[i], phys.on_ground[i], phys.wall_dir[i],
                                  phys.can_djump[i], phys.can_dash[i],
                                  phys.dash_t[i], phys.bash_cd[i], phys.facing[i],
                                  chaser_x=phys.x[N + i], chaser_y=phys.y[N + i])
            else:
                k = N + i                      # the learner (chaser) is at N+i
                expert_a = sp.act(env.arenas[i], phys.x[k], phys.y[k], phys.vx[k],
                                  phys.vy[k], phys.on_ground[k], phys.wall_dir[k],
                                  phys.can_djump[k], phys.can_dash[k],
                                  phys.dash_t[k], phys.facing[k],
                                  evader_x=phys.x[i], evader_y=phys.y[i])
            ep_obs[i].append(obs[i])
            ep_act[i].append(expert_a)
            acts[i] = expert_a if take_expert[i] else student[i]
        obs, _, done, infos = env.step(acts)
        for i in range(N):
            ep_steps[i] += 1
            if done[i] or ep_steps[i] >= max_steps:
                out = infos[i].get("outcome") if done[i] else None
                if out:
                    outcomes[out] = outcomes.get(out, 0) + 1
                    n_done += 1
                if keep_failures or out == "escaped":
                    obs_buf.append(np.stack(ep_obs[i]))
                    act_buf.append(np.stack(ep_act[i]))
                ep_obs[i], ep_act[i] = [], []
                ep_steps[i] = 0
    total = int(sum(len(o) for o in obs_buf))
    return obs_buf, act_buf, {"episodes": n_done, "kept": len(obs_buf),
                              "pairs": total, "outcomes": outcomes}


def dagger_rounds(env, scripts, policy, rounds: int = 4, episodes: int = 64,
                  epochs: int = 4, lr: float = 3e-4, device: str = "cpu",
                  max_steps: int = 1500, log=print, role: str = "evader",
                  opponent_cycle=None):
    """Interleaved collect/train rounds with a decaying expert mixing rate."""
    hist = []
    for r in range(rounds):
        beta = max(0.0, 1.0 - r / max(1, rounds - 1))
        obs_list, act_list, stats = dagger_collect(
            env, scripts, policy, n_episodes=episodes, max_steps=max_steps,
            beta=beta, device=device, seed=1000 + r, role=role)
        if not obs_list:
            log(f"[dagger] round {r + 1}: no data, stopping")
            break
        log(f"[dagger] round {r + 1}/{rounds} beta={beta:.2f} "
            f"{stats['kept']} eps / {stats['pairs']} pairs "
            f"outcomes={stats['outcomes']}")
        losses = behavior_clone(policy, obs_list, act_list, epochs=epochs,
                                lr=lr, device=device)
        hist.append({"round": r + 1, "beta": beta, "stats": stats,
                     "nll": losses[-1] if losses else None})
        log(f"[dagger] round {r + 1} final nll={hist[-1]['nll']}")
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/run21_fixed",
                    help="run whose params/nets/chaser define the task")
    ap.add_argument("--out", default="results/run22_init")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=64,
                    help="episodes collected per round (per vector batch)")
    ap.add_argument("--epochs", type=int, default=4, help="BC epochs per round")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--chaser", default=None,
                    help="path to a specific frozen chaser checkpoint "
                         "(default <run>/chaser.zip, which only exists once "
                         "the run has finished; snapshots are chaser_bNNN.zip)")
    ap.add_argument("--chaser-list", default=None,
                    help="comma-separated opponent checkpoints (or run "
                         "directories) to ROTATE per round; the literal "
                         "token 'scripted' adds the privileged scripted "
                         "pursuer. Single-opponent distillation overfits: "
                         "distilled policies scored high on the arenas they "
                         "were selected on and average elsewhere")
    ap.add_argument("--ghost", action="store_true",
                    help="train against a parked chaser instead of the run's")
    ap.add_argument("--legacy-obs-chaser", action="store_true",
                    help="run the frozen chaser in the pre-fix obs convention, "
                         "i.e. the convention it was trained in (without this "
                         "a pre-fix chaser is a crippled opponent)")
    ap.add_argument("--obs-path-hint", action="store_true",
                    help="append the 6 BFS waypoint features to the evader obs "
                         "(see docs/EXPERIMENTS.md E26)")
    ap.add_argument("--role", choices=["evader", "chaser"], default="evader",
                    help="which side to distil: the evader from the scripted "
                         "evader, or the chaser from the scripted pursuer "
                         "(a non-suicidal teacher, the learned chasers die on "
                         "spikes in 19-43%% of episodes)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    tp, mp, ep = load_run_params(args.run)
    set_nets_from_run(tp, args.run)
    os.makedirs(args.out, exist_ok=True)
    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    n_envs = args.workers or tp.n_envs

    role = args.role
    if args.obs_path_hint:
        ep.obs_path_hint = True
    rng = np.random.default_rng(tp.seed + 7)
    gen = ArenaGenerator(mp, rng, chaser_frac=ep.chaser_spawn_frac)
    mu = (np.load(os.path.join(args.run, "adv_mu.npy"))
          if os.path.exists(os.path.join(args.run, "adv_mu.npy"))
          else np.full(6, 0.5))

    from stable_baselines3 import PPO
    cep = ep
    if args.legacy_obs_chaser:
        from dataclasses import replace
        cep = replace(ep, obs_legacy_units=True)

    if role == "chaser":
        # Learner = chaser, opponent = the scripted evader: a privileged,
        # strong, non-fleeing target. The learned chasers die on spikes in
        # 19-43% of episodes, which gives the evader free "wins"; the scripted
        # pursuer never suicides, so distilling it fixes that source of noise.
        from .scripted import ScriptedChaser, expert_act_fn
        env = OriArenaVecEnv("chaser", n_envs, StaticArenaSampler(gen, mu, rng),
                             mp, ep, seed=tp.seed + 7)
        env.opp_acts_fn = expert_act_fn(env)     # read per step, not at init
        ppo = make_ppo(env, tp, tp.seed, dev, role)
        policy = ppo.policy
        policy.to(dev)
        scripts = [ScriptedChaser(mp, ep) for _ in range(n_envs)]
        print("[dagger] role=chaser, opponent=scripted evader (--ghost ignored)",
              flush=True)
        hist = dagger_rounds(env, scripts, policy, rounds=args.rounds,
                             episodes=args.episodes, epochs=args.epochs,
                             lr=args.lr, device=dev, max_steps=ep.max_steps,
                             log=lambda m: print(m, flush=True), role=role,
                             opponent_cycle=cycle)
        env.close()
        out_path = os.path.join(args.out, "chaser_init.pt")
        torch.save(policy.state_dict(), out_path)
        ppo.save(os.path.join(args.out, "chaser.zip"))
        import json as _json
        with open(os.path.join(args.out, "dagger_chaser.json"), "w") as f:
            _json.dump({"run": args.run, "role": role, "rounds": args.rounds,
                        "episodes": args.episodes, "epochs": args.epochs,
                        "history": hist}, f, indent=1)
        print(f"[dagger] wrote {out_path}")
        return

    cycle = None
    if args.chaser_list:
        from stable_baselines3 import PPO as _PPO
        cycle = []
        for spec in [x.strip() for x in args.chaser_list.split(",") if x.strip()]:
            if spec == "scripted":
                cycle.append("scripted")
                continue
            pth = (os.path.join(spec, "chaser.zip") if os.path.isdir(spec)
                   else spec)
            probe = OriArenaVecEnv("chaser", 1, StaticArenaSampler(gen, mu, rng),
                                   mp, cep, seed=0)
            cycle.append(_PPO.load(pth, env=probe, device="cpu"))
            probe.close()
            print(f"[dagger] rotation opponent {len(cycle)}: {pth}", flush=True)
    chaser = None
    if not args.ghost and cycle is None:
        ck = args.chaser or os.path.join(args.run, "chaser.zip")
        if os.path.exists(ck):
            probe = OriArenaVecEnv("chaser", 1, StaticArenaSampler(gen, mu, rng),
                                   mp, cep, seed=0)
            chaser = PPO.load(ck, env=probe, device="cpu")
            probe.close()
            print(f"[dagger] opponent = {ck}", flush=True)
        else:
            print(f"[dagger] no chaser at {ck} -> ghost rollouts", flush=True)
    if cycle:
        chaser = cycle[0]
    opp = (None if (cycle and cycle[0] == "scripted")
           else (frozen_policy(chaser.policy, True) if chaser is not None else None))

    env = OriArenaVecEnv("evader", n_envs, StaticArenaSampler(gen, mu, rng), mp, ep,
                         opponent=opp, seed=tp.seed + 7,
                         chaser_ghost=chaser is None and not cycle, opp_ep=cep)
    if cycle and cycle[0] == "scripted":
        from .scripted import scripted_chaser_act_fn
        env.opp_acts_fn = scripted_chaser_act_fn(env)
    ppo = make_ppo(env, tp, tp.seed, dev, role)
    policy = ppo.policy
    policy.to(dev)
    scripts = [ScriptedEvader(mp, ep) for _ in range(n_envs)]

    hist = dagger_rounds(env, scripts, policy, rounds=args.rounds,
                         episodes=args.episodes, epochs=args.epochs, lr=args.lr,
                         device=dev, max_steps=ep.max_steps,
                         log=lambda m: print(m, flush=True), role=role)
    env.close()

    stem = "evader"
    # write params.json so hardarena/select/traversal_probe pick up the same obs
    # geometry (e.g. obs_path_hint) when evaluating this policy
    from .config import save_run_params
    save_run_params(args.out, tp, mp, ep)
    out_path = os.path.join(args.out, f"{stem}_init.pt")
    torch.save(policy.state_dict(), out_path)
    # also write an SB3 checkpoint so the distilled policy can be entered
    # straight into hardarena.py as "trained (best)"
    ppo.save(os.path.join(args.out, f"{stem}.zip"))
    import json
    with open(os.path.join(args.out, f"dagger_{stem}.json"), "w") as f:
        json.dump({"run": args.run, "role": role, "rounds": args.rounds,
                   "episodes": args.episodes, "epochs": args.epochs,
                   "ghost": args.ghost, "history": hist}, f, indent=1)
    print(f"[dagger] wrote {out_path}")


if __name__ == "__main__":
    main()
