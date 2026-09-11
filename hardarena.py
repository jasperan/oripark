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
  python hardarena.py --run results/run17 --random-seeds 4   # chaos-floor distribution
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
from oripark.evals import play_episodes  # noqa: E402
from oripark.selfplay import frozen_policy, make_ppo, set_nets_from_run  # noqa: E402


def load(path, role, tp, mp, ep):
    rng = np.random.default_rng(0)
    gen = ArenaGenerator(mp, rng, chaser_frac=ep.chaser_spawn_frac)
    env = OriArenaVecEnv(role, 1, FixedSampler(gen), mp, ep, seed=0)
    if path.endswith(".zip"):
        p = PPO.load(path, env=env, device="cpu")
    else:
        p = make_ppo(env, tp, 0, "cpu", side=role)
        p.policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
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
                 mix="mixed", oracle="calm"):
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
        self.arenas = expert_winnable(cands, mp, ep, mode=oracle)
        print(f"  oracle ({oracle}): {len(self.arenas)}/{len(cands)} candidate "
              f"arenas winnable by the scripted expert", flush=True)
        if len(self.arenas) < n:
            raise SystemExit(f"only {len(self.arenas)} winnable arenas; "
                             f"lower --matches or raise max_cand")

    def sample(self, n):
        return [self.arenas[k % len(self.arenas)] for k in range(n)]


def expert_winnable(arenas, mp, ep, max_steps=None, batch=16, mode="calm"):
    """Return the arenas the scripted evader escapes with NO chaser — the
    oracle for 'hard but fair'.

    mode="calm"  (default, the documented intent): the chaser is parked out of
                  the way and the expert gets the REAL chaser position, so it
                  plays a calm traversal the way a competent player would.
    mode="panic" (legacy reproduction): the expert is handed its OWN position
                  as the chaser position, which pins `flee` permanently True
                  (panic hops/dashes), and a RANDOM chaser wanders the arena
                  (it can catch the evader or impale itself). That filter is
                  stricter and noisier than it claims to be — kept only so
                  pre-2026-09 numbers can be reproduced.
    """
    from oripark.scripted import ScriptedEvader
    max_steps = max_steps or ep.max_steps
    win = np.zeros(len(arenas), dtype=bool)
    for s in range(0, len(arenas), batch):
        chunk = arenas[s:s + batch]

        class Fix:
            def sample(self, m):
                return [chunk[k % len(chunk)] for k in range(m)]

        env = OriArenaVecEnv("evader", len(chunk), Fix(), mp, ep,
                             opponent=None, seed=0,
                             chaser_ghost=(mode == "calm"))
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
                if mode == "panic":                 # legacy bug reproduction
                    ch_x = ch_y = phys.x[i]
                else:
                    ch_x, ch_y = phys.x[len(chunk) + i], phys.y[len(chunk) + i]
                acts.append(sp.act(ar, phys.x[i], phys.y[i], phys.vx[i],
                                   phys.vy[i], phys.on_ground[i],
                                   phys.wall_dir[i], phys.can_djump[i],
                                   phys.can_dash[i], phys.dash_t[i],
                                   phys.bash_cd[i], phys.facing[i],
                                   chaser_x=ch_x, chaser_y=ch_y))
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
                seed: int = 7, opp_ep=None, scripted_chaser: bool = False):
    """Play `n_matches` episodes and count ALL FIVE endings.

    This used to be a fourth copy of the episode loop, and it folded
    `ev_hazard` (evader impaled itself) and `ch_hazard` (chaser did) into the
    "timeout" column: a policy that killed itself on spikes was reported as
    surviving to the step limit. It also printed the escape count against
    `args.matches` while dividing by the finished-episode count, so the
    denominator in the artifact disagreed with the one in the arithmetic.
    """
    n_envs = min(n_matches, 32)
    # identical stochastic draws for every entrant -> policy differences only
    torch.manual_seed(seed)
    np.random.seed(seed)
    # a FIXED opponent that never changes as policies are retrained: the
    # privileged scripted pursuer, useful as a stable yardstick and as the
    # strongest non-learning opponent in the repo
    kwargs = {}
    if scripted_chaser:
        from oripark.selfplay import _scripted_chaser_acts
        kwargs["opp_acts_fn"] = _scripted_chaser_acts(n_envs, mp, ep)
        ch = None
    else:
        ch = frozen_policy(ch_pol, deterministic=not chaser_stoch)
        kwargs["opponent"] = ch
    env = OriArenaVecEnv("evader", n_envs, sampler, mp, ep,
                         seed=seed, opp_ep=opp_ep, **kwargs)
    return play_episodes(
        env, lambda obs: ev_pol.predict(obs, deterministic=False)[0],
        ep.max_steps, n_matches)


def expert_escape_rate(ch_pol, sampler, mp, ep, n_matches, chaser_stoch: bool,
                       seed: int = 7, opp_ep=None, scripted_chaser: bool = False):
    """Scripted-expert escape rate on the same protocol (privileged state)."""
    from oripark.scripted import expert_act_fn
    n_envs = min(n_matches, 32)
    if chaser_stoch:
        torch.manual_seed(seed)
        np.random.seed(seed)
    kwargs = {}
    if scripted_chaser:
        from oripark.selfplay import _scripted_chaser_acts
        kwargs["opp_acts_fn"] = _scripted_chaser_acts(n_envs, mp, ep)
    else:
        kwargs["opponent"] = frozen_policy(ch_pol, deterministic=not chaser_stoch)
    env = OriArenaVecEnv("evader", n_envs, sampler, mp, ep,
                         seed=seed, opp_ep=opp_ep, **kwargs)
    return play_episodes(env, expert_act_fn(env), ep.max_steps, n_matches)


def _fmt(name: str, r: dict) -> str:
    return (f"  {name:<18} escape {r['esc_rate']:6.1%} ({r['esc']:>3}/{r['n']:<3}) "
            f"caught {r['caught']:>3} timeout {r['tm']:>3} "
            f"ev_hazard {r['ev_hazard']:>3} ch_hazard {r['ch_hazard']:>3} "
            f"surv {r['surv_rate']:5.1%} "
            f"zone {r['zone']:5.1f} len {r['length']:4.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/run17")
    ap.add_argument("--chaser-run", default=None,
                    help="run dir whose frozen chaser.zip to use (default: same as --run)")
    ap.add_argument("--matches", type=int, default=150)
    ap.add_argument("--chaser", choices=["det", "stoch"], default="det",
                    help="frozen chaser mode: argmax (det) or action sampling (stoch)")
    ap.add_argument("--set", choices=["mixed", "07", "085"], default="mixed",
                    help="arena difficulty mix (mixed = 2/3 x 0.7 + 1/3 x 0.85)")
    ap.add_argument("--entrants", choices=["trio", "all"], default="trio",
                    help="trio = random/BC/trained; all = every checkpoint too")
    ap.add_argument("--max-checkpoints", type=int, default=8)
    ap.add_argument("--random-seeds", type=int, default=1,
                    help="random entrant = population of K deterministic inits "
                         "(all members are fresh inits); mean+min/max "
                         "reported — turns the chaotic-init floor into a "
                         "measured distribution")
    ap.add_argument("--scripted-chaser", action="store_true",
                    help="use the privileged scripted pursuer instead of a "
                         "frozen checkpoint: a fixed yardstick whose strength "
                         "does not drift as policies are retrained")
    ap.add_argument("--legacy-obs-chaser", action="store_true",
                    help="same, for the frozen chaser checkpoint (pre-fix "
                         "chasers were also trained on the old scaling)")
    ap.add_argument("--legacy-obs", action="store_true",
                    help="feed the EVADER the pre-fix observation scaling "
                         "(32x direction features + the old 9 px no-orb "
                         "sentinel). REQUIRED to evaluate checkpoints trained "
                         "before the obs-units fix, otherwise they see inputs "
                         "they never trained on. Applies to the evader only: "
                         "pair it with --legacy-obs-chaser to put both sides "
                         "in their own training convention")
    ap.add_argument("--oracle", choices=["calm", "panic"], default="calm",
                    help="arena filter: 'calm' = competent traversal (default, "
                         "documented intent); 'panic' = legacy buggy filter that "
                         "handed the expert its own position as the chaser")
    ap.add_argument("--expert", action="store_true",
                    help="also evaluate the scripted expert (privileged-state "
                         "controller) as an entrant on the same arena set")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tp, mp, ep = load_run_params(args.run)
    if args.legacy_obs:
        ep.obs_legacy_units = True
    set_nets_from_run(tp, args.run)
    crun = args.chaser_run or args.run
    ctp, cmp_, cep = load_run_params(crun)
    if args.legacy_obs_chaser:
        cep.obs_legacy_units = True
    set_nets_from_run(ctp, crun)
    rng = np.random.default_rng(1)
    gen = ArenaGenerator(mp, rng, chaser_frac=ep.chaser_spawn_frac)
    sampler = HardSampler(gen, args.matches, mp, ep, mix=args.set,
                          oracle=args.oracle)

    ch = None if args.scripted_chaser else load(
        os.path.join(crun, "chaser.zip"), "chaser", ctp, cmp_, cep)

    from oripark.baselines import find_baseline, find_pool0
    ents = []
    # the pre-training baseline (see oripark/baselines.py: pool_ev_0.pt is NOT
    # it — the league trims the untrained member out of the pool)
    bp, blabel = find_baseline(args.run)
    if bp:
        ents.append((blabel, bp))
    p0, p0label = find_pool0(args.run)
    if p0:
        ents.append((p0label, p0))
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
    if args.expert:
        ents.append(("scripted expert", "__expert__"))

    mode = "stochastic chaser" if args.chaser == "stoch" else "deterministic chaser"
    print(f"hard-arena skill test: {args.matches} matches, stochastic evader, {mode}\n")
    rows = []
    for name, path in ents:
        if path == "__expert__":
            r = expert_escape_rate(
                ch.policy if ch is not None else None, sampler, mp, ep,
                args.matches, chaser_stoch=(args.chaser == "stoch"),
                opp_ep=cep, scripted_chaser=args.scripted_chaser)
        else:
            ev = load(path, "evader", tp, mp, ep)
            r = escape_rate(ev.policy, ch.policy if ch is not None else None,
                            sampler, mp, ep, args.matches,
                            chaser_stoch=(args.chaser == "stoch"), opp_ep=cep,
                            scripted_chaser=args.scripted_chaser)
        r["name"] = name
        r["rate"] = r["esc_rate"]
        rows.append(r)
        print(_fmt(name, r), flush=True)

    # ---- fresh-init population (the true random floor): every member is an
    # independent random init; it used to reuse the run's pool_ev_0 as member 0,
    # which is an early TRAINED snapshot (see oripark/baselines.py)
    if args.random_seeds > 1:
        rates = []
        for i in range(args.random_seeds):
            env = OriArenaVecEnv("evader", 1, FixedSampler(gen), mp, ep, seed=0)
            torch.manual_seed(1000 + i)
            ev = make_ppo(env, tp, 1000 + i, "cpu", "evader")
            env.close()
            r = escape_rate(ev.policy, ch.policy if ch is not None else None,
                            sampler, mp, ep, args.matches,
                            chaser_stoch=(args.chaser == "stoch"), opp_ep=cep,
                            scripted_chaser=args.scripted_chaser)
            rates.append(r["esc_rate"])
            print(f"  fresh init #{i:<2}" + _fmt("", r), flush=True)
        rates = np.array(rates)
        rows.append({"name": f"random pop (K={args.random_seeds})",
                     "rate": float(rates.mean()), "pop": True,
                     "lo": float(rates.min()), "hi": float(rates.max())})
        print(f"  fresh-init population K={args.random_seeds}: mean "
              f"{rates.mean():.1%} min {rates.min():.1%} max {rates.max():.1%} "
              f"spread {rates.max() - rates.min():.1%}", flush=True)

    md = [f"# Hard-arena skill test (`{args.run}`, {mode})",
          "",
          f"{args.matches} matches, stochastic evader, mixed 0.7/0.85 arenas, "
          f"chaser (`{'scripted' if args.scripted_chaser else args.chaser}`), "
          f"oracle `{args.oracle}` "
          f"({len(sampler.arenas)} arenas), obs "
          f"`{'legacy-32x' if args.legacy_obs else 'fraction'}` / chaser obs "
          f"`{'legacy-32x' if cep.obs_legacy_units else 'fraction'}`.",
          "",
          "| policy | escape rate | survival | escaped | caught | timeout | "
          "ev_hazard | ch_hazard | mean zone | avg len |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: -r["rate"]):
        if r.get("pop"):
            md.append(f"| {r['name']} | {r['rate']:.1%} | — | — | — "
                      "| — | — | — | — | — |\n"
                      f"|  min / max init | {r['lo']:.1%} / {r['hi']:.1%} "
                      "| — | — | — | — | — | — | — | — |")
            continue
        md.append(f"| {r['name']} | {r['rate']:.1%} ({r['esc']}/{r['n']}) "
                  f"| {r['surv_rate']:.1%} | {r['esc']} | {r['caught']} | {r['tm']} "
                  f"| {r['ev_hazard']} | {r['ch_hazard']} | {r['zone']:.1f} "
                  f"| {r['length']:.0f} |")
    md += ["",
           "`survival` = escaped + timeout + chaser-fell (the evader did not lose).",
           "`mean zone` = mean furthest arena zone (of 12) the evader reached;",
           "it discriminates policies that survive to the timestep limit without",
           "reaching the portal. Every finished episode counts in the escape-rate",
           "denominator, including both hazard endings.",
           "", "Caveat: a deterministic frozen chaser is partly exploitable by chaos;",
           "the stochastic-chaser number is the honest one."]
    out = args.out or os.path.join(args.run, f"hardarena_{args.chaser}.md")
    with open(out, "w") as f:
        f.write("\n".join(md))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
