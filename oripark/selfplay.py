"""Self-play league for the Ori tag game.

Two PPO policies (evader = Ori's full move kit, chaser = pursuer) are
trained in alternating blocks, each against a snapshot sampled from the
opponent's league pool (mostly the latest, sometimes an older checkpoint,
to avoid degenerate cycling). A third adversarial learner — the terrain
adversary — runs CEM over arena parameters to keep the evader's win rate
near 50%, so the curriculum stays at the edge of the agents' skill.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .adversaries import StaticSampler, TerrainAdversary
from .arena import Arena, ArenaGenerator
from .config import EnvParams, MoveParams, TrainParams
from .env import OriArenaVecEnv
from .metrics import Elo, Metrics


def frozen_policy(policy, deterministic: bool = False):
    """Wrap a torch policy (or PPO) into a VecEnv-friendly callable."""
    def act(obs):
        a, _ = policy.predict(obs, deterministic=deterministic)
        return np.asarray(a)
    return act


def infer_net_arch(path: str):
    """Read the policy net_arch from a saved PPO zip (works without an env
    — SB3 pickles the spaces and policy_kwargs). Used so .pt baseline shells
    are built with the same architecture the run was trained with."""
    from stable_baselines3 import PPO
    p = PPO.load(path, device="cpu")
    arch = p.policy_kwargs.get("net_arch", [256, 256])
    del p
    return list(arch)


def set_nets_from_run(tp: TrainParams, run_dir: str):
    """Align tp.evader_net/chaser_net with the saved policies in run_dir."""
    ev = os.path.join(run_dir, "evader.zip")
    ch = os.path.join(run_dir, "chaser.zip")
    if os.path.exists(ev):
        tp.evader_net = infer_net_arch(ev)
    if os.path.exists(ch):
        tp.chaser_net = infer_net_arch(ch)


def sample_opponent(pool, rng, latest_prob: float):
    if len(pool) == 1 or rng.random() < latest_prob:
        return pool[-1]
    return pool[int(rng.integers(0, len(pool) - 1))]


def sample_ladder_opponent(pool, rng, block: int, tp: TrainParams):
    """Chaser-strength curriculum: anneal latest-prob from a low start so the
    evader first faces OLD (weak) chaser snapshots and progressively the
    strongest. When not picking the latest, older snapshots are weighted more
    heavily early on — a natural difficulty ladder from self-play history."""
    if len(pool) == 1:
        return pool[-1]
    frac = min(1.0, block / max(1, tp.ladder_blocks))
    p_latest = tp.opp_latest_start + (tp.opp_latest_end - tp.opp_latest_start) * frac
    if rng.random() < p_latest:
        return pool[-1]
    n = len(pool) - 1
    w = np.arange(n, 0, -1, dtype=float)          # oldest gets most weight early
    w = (1 - frac) * w / w.sum() + frac * np.ones(n) / n
    return pool[int(rng.choice(n, p=w / w.sum()))]


class _Drain(BaseCallback):
    """Collects finished episodes from the vec env during PPO rollouts."""

    def __init__(self, env, metrics: Metrics, side: str):
        super().__init__()
        self.env = env
        self.metrics = metrics
        self.side = side

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        for rec in self.env.drain():
            rec["side"] = self.side
            self.metrics.add_episode(rec)


class _CandSampler:
    """Maps flat env index -> candidate param row with fixed seed."""

    def __init__(self, gen: ArenaGenerator, cands: np.ndarray, matches: int, seed0: int = 5000):
        self.gen = gen
        self.cands = cands
        self.matches = matches
        self.seed0 = seed0

    def sample(self, n: int) -> list[Arena]:
        out = []
        for i in range(n):
            c = i // self.matches
            s = self.seed0 + i % self.matches
            out.append(self.gen.generate(self.cands[c], seed=s))
        return out


def _cand_eval(cands, ev_pol, ch_pol, gen, mp, ep, tp, device) -> np.ndarray:
    """Evader win rate per candidate arena-parameter row (vs latest chaser)."""
    n_cand = len(cands)
    N = n_cand * tp.adv_matches
    sampler = _CandSampler(gen, cands, tp.adv_matches)
    env = OriArenaVecEnv("evader", N, sampler, mp, ep,
                         opponent=frozen_policy(ch_pol, deterministic=True), seed=0)
    obs = env.reset()
    wrs = np.zeros(n_cand)
    counts = np.zeros(n_cand)
    for _ in range(tp.eval_ep_len):
        # stochastic play — the honest measure; deterministic argmax gets
        # stuck in repetitive patterns and would report wr=0 everywhere,
        # collapsing the whole curriculum to easy levels
        a = ev_pol.predict(obs, deterministic=False)[0]
        obs, _, done, infos = env.step(a)
        for i, info in enumerate(infos):
            if "episode" in info:
                c = i // tp.adv_matches
                wrs[c] += 1.0 if info["ev_win"] else 0.5 if info["outcome"] == "timeout" else 0.0
                counts[c] += 1
        if counts.sum() >= N:
            break
    env.close()
    return wrs / np.maximum(counts, 1)


def evaluate(ev_pol, ch_pol, sampler, mp, ep, n_envs: int, max_steps: int, seed: int = 7) -> dict:
    """Head-to-head eval: evader policy vs chaser policy, both deterministic.

    Win semantics: escape/ch_hazard = evader win; caught/ev_hazard = chaser win;
    timeout = draw. Escape rate is the primary agility signal.
    """
    env = OriArenaVecEnv("evader", n_envs, sampler, mp, ep,
                         opponent=frozen_policy(ch_pol, deterministic=True), seed=seed)
    obs = env.reset()
    wins = losses = draws = 0
    escaped = caught = 0
    lens = []
    for _ in range(max_steps):
        a = ev_pol.predict(obs, deterministic=True)[0]
        obs, _, done, infos = env.step(a)
        for info in infos:
            if "episode" in info:
                lens.append(info["episode"]["l"])
                out = info["outcome"]
                if out == "escaped":
                    wins += 1; escaped += 1
                elif out == "ch_hazard":
                    wins += 1
                elif out == "caught":
                    losses += 1; caught += 1
                elif out == "ev_hazard":
                    losses += 1
                else:
                    draws += 1
        if wins + losses + draws >= n_envs:
            break
    env.close()
    return {"wins": wins, "losses": losses, "draws": draws, "escaped": escaped,
            "caught": caught,
            "avg_len": float(np.mean(lens)) if lens else 0.0}


def make_ppo(env, tp: TrainParams, seed: int, device: str, side: str = "evader") -> PPO:
    net = tp.evader_net if side == "evader" else tp.chaser_net
    lr = tp.evader_lr if side == "evader" else tp.chaser_lr
    return PPO(
        "MlpPolicy", env,
        n_steps=tp.n_steps, batch_size=tp.batch_size, n_epochs=tp.n_epochs,
        gamma=tp.gamma, gae_lambda=tp.gae_lambda, clip_range=tp.clip_range,
        ent_coef=tp.ent_coef, vf_coef=tp.vf_coef, max_grad_norm=tp.max_grad_norm,
        learning_rate=lambda _progress: lr,   # constant LR across self-play blocks
        policy_kwargs=dict(net_arch=list(net)),
        seed=seed, device=device, verbose=0,
    )


def clone_policy(policy, env, tp: TrainParams, seed: int, device: str,
                 side: str = "evader"):
    """Copy a policy via state_dict — torch deepcopy breaks after optimizer
    steps (parameters become non-leaf tensors)."""
    clone = make_ppo(env, tp, seed, device, side=side)
    clone.policy.load_state_dict(policy.state_dict())
    return clone.policy


def _scripted_chaser_acts(n_envs, mp, ep):
    """Build an opp_acts_fn driving ScriptedChaser pursuers from privileged
    state (used to collect duel demos with real chase pressure)."""
    from oripark.scripted import ScriptedChaser
    chasers = [ScriptedChaser(mp, ep) for _ in range(n_envs)]
    last = [None] * n_envs

    def fn(env):
        N = env.n_envs
        acts = []
        for i in range(N):
            ci = N + i
            if last[i] is None or env.t[i] == 0:
                chasers[i].reset(env.arenas[i])
                last[i] = env.t[i]
            ph = env.phys
            acts.append(chasers[i].act(
                env.arenas[i], ph.x[ci], ph.y[ci], ph.vx[ci], ph.vy[ci],
                ph.on_ground[ci], ph.wall_dir[ci], ph.can_djump[ci], ph.can_dash[ci],
                ph.dash_t[ci], ph.facing[ci], ph.x[i], ph.y[i]))
        return np.stack(acts)

    return fn


def run(tp: TrainParams, mp: MoveParams, ep: EnvParams, out_dir: str, device: str = "auto"):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(tp.seed)
    gen = ArenaGenerator(mp, rng)
    adv = TerrainAdversary(gen, rng, pop=tp.adv_pop, elites=tp.adv_elites,
                           sigma0=tp.adv_sigma0, target_wr=tp.adv_target_wr)
    metrics = Metrics(out_dir)

    boot_ev = OriArenaVecEnv("evader", tp.n_envs, StaticSampler(gen, adv.mean_params(), rng),
                             mp, ep, seed=tp.seed)
    boot_ch = OriArenaVecEnv("chaser", tp.n_envs, StaticSampler(gen, adv.mean_params(), rng),
                             mp, ep, seed=tp.seed + 1)
    evader = make_ppo(boot_ev, tp, tp.seed, device, side="evader")
    chaser = make_ppo(boot_ch, tp, tp.seed + 1, device, side="chaser")

    # league pools (snapshot[0] = random init, used for "before" demos)
    ev_pool = [clone_policy(evader.policy, boot_ev, tp, tp.seed, device, "evader")]
    ch_pool = [clone_policy(chaser.policy, boot_ch, tp, tp.seed + 1, device, "chaser")]
    elo_ev, elo_ch = Elo(), Elo()
    # true untrained baselines (the league pool trims old snapshots)
    torch.save(ev_pool[0].state_dict(), os.path.join(out_dir, "evader_init.pt"))
    torch.save(ch_pool[0].state_dict(), os.path.join(out_dir, "chaser_init.pt"))

    # ---- behavior cloning: pretrain the evader on scripted expert demos ----
    bc_demos = None
    if getattr(tp, "bc_epochs", 0) > 0:
        from oripark.scripted import ScriptedEvader, collect_demos
        from oripark.bc import behavior_clone
        n_flee = int(tp.bc_episodes * getattr(tp, "bc_flee_frac", 0.0))
        n_trav = tp.bc_episodes - n_flee
        # Phase A: traversal demos (ghost chaser) — pure movement skill
        demos_env = OriArenaVecEnv(
            "evader", tp.n_envs,
            StaticSampler(gen, adv.mean_params(), np.random.default_rng(tp.seed + 2)),
            mp, ep, seed=tp.seed + 2, chaser_ghost=True)
        scripts = [ScriptedEvader(mp, ep) for _ in range(tp.n_envs)]
        obs_list, act_list, _ = collect_demos(
            demos_env, scripts, n_episodes=n_trav, max_steps=ep.max_steps,
            require_escape=True, seed=0)
        demos_env.close()
        # Phase B: pursued demos — a pursuer hovering just behind triggers the
        # expert's flee rules (dash bursts, hops) so the NN starts knowing how
        # to run WITH a chaser on its heels
        if n_flee > 0 and obs_list:
            flee_env = OriArenaVecEnv(
                "evader", tp.n_envs,
                StaticSampler(gen, adv.mean_params(), np.random.default_rng(tp.seed + 3)),
                mp, ep, seed=tp.seed + 3, chaser_ghost=True)
            obs2, act2, _ = collect_demos(
                flee_env, scripts, n_episodes=n_flee, max_steps=ep.max_steps,
                require_escape=True, seed=0,
                fake_chaser_dx=getattr(tp, "bc_pursued_dx", -120.0))
            flee_env.close()
            obs_list += obs2
            act_list += act2
        if obs_list:
            behavior_clone(evader.policy, obs_list, act_list,
                           epochs=tp.bc_epochs, lr=tp.evader_lr, device=device)
            bc_demos = (obs_list, act_list)
            # the BC'd policy is now the "before" baseline for demos/evals
            torch.save(evader.policy.state_dict(), os.path.join(out_dir, "evader_init.pt"))
            ev_pool[0] = clone_policy(evader.policy, boot_ev, tp, tp.seed, device, "evader")

    # pre-created eval sampler: mean adversary params, fixed seeds -> stable Elo
    eval_sampler = StaticSampler(gen, adv.mean_params(), np.random.default_rng(4242))

    summary = {"evader": None, "chaser": None, "adv": None, "adv_mu": list(adv.mu)}

    for block in range(tp.blocks):
        t0 = time.time()
        # ---------------- train evader vs sampled chaser ----------------
        # warmup: train the evader against a random chaser so it learns level
        # traversal / escapes before the adversarial chase begins
        if block < tp.warmup_blocks:
            opp = None
            opp_name = "random"
        else:
            opp = sample_opponent(ch_pool, rng, tp.opp_latest_prob)
            opp_name = "pool"
        env = OriArenaVecEnv("evader", tp.n_envs, adv, mp, ep,
                             opponent=None if opp is None else frozen_policy(opp),
                             seed=tp.seed + block)
        evader.env = env
        evader.learn(total_timesteps=tp.block_steps,
                     callback=_Drain(env, metrics, "evader"),
                     reset_num_timesteps=False)
        env.close()

        # ---- BC regularization: keep the evader anchored to expert traversal
        if bc_demos is not None and block % tp.bc_reg_every == 0:
            from oripark.bc import behavior_clone
            obs_list, act_list = bc_demos
            k = max(1, len(obs_list) // 2)          # subsample for speed
            behavior_clone(evader.policy, obs_list[:k], act_list[:k],
                           epochs=tp.bc_reg_epochs,
                           lr=tp.evader_lr * tp.bc_reg_lr_frac, device=device)

        # ---------------- eval + snapshot evader ----------------
        res = evaluate(evader.policy, chaser.policy, eval_sampler, mp, ep,
                       n_envs=min(tp.eval_matches, tp.n_envs), max_steps=tp.eval_ep_len)
        n = res["wins"] + res["losses"] + res["draws"]
        score = (res["wins"] + 0.5 * res["draws"]) / max(n, 1)
        r_opp = elo_ch.rating
        elo_ev.update(r_opp, score)
        elo_ch.update(elo_ev.rating, 1.0 - score)
        ev_pool.append(clone_policy(evader.policy, boot_ev, tp, tp.seed, device, "evader"))
        if len(ev_pool) > tp.pool_size:
            ev_pool.pop(0)

        # ---------------- train chaser vs sampled evader ----------------
        # (during warmup the chaser also plays a random evader)
        if block < tp.warmup_blocks:
            opp = None
        else:
            opp = sample_opponent(ev_pool, rng, tp.opp_latest_prob)
        env = OriArenaVecEnv("chaser", tp.n_envs, adv, mp, ep,
                             opponent=None if opp is None else frozen_policy(opp),
                             seed=tp.seed + block + 1000)
        chaser.env = env
        chaser.learn(total_timesteps=tp.block_steps,
                     callback=_Drain(env, metrics, "chaser"),
                     reset_num_timesteps=False)

        # ---------------- eval + snapshot chaser ----------------
        res2 = evaluate(evader.policy, chaser.policy, eval_sampler, mp, ep,
                        n_envs=min(tp.eval_matches, tp.n_envs), max_steps=tp.eval_ep_len)
        n2 = res2["wins"] + res2["losses"] + res2["draws"]
        score2 = (res2["wins"] + 0.5 * res2["draws"]) / max(n2, 1)
        elo_ev.update(elo_ch.rating, score2)
        elo_ch.update(elo_ev.rating, 1.0 - score2)
        ch_pool.append(clone_policy(chaser.policy, boot_ch, tp, tp.seed + 1, device, "chaser"))
        if len(ch_pool) > tp.pool_size:
            ch_pool.pop(0)

        # ---------------- terrain adversary CEM update ----------------
        adv_rec = None
        if block % tp.adv_update_every == 0:
            adv_rec = adv.update(lambda cands: _cand_eval(
                cands, evader.policy, chaser.policy, gen, mp, ep, tp, device))

        # ---------------- block record ----------------
        if block % tp.save_every == 0 or block == tp.blocks - 1:
            evader.save(os.path.join(out_dir, f"evader_b{block:03d}.zip"))
            chaser.save(os.path.join(out_dir, f"chaser_b{block:03d}.zip"))
        ev_eps = [r for r in metrics.episodes if r.get("side") == "evader"]
        tr_ev = metrics.summarize_episodes(ev_eps)
        tr_ch = metrics.summarize_episodes([r for r in metrics.episodes if r.get("side") == "chaser"])
        n_ev_esc = sum(1 for r in ev_eps if r.get("outcome") == "escaped")
        metrics.save_episodes()
        rec = {
            "block": block,
            "elo_evader": round(elo_ev.rating, 1), "elo_chaser": round(elo_ch.rating, 1),
            "eval_ev_win_rate": round(score2, 3),
            "tr_ev_len": tr_ev.get("avg_len", 0), "tr_ch_len": tr_ch.get("avg_len", 0),
            "tr_ev_zone": tr_ev.get("avg_zone", 0),
            "tr_ev_win_rate": tr_ev.get("win_rate", 0),
            "tr_ev_dashes": tr_ev.get("ev_dashes", 0), "tr_ev_walljumps": tr_ev.get("ev_walljumps", 0),
            "tr_ev_djumps": tr_ev.get("ev_djumps", 0), "tr_ev_bashes": tr_ev.get("ev_bashes", 0),
            "tr_ev_airtime": tr_ev.get("ev_airtime", 0),
            "tr_ev_escapes": n_ev_esc,
            "tr_ch_dashes": tr_ch.get("ch_dashes", 0), "tr_ch_walljumps": tr_ch.get("ch_walljumps", 0),
            "tr_ch_airtime": tr_ch.get("ch_airtime", 0),
            "adv_wr": adv_rec["wr_mean"] if adv_rec else None,
            "time_s": round(time.time() - t0, 1),
        }
        metrics.add_block(rec)
        summary = {"evader": tr_ev, "chaser": tr_ch, "adv": adv_rec, "adv_mu": list(adv.mu)}
        print(f"[block {block:3d}] elo ev={rec['elo_evader']:6.1f} ch={rec['elo_chaser']:6.1f} | "
              f"eval ev-wr={rec['eval_ev_win_rate']:.2f} esc={n_ev_esc} | "
              f"len ev={rec['tr_ev_len']:5.0f} ch={rec['tr_ch_len']:5.0f} zone={rec['tr_ev_zone']:3.0f} | "
              f"agility ev(dash={rec['tr_ev_dashes']:.1f} wj={rec['tr_ev_walljumps']:.1f} dj={rec['tr_ev_djumps']:.1f} "
              f"bash={rec['tr_ev_bashes']:.1f}) | adv_wr={rec['adv_wr'] if rec['adv_wr'] is not None else 0:.2f} | {rec['time_s']:.0f}s",
              flush=True)

    # ---------------- persist ----------------
    evader.save(os.path.join(out_dir, "evader.zip"))
    chaser.save(os.path.join(out_dir, "chaser.zip"))
    for i, pol in enumerate(ev_pool):
        torch.save(pol.state_dict(), os.path.join(out_dir, f"pool_ev_{i}.pt"))
    for i, pol in enumerate(ch_pool):
        torch.save(pol.state_dict(), os.path.join(out_dir, f"pool_ch_{i}.pt"))
    np.save(os.path.join(out_dir, "adv_mu.npy"), adv.mu)
    np.save(os.path.join(out_dir, "adv_sigma.npy"), adv.sigma)
    metrics.plot()
    print(f"\nSaved results to {out_dir}: evader.zip, chaser.zip, pool snapshots, curves.png")
    return summary
