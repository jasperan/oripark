"""Behavior cloning: pretrain the evader policy from scripted expert demos.

The scripted expert (oripark.scripted) escapes ~77% of arenas even against
the frozen chaser — far better than the raw RL policy (30%). We use its
rollouts as supervised demonstrations so the NN starts self-play already
knowing how to traverse, then RL refines the escape under chaser pressure.
"""
from __future__ import annotations

import numpy as np
import torch


def behavior_clone(policy, obs_list, act_list, epochs: int = 10,
                   lr: float = 3e-4, batch: int = 1024, device: str = "cpu"):
    """Supervised imitation of (obs, act) pairs. obs_list/act_list are lists
    of per-episode arrays. Returns per-epoch mean NLL loss."""
    obs = np.concatenate([np.asarray(o, dtype=np.float32) for o in obs_list])
    acts = np.concatenate([np.asarray(a, dtype=np.int64) for a in act_list])
    n = len(obs)
    rng = np.random.default_rng(0)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    losses = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot, cnt = 0.0, 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            ob = torch.as_tensor(obs[idx], device=device)
            ac = torch.as_tensor(acts[idx], device=device)
            dist = policy.get_distribution(ob)
            logp = dist.log_prob(ac)
            loss = -logp.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            cnt += 1
        losses.append(tot / max(cnt, 1))
        print(f"  bc epoch {ep + 1}/{epochs}: nll={losses[-1]:.4f}", flush=True)
    return losses


def collect_and_clone(env, scripts, policy, n_episodes: int = 400,
                      max_steps: int = 1500, epochs: int = 8, lr: float = 3e-4,
                      device: str = "cpu", require_escape: bool = True):
    """Collect demos with the scripted evader and BC-train `policy`."""
    from oripark.scripted import collect_demos
    print(f"[bc] collecting {n_episodes} scripted demos...", flush=True)
    obs_list, act_list, stats = collect_demos(
        env, scripts, n_episodes=n_episodes, max_steps=max_steps,
        require_escape=require_escape, seed=0)
    print(f"[bc] collected {stats['episodes']} escaped episodes, "
          f"{stats['pairs']} (obs, act) pairs", flush=True)
    if not obs_list:
        print("[bc] WARNING: no demos collected — skipping BC", flush=True)
        return []
    losses = behavior_clone(policy, obs_list, act_list, epochs=epochs,
                            lr=lr, device=device)
    return losses
