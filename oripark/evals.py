"""Shared episode-playing loop for the CLI evaluation tools.

Every finished episode counts in the denominator, including the hazard
outcomes (`ev_hazard`: the evader hit spikes, `ch_hazard`: the chaser did).
Three separate copies of this loop used to drop hazards from `n` and break
out of the loop early on `esc + caught + timeout`, which inflated escape
rates — and by more for weak policies, since those die on spikes most.
"""
from __future__ import annotations

import numpy as np

OUTCOMES = ("escaped", "caught", "timeout", "ev_hazard", "ch_hazard")


def play_episodes(env, act_fn, max_steps: int, n_target: int) -> dict:
    """Play until `n_target` episodes have finished.

    act_fn(obs) -> actions for the learner side. Returns outcome counts
    (all five), escape rate, mean traversal zone and mean episode length,
    and closes the env.
    """
    obs = env.reset()
    counts = {o: 0 for o in OUTCOMES}
    zones, lens = [], []
    n_done = 0
    for _ in range(max_steps):
        obs, _, done, infos = env.step(act_fn(obs))
        for info in infos:
            if "episode" in info:
                out = info["outcome"]
                counts[out] = counts.get(out, 0) + 1
                n_done += 1
                zones.append(info.get("ev_max_zone", 0))
                lens.append(info["episode"]["l"])
        if n_done >= n_target:
            break
    env.close()
    return {
        "counts": counts, "n": n_done,
        "esc": counts["escaped"], "caught": counts["caught"], "tm": counts["timeout"],
        "ev_hazard": counts["ev_hazard"], "ch_hazard": counts["ch_hazard"],
        "esc_rate": counts["escaped"] / max(n_done, 1),
        # the evader did not lose: reached the portal, outlasted the clock, or
        # the chaser killed itself. Pairing this with esc_rate separates
        # "could not find the portal" from "died trying".
        "surv_rate": (counts["escaped"] + counts["timeout"]
                      + counts["ch_hazard"]) / max(n_done, 1),
        "zone": float(np.mean(zones)) if zones else 0.0,
        "length": float(np.mean(lens)) if lens else 0.0,
    }
