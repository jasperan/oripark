"""Which checkpoint is a run's *pre-training* baseline?

`pool_ev_0.pt` looks like one, and every tool in this repo used to label it
"random" / "untrained" — including the progress plots, the hero GIF's
UNTRAINED panel and the random-population floor. It is not one. The league
trims its pool front (`ev_pool.pop(0)`, selfplay.py) once the pool exceeds
`pool_size`, and the final save writes `pool_ev_{i}` from the *trimmed* list,
so `pool_ev_0` is the oldest surviving **trained** snapshot.

Evidence: its weight L2 norm matches the finished policy (77-92) rather than a
fresh init (46), and on the hard-arena protocol it escapes 18-42% where five
fresh inits escape 0%.

Runs from now on write `pool_ev_boot.pt` / `pool_ch_boot.pt` at startup, so the
true baseline survives the trim. For older runs the best pre-training proxy is
`evader_init.pt` (the initial policy, replaced by the BC policy when behaviour
cloning runs).
"""
from __future__ import annotations

import os

BOOT = "pool_ev_boot.pt"
INIT = "evader_init.pt"
POOL0 = "pool_ev_0.pt"

# label -> (filename preference)
BASELINE_LADDER = (
    ("untrained init", BOOT),          # written by runs after this fix
    ("BC/init policy", INIT),          # what older runs have
)
POOL0_LABEL = "pool b0 (early trained snapshot)"


def find_baseline(run_dir: str):
    """Return (path, label) for the best available pre-training checkpoint."""
    for label, name in BASELINE_LADDER:
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            return p, label
    return None, None


def find_pool0(run_dir: str):
    p = os.path.join(run_dir, POOL0)
    return (p, POOL0_LABEL) if os.path.exists(p) else (None, None)
