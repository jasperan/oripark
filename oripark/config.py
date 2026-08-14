"""Central configuration for the Ori-park physics + RL project.

Movement parameters are tuned to recreate the *feel* of Ori and the Will
of the Wisps: snappy ground control, variable-height jumps with coyote
time and input buffering, one air double-jump, wall slide / wall jump,
an instant dash with momentum retention, and bash launches off orbs.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MoveParams:
    """Ori-inspired movement constants. Units: pixels, seconds."""

    gravity: float = 2600.0
    rise_gravity_mult: float = 0.92          # slight float while rising (Ori feel)
    max_run: float = 430.0                   # top horizontal run speed
    ground_accel: float = 3600.0             # snappy ground acceleration
    ground_decel: float = 4400.0             # snappy ground braking
    air_accel: float = 2900.0                # strong air control (~80% of ground)
    post_dash_decay: float = 900.0           # gentle bleed-off of dash momentum

    jump_v: float = -820.0                   # max jump height ~4 tiles
    jump_cut: float = 0.50                   # releasing early -> shorter hop
    jump_min_hold: float = 0.05              # taps shorter than this keep full height
    jump_buffer: float = 0.15                # input buffer window
    coyote_time: float = 0.10                # grace after leaving a ledge

    djump_v: float = -780.0                  # single air double jump
    wall_slide_max: float = 170.0            # terminal wall-slide fall speed
    wall_slide_accel: float = 1500.0
    walljump_h: float = 640.0                # horizontal kick away from wall
    walljump_v: float = -900.0               # vertical pop (~5 tiles)
    walljump_grace: float = 0.14             # re-stick prevention after wall jump

    dash_speed: float = 1250.0               # burst speed (~39 px/frame at 60Hz)
    dash_time: float = 0.12
    dash_momentum: float = 0.65              # velocity retained after dash ends

    bash_speed: float = 1150.0
    bash_range: float = 100.0
    bash_cooldown: float = 0.22
    bash_refresh: bool = True                # bash refreshes dash + double jump

    half_w: float = 13.0
    half_h: float = 15.0
    tile: int = 32
    substeps: int = 2                        # physics substeps per 60 Hz frame (120 Hz)
    step_hz: int = 60

    # arena dimensions (tiles)
    arena_w: int = 60
    arena_h: int = 40


@dataclass
class EnvParams:
    max_steps: int = 1500                    # 25 s at 60 Hz
    catch_dist: float = 26.0                 # AABB half-extent sum for tag
    evader_spawn_x: int = 3
    chaser_spawn_frac: float = 0.65          # fraction of arena width for chaser spawn
    # reward shaping: escape-oriented (portal >> survive-to-timeout)
    r_time: float = 0.002                    # tiny survival tick (timeout ≈ 0 total)
    r_dist_gain: float = 0.010               # per 50 px of separation change (secondary)
    r_portal_progress: float = 0.05          # per 100 px of portal-distance reduction (primary)
    r_milestone: float = 0.4                 # first-time crossing of each 128 px rightward zone
    r_pass: float = 1.0                      # one-time bonus for getting past the chaser
    r_proximity: float = 0.004
    r_agility: float = 0.06
    r_portal: float = 3.0                    # clean escape — the dominant objective
    r_caught: float = -2.5
    r_hazard: float = -2.0
    r_timeout: float = -3.0                  # failing to escape is a loss for the evader
    patch_w: int = 13
    patch_h: int = 9
    arena_difficulty_mix: float = 0.15       # chance a reset uses mean params


@dataclass
class TrainParams:
    n_envs: int = 16
    n_steps: int = 512                      # rollout steps per env (SB3)
    n_epochs: int = 4
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 2.5e-4
    policy_net: list = field(default_factory=lambda: [256, 256])
    # asymmetric self-play: the protagonist (evader) gets more capacity and a
    # faster learning rate than the adversary (chaser), keeping the fight fair
    evader_net: list = field(default_factory=lambda: [256, 256])
    chaser_net: list = field(default_factory=lambda: [128, 128])
    evader_lr: float = 3.0e-4
    chaser_lr: float = 1.5e-4
    blocks: int = 150
    block_steps: int = 8192                  # per agent per block
    save_every: int = 10                     # checkpoint models every N blocks (progress evals)
    warmup_blocks: int = 30                  # both sides first train vs random opponents
    pool_size: int = 8                       # opponent snapshots kept per side
    opp_latest_prob: float = 0.6             # P(pick most recent opponent)
    eval_matches: int = 16                   # per block, latest-vs-latest
    eval_ep_len: int = 360                   # 6 s cap for eval matches
    adv_update_every: int = 4                # terrain adversary CEM cadence
    adv_pop: int = 6
    adv_elites: int = 3
    adv_matches: int = 2
    adv_sigma0: float = 0.30
    adv_target_wr: float = 0.50
    seed: int = 0
    out_dir: str = "results/run1"
