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
    climb_speed: float = 170.0               # WotW wall climb: hold toward a wall
    fall_gravity_mult: float = 1.12          # snappy WotW descent (float on rise)
    max_fall: float = 1600.0                 # terminal velocity

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
    r_portal_progress: float = 0.10         # per 100 px of portal-distance reduction (primary)
    r_milestone: float = 0.35               # first-time crossing of each 128 px rightward zone
    r_pass: float = 1.0                      # one-time bonus for getting past the chaser
    r_proximity: float = 0.004
    r_agility: float = 0.06
    r_portal: float = 10.0                   # clean escape — the dominant objective
    r_caught: float = -8.0
    r_hazard: float = -8.0
    r_timeout: float = -10.0                 # failing to escape is a loss for the evader
    r_hindsight: float = 2.0                 # credit best-ever portal progress on failure
    # forward-biased observation patch: the game is a rightward escape, and
    # a full jump apex is ~4.5 tiles — the old centered 13x9 patch showed
    # only 4 tiles up, hiding the landing zone at apex. New window: 5 tiles
    # behind, 13 ahead, 7 up, 2 down (19x10 = 190 tiles).
    patch_back: int = 5
    patch_front: int = 13
    patch_up: int = 7
    patch_down: int = 2
    patch_w: int = 19          # back + 1 + front (kept in sync for readability)
    patch_h: int = 10          # up + 1 + down
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
    # chaser-strength curriculum: the evader's opponent latest-prob anneals
    # from opp_latest_start -> opp_latest_end over ladder_blocks, so early
    # blocks face OLD (weak) chaser snapshots and late blocks the strongest
    opp_latest_start: float = 0.10
    opp_latest_end: float = 0.65
    ladder_blocks: int = 120
    bc_episodes: int = 400                   # scripted demos for BC pretraining
    bc_epochs: int = 8                       # BC epochs (0 = skip behavior cloning)
    bc_flee_frac: float = 0.35               # fraction of BC demos collected with a
                                             # pursuer hovering behind (flee patterns)
    bc_pursued_dx: float = -120.0            # fake chaser offset for flee demos
    bc_reg_every: int = 5                    # BC fine-tune the evader every N blocks
    bc_reg_epochs: int = 1                   # BC epochs per fine-tune
    bc_reg_lr_frac: float = 0.2              # fine-tune LR as fraction of evader_lr
    eval_matches: int = 16                   # per block, latest-vs-latest
    eval_ep_len: int = 360                   # 6 s cap for eval matches
    # gap-sprinkling: the CEM never raises gap_scale (wide gaps hurt the
    # chaser too, so its win-rate signal saturates on spike loss instead),
    # leaving the evader untrained on 4-5-tile gap gauntlets. With prob
    # gap_force_prob, a training arena is regenerated with gap forced to
    # gap_force so both agents learn wide-gap traversal.
    gap_force_prob: float = 0.0
    gap_force: float = 0.6
    adv_update_every: int = 4                # terrain adversary CEM cadence
    adv_pop: int = 6
    adv_elites: int = 3
    adv_matches: int = 4
    adv_sigma0: float = 0.30
    adv_target_wr: float = 0.50
    seed: int = 0
    out_dir: str = "results/run1"


# ---------------------------------------------------------------------------
# per-run parameter persistence
# ---------------------------------------------------------------------------
# Every run saves its exact (MoveParams, EnvParams, TrainParams) as
# params.json, and every eval tool loads them back. This is what makes a
# metric "one command away": changing defaults never silently changes how
# old runs are measured.


def params_to_dict(tp: TrainParams, mp: MoveParams, ep: EnvParams) -> dict:
    import dataclasses

    return {
        "train": dataclasses.asdict(tp),
        "move": dataclasses.asdict(mp),
        "env": dataclasses.asdict(ep),
    }


def dict_to_params(d: dict):
    def _fill(cls, data):
        ok = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**ok)

    return (_fill(TrainParams, d["train"]),
            _fill(MoveParams, d["move"]),
            _fill(EnvParams, d["env"]))


def save_run_params(run_dir: str, tp: TrainParams, mp: MoveParams, ep: EnvParams):
    import json
    import os

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(params_to_dict(tp, mp, ep), f, indent=1, sort_keys=True)


def load_run_params(run_dir: str):
    """Load (tp, mp, ep) for a run dir; falls back to current defaults."""
    import json
    import os

    path = os.path.join(run_dir, "params.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return dict_to_params(json.load(f))
        except Exception:
            pass
    return TrainParams(), MoveParams(), EnvParams()
