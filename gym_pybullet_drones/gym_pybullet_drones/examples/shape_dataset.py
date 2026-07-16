"""Script for collecting a (state, action) dataset of a single drone tracing a randomized-polygon path.

The simulation is run by a `CtrlAviary` environment. The control is given by the PID
implementation in `DSLPIDControl`, used in *pure velocity* mode (as in `VelocityAviary`):
`target_pos` is always set to the drone's current position, so the position P/I terms
contribute nothing and motion is driven only by `target_vel`. This keeps every logged
(state, action) pair self-consistent -- `action` (the target velocity) is what actually
produced the next state -- which matters for a policy meant to be rolled out later
through the same pure-velocity controller.

Supported shapes are polygons built around a center point: `triangle` (3 sides),
`square` (4 sides), `pentagon` (5 sides), and `circle` (approximated as a regular
72-sided polygon, which never varies except in size).

For triangle/square/pentagon, each side's length is independently randomized
(`--side_jitter`) by perturbing each vertex's distance from the shape's center --
vertices stay in increasing angular order around the center, so the polygon is
always simple (non-self-intersecting) by construction, no extra checks needed.

Each run randomizes, within a `--workspace_size`-meter cube (default 5x5x5m):
  - the shape's placement (X-Y-Z center),
  - its starting rotation (yaw) around Z,
  - a small tilt of the shape's plane away from horizontal (`--tilt_max_deg`).
Pass `--seed` for reproducible draws.

Example
-------
In a terminal, run as:

    $ python shape_dataset.py --shape triangle
    $ python shape_dataset.py --shape square --gui False --duration_sec 30 --seed 0

Output
------
One row per control step is appended to a CSV under `<output_folder>/shape_dataset/`,
with columns, matching the (s, a, r, s') schema used directly by the offline-RL training
code (s' is just the next row's s within the same episode):

    step,                   # step index within the episode (0-based)
    tx-x, ty-y, tz-z,       # s[0:3]  -- target_pos - pos (position error, not absolute position)
    qx, qy, qz, qw,         # s[3:7]  -- orientation quaternion (PyBullet's native quat, no Euler
                            #            round-trip -- avoids roll/pitch/yaw gimbal-lock/wraparound)
    vx, vy, vz,             # s[7:10] -- velocity
    wx, wy, wz,             # s[10:13]-- angular velocity
    ax, ay, az,             # a[0:3]  -- action (target velocity); yaw rate is dropped, always 0 here
    reward, done

No per-episode metadata (shape, center, tilt, max_speed/max_accel) is logged -- it isn't part
of the training tuple, so it's dropped to save space (each dataset's max_speed/max_accel
choice is still recoverable from which `--output_folder` it was collected into).

`reward` is the negative tracking-error distance (-|pos - target_pos|); `done` is True only
on an episode's last row (each CSV file is one episode, so this just marks that boundary
explicitly for
later use once multiple episodes' CSVs get concatenated).

"""
import os
import csv
import time
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

#### Regular polygons approximating each shape, by number of sides ##########
SHAPE_SIDES = {'triangle': 3, 'square': 4, 'pentagon': 5, 'circle': 72, 'star': 10}
STAR_INNER_RATIO = 0.45  # star's inner-vertex radius as a fraction of the outer radius (5-pointed
# star = 10 vertices with radius alternating outer/inner). A concave shape the policy never trained
# on -- used to test generalization to a genuinely new shape family.

DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_SHAPE = 'triangle'
DEFAULT_PHYSICS = Physics("pyb_drag")
DEFAULT_GUI = True
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 1000
DEFAULT_CONTROL_FREQ_HZ = 100
DEFAULT_RADIUS = 2.2
DEFAULT_SIDE_JITTER = 0.3
DEFAULT_TILT_MAX_DEG = 30
DEFAULT_WORKSPACE_SIZE = 5.0
DEFAULT_FLOOR_CLEARANCE = 0.3
DEFAULT_PATH_RESOLUTION = 3000  # waypoints per lap; purely spatial now, decoupled from any duration
DEFAULT_N_LAPS = 3
DEFAULT_DURATION_SEC = None  # None = auto: n_laps * the time-optimal lap time
DEFAULT_MAX_SPEED_MIN = 2.0
DEFAULT_MAX_SPEED_MAX = 2.0  # == min by default -> no randomization unless overridden
DEFAULT_MAX_ACCEL_MIN = 2.0
DEFAULT_MAX_ACCEL_MAX = 2.0
DEFAULT_SPEED_MARGIN = 0.7  # fraction of max_speed/max_accel the *planned* profile targets
DEFAULT_LOOKAHEAD_DIST = 0.3  # meters. (Tried 0.5 to ease corner stalls -- it fixed some
# corners but broke others, net 10/12 -> 9/12; look-ahead length just relocates the stall,
# so reverted. Corner traversal is addressed via corner-focused DAgger instead. Kept as the
# default so collection/DAgger/eval/progress all use the same value -- the lookahead lx/ly/lz
# state feature is only consistent if this matches everywhere.)
DEFAULT_ADAPTIVE_LOOKAHEAD_K = 0.0  # 0 = fixed look-ahead (original). >0 shrinks the look-ahead
# in proportion to off-path deviation: eff_steps = lookahead_steps / (1 + k*deviation). Motivation:
# with a fixed look-ahead, a large off-path error keeps the forward component competitive with the
# recovery pull, so pure-pursuit keeps advancing instead of returning and spirals outward at corners
# (measured: kick 0.2m -> action nearly perpendicular to the return direction, |pos_err| barely
# shrinks). Shrinking look-ahead with deviation makes far-off-path targets approach the closest
# point (near-pure recovery) while on-path keeps full forward progress. NOTE: changing this changes
# the logged lx/ly/lz state feature, so collection/DAgger/eval must all use the SAME k.
DEFAULT_ADAPTIVE_SLEW_K = 0.0  # 0 = fixed slew-rate cap (original). >0 RELAXES the slew cap in
# proportion to off-path deviation: eff_max_delta_v = max_delta_v * (1 + k*deviation). Motivation
# (measured): after a kick the raw target already points almost perfectly back to the path
# (cos~0.99) at full speed, but the slew cap crushes the COMMANDED speed to ~1/3 while it rotates
# prev_target_vel from "forward" to "return", so recovery is slow. Relaxing the cap when far
# off-path lets the command swing to the return direction fast. Trade-off: a larger commanded
# accel may exceed what the low-level PID / sim physics can actually track -- logged and checked.
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_SEED = None
DEFAULT_PLOT = False
DEFAULT_PLOT_PATH = False
DEFAULT_ATT_D_GAIN_SCALE = 1.0  # 1.0 = DSLPIDControl's unmodified (real-hardware-tuned) gains
DEFAULT_PERTURB_PROB = 0.0  # 0.0 = off, unmodified behavior
DEFAULT_PERTURB_MAGNITUDE = 1.5  # meters, position-kick magnitude when a perturbation fires
DEFAULT_PERTURB_COUNT = 1  # number of independent kicks within an episode that gets perturbed
DEFAULT_OBS_POS_NOISE_STD = 0.0  # meters, 0.0 = off. Gaussian noise added to the *logged*
# position-error columns only (not reward, not the tracker/policy's actual control input) --
# mimics real GPS/position-sensor noise so the trained policy learns "noisy-looking
# observation -> the action that was actually correct", a standard robustness augmentation.


def generate_local_shape_waypoints(shape, num_wp, radius, side_jitter, rng):
    """Samples `num_wp` waypoints (by arc length) along a closed polygon, in its own local X-Y frame (Z=0).

    Vertices are placed at evenly-spaced angles around the origin, each at an independently
    jittered distance (`radius * (1 +/- side_jitter)`) -- since angles stay in strictly
    increasing order, the resulting polygon is always simple (non-self-intersecting), and
    different vertex distances naturally give each side a different length. `circle` ignores
    `side_jitter` (a circle only has a radius to vary).

    Parameters
    ----------
    shape : str
        One of the keys of `SHAPE_SIDES`.
    num_wp : int
        Number of waypoints to sample along the closed path.
    radius : float
        Base circumradius of the shape, in meters.
    side_jitter : float
        Fractional random perturbation applied to each vertex's radius (0 = regular polygon).
    rng : np.random.Generator
        Source of randomness for the per-vertex radius jitter.

    Returns
    -------
    ndarray
        (num_wp, 3)-shaped array of waypoints, Z always 0.

    """
    if shape == 'circle':
        #### Use the full sample resolution as the vertex count, not SHAPE_SIDES['circle'] --
        #### building 72 real corners and then arc-length-resampling to num_wp points would
        #### concentrate each corner's full turning angle onto a single resampled point
        #### (curvature looks like a near-zero-radius kink there instead of the smooth,
        #### constant curvature a real circle has), badly confusing any curvature-based
        #### speed limit downstream (e.g. compute_time_optimal_speed_profile).
        n_sides = num_wp
        vertex_radii = np.full(n_sides, radius)
    elif shape == 'star':
        #### 5-pointed star: 10 vertices at evenly-spaced angles (36 deg apart), radius
        #### ALTERNATING between the outer radius and radius*STAR_INNER_RATIO -- that alternation
        #### is what makes the concave star points. Still built with the same angle-increasing,
        #### arc-length-resampled machinery as the convex shapes below.
        n_sides = SHAPE_SIDES['star']
        base = np.where(np.arange(n_sides) % 2 == 0, radius, radius * STAR_INNER_RATIO)
        vertex_radii = base * (1 + rng.uniform(-side_jitter, side_jitter, size=n_sides))
    else:
        n_sides = SHAPE_SIDES[shape]
        vertex_radii = radius * (1 + rng.uniform(-side_jitter, side_jitter, size=n_sides))
    angles = (np.arange(n_sides + 1) / n_sides) * 2 * np.pi
    r_closed = np.append(vertex_radii, vertex_radii[0])
    verts = np.stack([r_closed * np.cos(angles), r_closed * np.sin(angles)], axis=1)
    edge_vecs = np.diff(verts, axis=0)
    edge_lens = np.linalg.norm(edge_vecs, axis=1)
    cum_lens = np.concatenate([[0], np.cumsum(edge_lens)])
    total_len = cum_lens[-1]
    target_dists = (np.arange(num_wp) / num_wp) * total_len
    xy = np.zeros((num_wp, 2))
    for i, d in enumerate(target_dists):
        edge_idx = min(np.searchsorted(cum_lens, d, side='right') - 1, n_sides - 1)
        frac = 0 if edge_lens[edge_idx] == 0 else (d - cum_lens[edge_idx]) / edge_lens[edge_idx]
        xy[i] = verts[edge_idx] + frac * edge_vecs[edge_idx]
    local = np.zeros((num_wp, 3))
    local[:, 0:2] = xy
    return local


def place_waypoints(local_pts, start_yaw, tilt_deg, tilt_axis_deg, center):
    """Rotates a flat, origin-centered path (start yaw, then tilt) and translates it to `center`."""
    pts = Rotation.from_euler('z', start_yaw).apply(local_pts)
    axis = np.array([np.cos(tilt_axis_deg), np.sin(tilt_axis_deg), 0])
    pts = Rotation.from_rotvec(axis * tilt_deg).apply(pts)
    return pts + np.asarray(center)


def compute_time_optimal_speed_profile(path, max_speed, max_accel, n_smoothing_laps=3):
    """Computes the fastest speed achievable at each point of a closed path under speed/accel limits.

    This is a simplified time-optimal path parameterization (as used in robot/CNC motion
    planning): straight stretches ramp up to `max_speed`; corners are approached only as fast
    as still allows braking to a safe cornering speed within `max_accel`, and accelerated back
    out of just as fast. Unlike guessing a lap duration and deriving speed from it, this instead
    *starts* from the physical limits and derives the fastest consistent lap time -- there's no
    faster way around this path without exceeding `max_speed` or `max_accel`.

    Steps: (1) a per-point corner speed limit from the discretized turning angle, treating the
    local curvature as `angle / avg_segment_length` and capping speed via `v = sqrt(a * radius)`
    (the standard centripetal-acceleration bound); (2) a forward/backward accel-limited sweep
    along arc length (`v[i]^2 <= v[i-1]^2 + 2*a*ds`), run for several laps since the path is a
    closed loop, so a slowdown required near one corner correctly propagates backward into the
    preceding straight to leave room to brake.

    Parameters
    ----------
    path : ndarray
        (N, 3)-shaped array of positions describing the closed path.
    max_speed : float
        Maximum allowed speed, in m/s.
    max_accel : float
        Maximum allowed (tangential and centripetal) acceleration, in m/s^2.
    n_smoothing_laps : int, optional
        Number of forward/backward sweeps around the closed loop.

    Returns
    -------
    ndarray
        (N,)-shaped array, the time-optimal speed (m/s) at each path point.
    float
        Total time (s) to complete one lap at this speed profile.

    """
    n = len(path)
    seg_vec = np.roll(path, -1, axis=0) - path
    ds = np.maximum(np.linalg.norm(seg_vec, axis=1), 1e-9)
    seg_dir = seg_vec / ds[:, None]
    incoming_dir = np.roll(seg_dir, 1, axis=0)  # direction of the segment arriving at each point
    cos_theta = np.clip(np.sum(incoming_dir * seg_dir, axis=1), -1, 1)
    theta = np.arccos(cos_theta)  # turning angle at each point; 0 = straight through

    ds_avg = (ds + np.roll(ds, 1)) / 2
    radius_of_curvature = np.where(theta > 1e-6, ds_avg / np.maximum(theta, 1e-6), np.inf)
    v = np.minimum(max_speed, np.sqrt(max_accel * radius_of_curvature))

    for _ in range(n_smoothing_laps):
        for i in range(n):  # forward: limited by how fast we could have accelerated since i-1
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * max_accel * ds[i - 1]))
        for i in range(n - 1, -1, -1):  # backward: limited by how much braking room remains before i+1
            nxt = (i + 1) % n
            v[i] = min(v[i], np.sqrt(v[nxt] ** 2 + 2 * max_accel * ds[i]))

    dt = ds / ((v + np.roll(v, -1)) / 2)
    loop_time = float(dt.sum())
    return v, loop_time


class PurePursuitTracker:
    """Turns a fixed path into a closed-loop velocity command, so drift self-corrects.

    A velocity command computed from a *fixed time schedule* (e.g. "waypoint i at step i")
    silently assumes the drone is exactly where the schedule expects -- if it lags behind
    (which it always does, since real acceleration is finite), the commanded direction is
    aimed at where the drone *should* be, not where it *is*, and the error keeps compounding
    (dead reckoning drift).

    This tracker instead looks up, every step, the closest point on `path` to the drone's
    *actual* current position, aims a fixed number of steps ahead of that point (the
    "lookahead"), and asks for `speed_profile` at that lookahead point. Because the aim point
    is always re-anchored to the real position, small deviations get steered out instead of
    accumulating. Speed changes are slew-rate-limited (`max_accel`) so the commanded velocity
    itself stays physically achievable, e.g. around sharp corners -- and that accel limit is
    itself tapered to zero as the commanded speed approaches `max_speed`, the same way a real
    drone has less and less thrust margin left to accelerate further the faster it already
    goes (drag grows with speed; more of the fixed max thrust is needed just to fight it).
    Braking (reducing speed) is never tapered -- only *further* acceleration is.

    Parameters
    ----------
    path : ndarray
        (N, 3)-shaped array of positions describing the closed path.
    speed_profile : ndarray
        (N,)-shaped array, the desired speed magnitude (m/s) at each path point.
    lookahead_steps : int
        How many path-points ahead of the closest point to aim at.
    max_speed : float
        Maximum commanded speed, in m/s; also the speed at which the accel taper reaches zero.
    max_accel : float
        Maximum allowed change in the commanded velocity per second at zero speed, in m/s^2.
    control_freq_hz : float
        Rate at which `.step()` is called, in Hz.

    """

    def __init__(self, path, speed_profile, lookahead_steps, max_speed, max_accel, control_freq_hz,
                 adaptive_lookahead_k=0.0, adaptive_slew_k=0.0):
        self.path = path
        self.speed_profile = speed_profile
        self.n = len(path)
        self.lookahead_steps = lookahead_steps
        self.max_speed = max_speed
        self.max_delta_v = max_accel / control_freq_hz
        self.prev_target_vel = np.zeros(3)
        #### Adaptive look-ahead gain: >0 shrinks the look-ahead as the off-path deviation
        #### grows (see step()). 0 = fixed look-ahead (original behavior).
        self.adaptive_lookahead_k = adaptive_lookahead_k
        #### Adaptive slew gain: >0 relaxes the per-step slew cap as deviation grows (see step()).
        #### 0 = fixed slew cap (original behavior). This is the actual fix for the kick-recovery
        #### spiral (the raw target is already correct; the slew cap was throttling the return).
        self.adaptive_slew_k = adaptive_slew_k
        #### Diagnostics for the physical-plausibility check: the last commanded per-step velocity
        #### change (|target_vel - prev|) and the effective cap that allowed it, in m/s.
        self.last_commanded_dv = 0.0
        self.last_eff_max_delta_v = self.max_delta_v

    def step(self, cur_pos):
        """Returns (target_vel, closest_idx) for the drone's current position."""
        closest_idx = int(np.argmin(np.linalg.norm(self.path - cur_pos, axis=1)))
        #### Adaptive look-ahead: the look-ahead vector is (forward component ~ lookahead_dist)
        #### + (recovery component ~ deviation). With a FIXED look-ahead, once the drone is far
        #### off-path the fixed-length forward component keeps "go forward" competitive with the
        #### growing recovery pull, so pure-pursuit keeps advancing along the path instead of
        #### returning -- at a corner this compounds into an outward spiral. Shrinking the
        #### look-ahead in proportion to the deviation makes the target point approach the
        #### CLOSEST point (near-pure recovery) when far off, while restoring the full look-ahead
        #### (forward progress) once back on-path. eff_steps = lookahead_steps / (1 + k*deviation).
        #### deviation = distance to the closest path point (= |pos_err|); shared by adaptive
        #### look-ahead (here) and adaptive slew (below).
        deviation = float(np.linalg.norm(self.path[closest_idx] - cur_pos))
        if self.adaptive_lookahead_k > 0:
            eff_steps = max(1, int(self.lookahead_steps / (1.0 + self.adaptive_lookahead_k * deviation)))
        else:
            eff_steps = self.lookahead_steps
        lookahead_idx = (closest_idx + eff_steps) % self.n
        to_lookahead = self.path[lookahead_idx] - cur_pos
        dist = np.linalg.norm(to_lookahead)
        direction = to_lookahead / dist if dist > 1e-6 else np.zeros(3)
        raw_target_vel = direction * self.speed_profile[lookahead_idx]
        #### The un-slewed target (goal velocity toward the look-ahead point). Exposed so a
        #### DAgger relabel can log THIS as the action label instead of the slew-limited one:
        #### at a large tracking error the slew-limited command is near-zero (physically you
        #### can only change velocity so fast from a standstill), which is a too-weak recovery
        #### label; the raw goal ("head to the path at profile speed") is the right target to
        #### learn, with the slew cap re-applied as a post-step at rollout.
        self.last_raw_target_vel = raw_target_vel
        #### The vector from the drone to the look-ahead point (world frame). Unlike pos_err
        #### (offset to the NEAREST point = perpendicular recovery direction), this points
        #### AHEAD along the path, so it carries the travel/progress direction the policy
        #### needs to keep moving forward once on-path. Logged as lx/ly/lz for a state feature.
        self.last_lookahead_vec = to_lookahead

        prev_speed = np.linalg.norm(self.prev_target_vel)
        raw_speed = np.linalg.norm(raw_target_vel)
        #### Base slew cap, optionally RELAXED when far off-path (adaptive_slew_k>0) so the command
        #### can swing to the already-correct raw return direction fast instead of being throttled.
        base_max_delta_v = (self.max_delta_v * (1.0 + self.adaptive_slew_k * deviation)
                            if self.adaptive_slew_k > 0 else self.max_delta_v)
        if raw_speed > prev_speed and self.max_speed > 0:
            #### Less and less accel headroom left as we approach max_speed ####
            headroom = max(0.0, 1 - (prev_speed / self.max_speed) ** 2)
            max_delta_v = base_max_delta_v * headroom
        else:
            max_delta_v = base_max_delta_v  # braking is never limited by the taper

        delta = raw_target_vel - self.prev_target_vel
        dmag = np.linalg.norm(delta)
        target_vel = self.prev_target_vel + delta * min(1, max_delta_v / max(dmag, 1e-9))
        #### Diagnostics for the physical-plausibility check (commanded per-step dv & the cap used).
        self.last_commanded_dv = float(np.linalg.norm(target_vel - self.prev_target_vel))
        self.last_eff_max_delta_v = float(max_delta_v)
        self.prev_target_vel = target_vel
        return target_vel, closest_idx


def sample_episode_params(shape, radius, side_jitter, tilt_max_deg, workspace_size, floor_clearance, rng):
    """Randomly draws a placement (center, start yaw, tilt) that keeps the shape inside the workspace.

    The workspace is a cube of side `workspace_size`, centered on the X-Y origin, resting on the
    ground (Z from 0 to `workspace_size`), with `floor_clearance` meters kept free at the floor
    and the ceiling.

    """
    max_extent = radius if shape == 'circle' else radius * (1 + side_jitter)
    tilt_deg = rng.uniform(0, tilt_max_deg)
    tilt_axis_deg = rng.uniform(0, 360)
    start_yaw_deg = rng.uniform(0, 360)

    xy_half = workspace_size / 2
    xy_margin = max(0, xy_half - max_extent)
    center_x = rng.uniform(-xy_margin, xy_margin)
    center_y = rng.uniform(-xy_margin, xy_margin)

    vertical_extent = max_extent * np.sin(np.radians(tilt_deg))
    z_lo = floor_clearance + vertical_extent
    z_hi = workspace_size - floor_clearance - vertical_extent
    center_z = rng.uniform(z_lo, z_hi) if z_lo < z_hi else workspace_size / 2

    return dict(center=(center_x, center_y, center_z),
                start_yaw_deg=start_yaw_deg,
                tilt_deg=tilt_deg,
                tilt_axis_deg=tilt_axis_deg)


def run(
        drone=DEFAULT_DRONE,
        shape=DEFAULT_SHAPE,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        radius=DEFAULT_RADIUS,
        side_jitter=DEFAULT_SIDE_JITTER,
        tilt_max_deg=DEFAULT_TILT_MAX_DEG,
        workspace_size=DEFAULT_WORKSPACE_SIZE,
        path_resolution=DEFAULT_PATH_RESOLUTION,
        n_laps=DEFAULT_N_LAPS,
        duration_sec=DEFAULT_DURATION_SEC,
        max_speed_min=DEFAULT_MAX_SPEED_MIN,
        max_speed_max=DEFAULT_MAX_SPEED_MAX,
        max_accel_min=DEFAULT_MAX_ACCEL_MIN,
        max_accel_max=DEFAULT_MAX_ACCEL_MAX,
        speed_margin=DEFAULT_SPEED_MARGIN,
        lookahead_dist=DEFAULT_LOOKAHEAD_DIST,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        seed=DEFAULT_SEED,
        plot=DEFAULT_PLOT,
        plot_path=DEFAULT_PLOT_PATH,
        att_d_gain_scale=DEFAULT_ATT_D_GAIN_SCALE,
        policy_fn=None,
        perturb_prob=DEFAULT_PERTURB_PROB,
        perturb_magnitude=DEFAULT_PERTURB_MAGNITUDE,
        perturb_count=DEFAULT_PERTURB_COUNT,
        obs_pos_noise_std=DEFAULT_OBS_POS_NOISE_STD,
        dagger_relabel=False,
        clockwise=False,
        adaptive_lookahead_k=DEFAULT_ADAPTIVE_LOOKAHEAD_K,
        adaptive_slew_k=DEFAULT_ADAPTIVE_SLEW_K,
        fixed_tilt_deg=None,
        ):
    """`policy_fn`, if given, is called each step as `policy_fn(pos_err, state)` and its
    return value is used as `target_vel` instead of the pure-pursuit tracker's -- lets an
    evaluation script fly a trained policy through this same path/physics setup while
    reusing this function's path generation and reward bookkeeping unchanged. Default
    (None) preserves the original pure-pursuit-only behavior exactly.

    `perturb_prob` (default 0.0 = off, fully backward compatible): probability this episode
    gets `perturb_count` random mid-episode position displacements (spread roughly evenly
    across the middle 60% of the episode, each independently jittered in timing/direction)
    applied directly to the drone's physics state (a `p.resetBasePositionAndOrientation`
    call, not a control input -- velocity/orientation are left untouched), each of magnitude
    drawn from `[0.5, 1.0] * perturb_magnitude` meters in a random direction (default range
    ~0.75-1.5m, matching the scale of tracking-error observed when a trained policy's
    rollout actually diverges). The pure-pursuit tracker re-anchors to actual position every
    step regardless of how it got there, so the steps following each kick record a genuine,
    causally-consistent recovery trajectory -- data the un-perturbed expert never produces
    (it never leaves near-zero tracking error), which is otherwise completely missing from
    this pipeline's datasets. `perturb_count > 1` exists because a single dataset-wide
    perturbation *rate* still only yields a handful of distinct excursion events across a
    whole collection run -- multiplying kicks per perturbed episode is a cheaper way to grow
    the diversity of recovery examples than growing total steps. See
    project_drone_offline_rl memory for why this was added.

    `obs_pos_noise_std` (default 0.0 = off, meters): Gaussian noise added independently
    per-axis to the *logged* `tx-x/ty-y/tz-z` columns only -- not to `reward` (still computed
    from the true error) and not to the tracker/policy's actual control input (still driven
    by the true simulator state). Mimics real position-sensor noise so the trained policy
    sees "state that looks slightly off" paired with the action that was genuinely correct
    for the true state, a standard robustness augmentation. This is complementary to (does
    not substitute for) `perturb_prob`/`perturb_magnitude`: it densifies coverage of small
    deviations near the path, it does not teach recovery from meter-scale excursions.
    """
    rng = np.random.default_rng(seed)

    #### Each episode gets its own max_speed/max_accel, drawn once here -- with
    #### *_min == *_max (the default) this reduces to the old fixed-value behavior, and
    #### skips the rng draw entirely so a fixed-range run still consumes `rng` in exactly
    #### the same order as before, keeping old seeds reproducible (see sample_episode_params
    #### and generate_local_shape_waypoints below, which are the only other rng consumers).
    #### Widening the range gives the collected dataset a mix of speed regimes instead
    #### of every episode being flown at the exact same limits, for a more general
    #### (and more RL-useful) policy target.
    max_speed = max_speed_min if max_speed_min == max_speed_max else rng.uniform(max_speed_min, max_speed_max)
    max_accel = max_accel_min if max_accel_min == max_accel_max else rng.uniform(max_accel_min, max_accel_max)

    #### Sample this episode's placement and build the path ####
    ep = sample_episode_params(shape, radius, side_jitter, tilt_max_deg, workspace_size, DEFAULT_FLOOR_CLEARANCE, rng)
    #### fixed_tilt_deg overrides the randomly-sampled plane tilt (e.g. 90 = a vertical-plane
    #### shape) -- used to test whether the policy generalizes to steep out-of-plane paths.
    if fixed_tilt_deg is not None:
        ep['tilt_deg'] = float(fixed_tilt_deg)

    local_pts = generate_local_shape_waypoints(shape, path_resolution, radius, side_jitter, rng)
    TARGET_POS = place_waypoints(local_pts,
                                  start_yaw=np.radians(ep['start_yaw_deg']),
                                  tilt_deg=np.radians(ep['tilt_deg']),
                                  tilt_axis_deg=np.radians(ep['tilt_axis_deg']),
                                  center=ep['center'])
    #### clockwise=True reverses the waypoint order so the tracker (which always steps toward
    #### increasing path indices) traverses the same closed shape in the opposite direction.
    #### The path is a closed loop, so which point ends up at index 0 (the spawn point) is
    #### irrelevant. Building half a collection with this flag set gives the policy both
    #### traversal directions instead of a counter-clockwise-only bias. SPEED_PROFILE and the
    #### look-ahead vector are computed AFTER this, so they follow the reversed path correctly.
    if clockwise:
        TARGET_POS = TARGET_POS[::-1].copy()
    #### Time-optimal speed at each path point given max_speed/max_accel (see docstring) --
    #### not a guessed lap duration, so lap_time is the fastest this shape can be flown.
    #### The profile itself targets speed_margin * (max_speed, max_accel), not the full 100%:
    #### a profile planned at the bare kinematic limit (e.g. cornering right at max_accel)
    #### leaves the real tracker zero headroom to correct for its own lag/discretization/
    #### drift-correction, which empirically causes *larger* tracking error, not less -- the
    #### margin trades a slightly longer lap for a profile that's actually trackable.
    SPEED_PROFILE, lap_time = compute_time_optimal_speed_profile(TARGET_POS, max_speed * speed_margin, max_accel * speed_margin)
    if duration_sec is None:
        duration_sec = n_laps * lap_time
    print(f"[INFO] max_speed={max_speed:.2f}m/s max_accel={max_accel:.2f}m/s^2 -- "
          f"time-optimal lap time: {lap_time:.2f}s -- {n_laps} laps -> duration_sec={duration_sec:.2f}s")

    perimeter = np.sum(np.linalg.norm(np.roll(TARGET_POS, -1, axis=0) - TARGET_POS, axis=1))
    lookahead_steps = max(1, round(lookahead_dist / (perimeter / path_resolution)))
    tracker = PurePursuitTracker(TARGET_POS, SPEED_PROFILE, lookahead_steps, max_speed, max_accel, control_freq_hz,
                                  adaptive_lookahead_k=adaptive_lookahead_k, adaptive_slew_k=adaptive_slew_k)

    INIT_XYZ = np.array([TARGET_POS[0]])
    INIT_RPY = np.array([[0, 0, 0]])

    #### Create the environment #################################
    env = CtrlAviary(drone_model=drone,
                      num_drones=1,
                      initial_xyzs=INIT_XYZ,
                      initial_rpys=INIT_RPY,
                      physics=physics,
                      pyb_freq=simulation_freq_hz,
                      ctrl_freq=control_freq_hz,
                      gui=gui,
                      obstacles=obstacles,
                      )
    ctrl = DSLPIDControl(drone_model=drone)
    #### DSLPIDControl's default attitude D-gain is tuned/validated for normal position-target
    #### control (see cf.py), not this pipeline's "target_pos=cur_pos, driven only via target_vel"
    #### mode -- which shows a sustained ~1Hz +/-11deg roll/pitch oscillation under the defaults.
    #### Scaling it down (att_d_gain_scale < 1) is an experiment to damp that out in *this*
    #### velocity-only mode; only applied to this instance, so DSLPIDControl's real-hardware
    #### defaults (used elsewhere in the repo) are untouched.
    ctrl.D_COEFF_TOR = ctrl.D_COEFF_TOR * att_d_gain_scale
    #### Reuses gym_pybullet_drones' own Logger (position/velocity/attitude vs. time,
    #### one subplot per axis) -- off by default so batch collection never pops a window.
    logger = Logger(logging_freq_hz=control_freq_hz, num_drones=1,
                     duration_sec=int(np.ceil(duration_sec)), output_folder=output_folder) if plot else None

    #### Prepare the output CSV #################################
    dataset_dir = os.path.join(output_folder, 'shape_dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    seed_suffix = f"-seed{seed}" if seed is not None else ""
    csv_path = os.path.join(dataset_dir, f"{shape}{seed_suffix}-{datetime.now().strftime('%m.%d.%Y_%H.%M.%S.%f')}.csv")
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        'step',
        'tx-x', 'ty-y', 'tz-z',
        'qx', 'qy', 'qz', 'qw',
        'vx', 'vy', 'vz',
        'wx', 'wy', 'wz',
        'lx', 'ly', 'lz',
        'ax', 'ay', 'az',
        'reward', 'done',
    ])
    actual_pos = [] if plot_path else None

    #### Run the simulation ######################################
    action = np.zeros((1, 4))
    START = time.time()
    total_steps = int(duration_sec * control_freq_hz)

    #### perturb_prob==0 (default) draws nothing from `rng`, so existing seeds/episodes are
    #### reproduced exactly unless a caller opts in -- same pattern as max_speed/max_accel above.
    #### Kicks are spread across the middle 60% of the episode (evenly-spaced base points,
    #### each jittered +/-5% of total_steps) so multiple kicks land at distinct, separated
    #### times rather than clustering.
    perturb_map = {}
    if perturb_prob > 0 and rng.uniform() < perturb_prob:
        n_kicks = max(1, int(perturb_count))
        base_steps = np.linspace(0.2, 0.8, n_kicks) * total_steps
        for base_step in base_steps:
            step_i = int(base_step + rng.uniform(-0.05, 0.05) * total_steps)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            perturb_map[step_i] = direction * rng.uniform(0.5 * perturb_magnitude, perturb_magnitude)

    for i in range(total_steps):
        #### CtrlAviary's own reward/terminated/truncated are unused dummies (it's not
        #### built for RL) -- ours is computed below instead, from the actual tracking error.
        obs, _, _, _, info = env.step(action)
        state = obs[0]

        #### Directly overrides the drone's physics position (not a control input, velocity/
        #### orientation untouched) so the *next* step() starts from a genuinely off-path
        #### state -- the tracker/PID then have to recover from it for real, producing
        #### causally-consistent recovery data instead of anything synthetic. This step's
        #### own row is unaffected (state above was already read before the kick); only
        #### step i+1 onward reflects it.
        if i in perturb_map:
            cur_pos, cur_quat = p.getBasePositionAndOrientation(env.DRONE_IDS[0], physicsClientId=env.CLIENT)
            new_pos = np.array(cur_pos) + perturb_map[i]
            p.resetBasePositionAndOrientation(env.DRONE_IDS[0], new_pos.tolist(), cur_quat,
                                               physicsClientId=env.CLIENT)
            print(f"[INFO] perturbation applied at step {i}: offset={perturb_map[i].round(2)} m")

        #### Pure velocity control: target_pos = current position, so the PID's
        #### position term (P and I) always sees zero error and contributes nothing.
        #### target_vel comes from the pure-pursuit tracker, re-anchored every step to
        #### the drone's actual position, so small deviations self-correct instead of
        #### compounding (dead-reckoning drift). Motion -- and therefore the state
        #### transition -- is driven only by target_vel, which is exactly what's logged
        #### as `action` below: (state, action) stays self-consistent for a policy that
        #### will later be rolled out through the same pure-velocity controller.
        #### DAgger relabel: the drone is driven by the policy (to visit the states the
        #### policy actually reaches -- the whole point of DAgger), so the tracker's own
        #### slew memory (prev_target_vel) is stale. Reset it to the drone's true current
        #### velocity so the pure-pursuit label is the physically-reachable expert command
        #### from where the drone really is.
        if dagger_relabel and policy_fn is not None:
            tracker.prev_target_vel = state[10:13].copy()

        pursuit_vel, closest_idx = tracker.step(state[0:3])
        lookahead_vec = tracker.last_lookahead_vec  # drone -> look-ahead point (progress direction)

        #### reward = -(tracking error distance), i.e. how close `pos` landed to the reference
        #### path point pure pursuit was aiming at that step.
        pos_err = TARGET_POS[closest_idx] - state[0:3]

        #### policy_fn gets the look-ahead vector too (progress-direction state feature); a
        #### policy that doesn't use it just ignores the 3rd arg.
        exec_vel = policy_fn(pos_err, state, lookahead_vec) if policy_fn is not None else pursuit_vel

        #### The drone always moves under `exec_vel`. The LOGGED action is the label to
        #### learn: in DAgger mode it's pure-pursuit's expert answer for this (on-policy)
        #### state, not what the policy did -- that's what teaches recovery on the states
        #### the policy visits. `raw` (un-slewed goal velocity) is the stronger label at large
        #### errors (see tracker.last_raw_target_vel); the slew cap is re-applied at rollout.
        #### Otherwise (plain collection or plain policy rollout) the logged action is exec_vel.
        if dagger_relabel and policy_fn is not None:
            logged_vel = tracker.last_raw_target_vel
        else:
            logged_vel = exec_vel

        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state,
            target_pos=state[0:3],
            target_rpy=INIT_RPY[0],
            target_vel=exec_vel,
        )

        step_reward = -float(np.linalg.norm(pos_err))
        done = (i == total_steps - 1)

        #### Logged separately from the true values -- reward and the control decision above
        #### both used the true (noiseless) error/look-ahead; only the CSV's state columns get
        #### the noisy version, so this can't leak into causal consistency.
        #### ONE shared position-measurement noise vector is subtracted from BOTH pos_err and
        #### the look-ahead vector: real GPS/position error shifts the drone's *estimated
        #### position*, and pos_err (= target - pos) and lookahead (= lookahead_pt - pos) are
        #### both measured relative to that same position, so a position error of `n` moves
        #### both by `-n`. Applying independent noise to each (or only to pos_err) would be
        #### physically inconsistent and let the policy lean on a wrongly-clean look-ahead.
        if obs_pos_noise_std == 0:
            logged_pos_err, logged_lookahead = pos_err, lookahead_vec
        else:
            pos_noise = rng.normal(0, obs_pos_noise_std, size=3)
            logged_pos_err = pos_err - pos_noise
            logged_lookahead = lookahead_vec - pos_noise

        writer.writerow([
            i,
            *logged_pos_err,
            *state[3:7],
            *state[10:13],
            *state[13:16],
            *logged_lookahead,
            *logged_vel,
            step_reward, done,
        ])
        if logger is not None:
            logger.log(drone=0, timestamp=i / control_freq_hz, state=state,
                       control=np.hstack([TARGET_POS[closest_idx], logged_vel, np.zeros(6)]))
        if actual_pos is not None:
            actual_pos.append(state[0:3].copy())

        env.render()
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    csv_file.close()
    env.close()
    print(f"[INFO] Dataset saved to {csv_path}")
    if logger is not None:
        logger.plot()
    if actual_pos is not None:
        actual_pos = np.array(actual_pos)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(TARGET_POS[:, 0], TARGET_POS[:, 1], TARGET_POS[:, 2],
                'k--', linewidth=1, label='target path')
        ax.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2],
                'b-', linewidth=1.5, label='actual (flown) path')
        #### matplotlib's 3D axes don't share a common scale by default -- each of X/Y/Z is
        #### independently stretched to fill the plot box, so a mostly-flat shape (small
        #### tilt_deg, so its Z range is much smaller than X/Y) ends up looking stood on
        #### edge. Force a common scale by giving all three axes the same range.
        all_pts = np.vstack([TARGET_POS, actual_pos])
        mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
        mid = (mins + maxs) / 2
        half_range = max((maxs - mins).max(), 1e-6) / 2
        ax.set_xlim(mid[0] - half_range, mid[0] + half_range)
        ax.set_ylim(mid[1] - half_range, mid[1] + half_range)
        ax.set_zlim(mid[2] - half_range, mid[2] + half_range)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{shape} -- target vs. actual path')
        ax.legend()
        plt.show()
    return total_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect a state-action dataset of a drone tracing a randomized polygon path')
    parser.add_argument('--drone',              default=DEFAULT_DRONE,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--shape',              default=DEFAULT_SHAPE,     type=str,           help='Path shape (default: triangle)', metavar='', choices=SHAPE_SIDES.keys())
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,   type=Physics,       help='Physics updates (default: PYB_DRAG)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES, type=str2bool,      help='Whether to add obstacles to the environment (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,  help='Simulation frequency in Hz (default: 1000)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,    type=int,  help='Control frequency in Hz (default: 100)', metavar='')
    parser.add_argument('--radius',             default=DEFAULT_RADIUS,    type=float,         help='Base circumradius of the shape in meters, before jitter (default: 2.2)', metavar='')
    parser.add_argument('--side_jitter',        default=DEFAULT_SIDE_JITTER, type=float,       help='Fractional random per-side length variation, 0=regular polygon (default: 0.3); ignored for circle', metavar='')
    parser.add_argument('--tilt_max_deg',       default=DEFAULT_TILT_MAX_DEG, type=float,      help='Max random tilt of the shape plane away from horizontal, in degrees (default: 30)', metavar='')
    parser.add_argument('--workspace_size',     default=DEFAULT_WORKSPACE_SIZE, type=float,    help='Side length in meters of the cubic workspace the shape is randomly placed within (default: 5.0)', metavar='')
    parser.add_argument('--path_resolution',    default=DEFAULT_PATH_RESOLUTION, type=int,     help='Number of waypoints sampled along the path (spatial resolution, independent of duration) (default: 3000)', metavar='')
    parser.add_argument('--n_laps',             default=DEFAULT_N_LAPS, type=float,            help='Number of laps to fly when --duration_sec is not set (default: 3)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC, type=float,      help='Total duration of the simulation in seconds (default: auto = n_laps * time-optimal lap time)', metavar='')
    parser.add_argument('--max_speed_min',      default=DEFAULT_MAX_SPEED_MIN, type=float,     help='Lower bound (m/s) episode max speed is drawn from -- set equal to --max_speed_max for a fixed value (default: 2.0)', metavar='')
    parser.add_argument('--max_speed_max',      default=DEFAULT_MAX_SPEED_MAX, type=float,     help='Upper bound (m/s) episode max speed is drawn from (default: 2.0)', metavar='')
    parser.add_argument('--max_accel_min',      default=DEFAULT_MAX_ACCEL_MIN, type=float,     help='Lower bound (m/s^2) episode max acceleration is drawn from -- also drives the time-optimal speed profile\'s cornering/ramp limits (default: 2.0)', metavar='')
    parser.add_argument('--max_accel_max',      default=DEFAULT_MAX_ACCEL_MAX, type=float,     help='Upper bound (m/s^2) episode max acceleration is drawn from (default: 2.0)', metavar='')
    parser.add_argument('--speed_margin',       default=DEFAULT_SPEED_MARGIN, type=float,      help='Fraction of max_speed/max_accel the planned profile targets, leaving headroom for real tracking error (default: 0.7)', metavar='')
    parser.add_argument('--lookahead_dist',     default=DEFAULT_LOOKAHEAD_DIST, type=float,    help='Pure-pursuit lookahead distance in meters (default: 0.3)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,       help='Folder where to save the dataset (default: "results")', metavar='')
    parser.add_argument('--seed',               default=DEFAULT_SEED,      type=int,           help='Random seed for shape/placement sampling (default: random)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,      type=str2bool,      help='Show a position/velocity/attitude vs. time plot (via Logger) after the run (default: False)', metavar='')
    parser.add_argument('--plot_path',          default=DEFAULT_PLOT_PATH, type=str2bool,      help='Show a 3D plot comparing the target path to the actual flown path after the run (default: False)', metavar='')
    parser.add_argument('--att_d_gain_scale',   default=DEFAULT_ATT_D_GAIN_SCALE, type=float,   help='Scales DSLPIDControl.D_COEFF_TOR (attitude D-gain) for this run only, e.g. 0.5 to damp roll/pitch chatter under velocity-only control (default: 1.0 = unmodified)', metavar='')
    parser.add_argument('--perturb_prob',       default=DEFAULT_PERTURB_PROB, type=float,      help='Probability this episode gets perturb_count random mid-episode position displacements, to record off-path recovery data (default: 0.0 = off)', metavar='')
    parser.add_argument('--perturb_magnitude',  default=DEFAULT_PERTURB_MAGNITUDE, type=float, help='Max meters of each position displacement when perturbation fires (default: 1.5)', metavar='')
    parser.add_argument('--perturb_count',      default=DEFAULT_PERTURB_COUNT, type=int,       help='Number of independent kicks within an episode that gets perturbed (default: 1)', metavar='')
    parser.add_argument('--obs_pos_noise_std',  default=DEFAULT_OBS_POS_NOISE_STD, type=float, help='Meters of Gaussian position-measurement noise; one shared noise vector is applied to the logged tx-x/ty-y/tz-z AND lx/ly/lz columns (not reward, not control); default: 0.0 = off', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
