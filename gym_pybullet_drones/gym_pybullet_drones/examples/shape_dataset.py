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
with columns:

    step, t, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
    roll, pitch, yaw, ang_vel_x, ang_vel_y, ang_vel_z,
    action_vx, action_vy, action_vz, action_yaw_rate,
    target_pos_x, target_pos_y, target_pos_z,
    reward, done,
    shape, center_x, center_y, center_z, start_yaw_deg, tilt_deg, tilt_axis_deg,
    max_speed, max_accel

`reward` is the negative squared tracking error (-(pos - target_pos)^2), consistent with
the time+error^2 loss used to pick `--speed_margin`; `done` is True only on an episode's
last row (each CSV file is one episode, so this just marks that boundary explicitly for
later use once multiple episodes' CSVs get concatenated).

"""
import os
import csv
import time
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

#### Regular polygons approximating each shape, by number of sides ##########
SHAPE_SIDES = {'triangle': 3, 'square': 4, 'pentagon': 5, 'circle': 72}

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
DEFAULT_LOOKAHEAD_DIST = 0.3  # meters
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_SEED = None
DEFAULT_PLOT = False
DEFAULT_PLOT_PATH = False


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

    def __init__(self, path, speed_profile, lookahead_steps, max_speed, max_accel, control_freq_hz):
        self.path = path
        self.speed_profile = speed_profile
        self.n = len(path)
        self.lookahead_steps = lookahead_steps
        self.max_speed = max_speed
        self.max_delta_v = max_accel / control_freq_hz
        self.prev_target_vel = np.zeros(3)

    def step(self, cur_pos):
        """Returns (target_vel, closest_idx) for the drone's current position."""
        closest_idx = int(np.argmin(np.linalg.norm(self.path - cur_pos, axis=1)))
        lookahead_idx = (closest_idx + self.lookahead_steps) % self.n
        to_lookahead = self.path[lookahead_idx] - cur_pos
        dist = np.linalg.norm(to_lookahead)
        direction = to_lookahead / dist if dist > 1e-6 else np.zeros(3)
        raw_target_vel = direction * self.speed_profile[lookahead_idx]

        prev_speed = np.linalg.norm(self.prev_target_vel)
        raw_speed = np.linalg.norm(raw_target_vel)
        if raw_speed > prev_speed and self.max_speed > 0:
            #### Less and less accel headroom left as we approach max_speed ####
            headroom = max(0.0, 1 - (prev_speed / self.max_speed) ** 2)
            max_delta_v = self.max_delta_v * headroom
        else:
            max_delta_v = self.max_delta_v  # braking is never limited by the taper

        delta = raw_target_vel - self.prev_target_vel
        dmag = np.linalg.norm(delta)
        target_vel = self.prev_target_vel + delta * min(1, max_delta_v / max(dmag, 1e-9))
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
        ):
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

    local_pts = generate_local_shape_waypoints(shape, path_resolution, radius, side_jitter, rng)
    TARGET_POS = place_waypoints(local_pts,
                                  start_yaw=np.radians(ep['start_yaw_deg']),
                                  tilt_deg=np.radians(ep['tilt_deg']),
                                  tilt_axis_deg=np.radians(ep['tilt_axis_deg']),
                                  center=ep['center'])
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
    tracker = PurePursuitTracker(TARGET_POS, SPEED_PROFILE, lookahead_steps, max_speed, max_accel, control_freq_hz)

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
        'step', 't',
        'pos_x', 'pos_y', 'pos_z',
        'vel_x', 'vel_y', 'vel_z',
        'roll', 'pitch', 'yaw',
        'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
        'action_vx', 'action_vy', 'action_vz', 'action_yaw_rate',
        'target_pos_x', 'target_pos_y', 'target_pos_z',
        'reward', 'done',
        'shape', 'center_x', 'center_y', 'center_z', 'start_yaw_deg', 'tilt_deg', 'tilt_axis_deg',
        'max_speed', 'max_accel',
    ])
    episode_meta = [shape, *ep['center'], ep['start_yaw_deg'], ep['tilt_deg'], ep['tilt_axis_deg'], max_speed, max_accel]
    actual_pos = [] if plot_path else None

    #### Run the simulation ######################################
    action = np.zeros((1, 4))
    START = time.time()
    total_steps = int(duration_sec * control_freq_hz)
    for i in range(total_steps):
        #### CtrlAviary's own reward/terminated/truncated are unused dummies (it's not
        #### built for RL) -- ours is computed below instead, from the actual tracking error.
        obs, _, _, _, info = env.step(action)
        state = obs[0]

        #### Pure velocity control: target_pos = current position, so the PID's
        #### position term (P and I) always sees zero error and contributes nothing.
        #### target_vel comes from the pure-pursuit tracker, re-anchored every step to
        #### the drone's actual position, so small deviations self-correct instead of
        #### compounding (dead-reckoning drift). Motion -- and therefore the state
        #### transition -- is driven only by target_vel, which is exactly what's logged
        #### as `action` below: (state, action) stays self-consistent for a policy that
        #### will later be rolled out through the same pure-velocity controller.
        target_vel, closest_idx = tracker.step(state[0:3])
        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state,
            target_pos=state[0:3],
            target_rpy=INIT_RPY[0],
            target_vel=target_vel,
        )

        #### reward = -(tracking error)^2, i.e. how close `pos` landed to the reference
        #### path point pure pursuit was aiming at that step -- consistent with the
        #### time+error^2 loss used to pick speed_margin (see compute_time_optimal_speed_profile).
        tracking_error_sq = float(np.sum((state[0:3] - TARGET_POS[closest_idx]) ** 2))
        step_reward = -tracking_error_sq
        done = (i == total_steps - 1)

        writer.writerow([
            i, i / control_freq_hz,
            *state[0:3],
            *state[10:13],
            *state[7:10],
            *state[13:16],
            *target_vel, 0.0,
            *TARGET_POS[closest_idx],
            step_reward, done,
            *episode_meta,
        ])
        if logger is not None:
            logger.log(drone=0, timestamp=i / control_freq_hz, state=state,
                       control=np.hstack([TARGET_POS[closest_idx], target_vel, np.zeros(6)]))
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
    ARGS = parser.parse_args()

    run(**vars(ARGS))
