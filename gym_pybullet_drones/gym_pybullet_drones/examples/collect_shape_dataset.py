"""Batch-collect shape_dataset.py episodes until a target number of environment steps is reached.

Round-robins through `--shapes` (default: all of triangle/square/pentagon/circle), running one
full `shape_dataset.run()` episode (one CSV file) at a time with an incrementing seed. The step
budget is only checked *between* episodes -- an episode already in progress always finishes in
full, so the final total can exceed `--target_steps` by up to one episode's worth of steps, but
is never cut short mid-episode.

Example
-------
In a terminal, run as:

    $ python collect_shape_dataset.py --target_steps 200000
    $ python collect_shape_dataset.py --target_steps 1000000 --shapes triangle square

"""
import argparse

import shape_dataset as sd
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import str2bool

DEFAULT_TARGET_STEPS = 200_000
DEFAULT_SHAPES = list(sd.SHAPE_SIDES.keys())
DEFAULT_SEED_START = 0


def collect(
        target_steps=DEFAULT_TARGET_STEPS,
        shapes=DEFAULT_SHAPES,
        seed_start=DEFAULT_SEED_START,
        drone=sd.DEFAULT_DRONE,
        physics=sd.DEFAULT_PHYSICS,
        gui=False,
        obstacles=sd.DEFAULT_OBSTACLES,
        simulation_freq_hz=sd.DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=sd.DEFAULT_CONTROL_FREQ_HZ,
        radius=sd.DEFAULT_RADIUS,
        side_jitter=sd.DEFAULT_SIDE_JITTER,
        tilt_max_deg=sd.DEFAULT_TILT_MAX_DEG,
        workspace_size=sd.DEFAULT_WORKSPACE_SIZE,
        path_resolution=sd.DEFAULT_PATH_RESOLUTION,
        n_laps=sd.DEFAULT_N_LAPS,
        duration_sec=sd.DEFAULT_DURATION_SEC,
        max_speed_min=sd.DEFAULT_MAX_SPEED_MIN,
        max_speed_max=sd.DEFAULT_MAX_SPEED_MAX,
        max_accel_min=sd.DEFAULT_MAX_ACCEL_MIN,
        max_accel_max=sd.DEFAULT_MAX_ACCEL_MAX,
        speed_margin=sd.DEFAULT_SPEED_MARGIN,
        lookahead_dist=sd.DEFAULT_LOOKAHEAD_DIST,
        output_folder=sd.DEFAULT_OUTPUT_FOLDER,
        att_d_gain_scale=sd.DEFAULT_ATT_D_GAIN_SCALE,
        perturb_prob=sd.DEFAULT_PERTURB_PROB,
        perturb_magnitude=sd.DEFAULT_PERTURB_MAGNITUDE,
        perturb_count=sd.DEFAULT_PERTURB_COUNT,
        obs_pos_noise_std=sd.DEFAULT_OBS_POS_NOISE_STD,
        direction='both',
        ):
    total_steps = 0
    episode_idx = 0
    per_shape_count = {shape: 0 for shape in shapes}
    per_dir_count = {'ccw': 0, 'cw': 0}

    while total_steps < target_steps:
        shape = shapes[episode_idx % len(shapes)]
        seed = seed_start + episode_idx
        #### direction='both' alternates traversal direction every full round through `shapes`
        #### (not every episode) so each shape gets exactly half its episodes clockwise and half
        #### counter-clockwise -- a per-episode `episode_idx % 2` toggle would instead pin each
        #### shape to one direction (period-2 toggle vs period-len(shapes) shape cycle). 'ccw'/
        #### 'cw' force a single direction. Both directions give the policy a traversal-symmetric
        #### dataset instead of a counter-clockwise-only bias.
        if direction == 'both':
            clockwise = ((episode_idx // len(shapes)) % 2 == 1)
        else:
            clockwise = (direction == 'cw')
        print(f"[INFO] episode {episode_idx} -- shape={shape} seed={seed} "
              f"dir={'CW' if clockwise else 'CCW'} "
              f"({total_steps}/{target_steps} steps so far)")
        #### duration_sec is auto (time-optimal lap time * n_laps) unless overridden, and
        #### differs per shape/geometry, so read back the actual step count run() used
        #### rather than assuming a fixed steps-per-episode.
        episode_steps = sd.run(drone=drone, shape=shape, physics=physics, gui=gui, obstacles=obstacles,
                                simulation_freq_hz=simulation_freq_hz, control_freq_hz=control_freq_hz,
                                radius=radius, side_jitter=side_jitter, tilt_max_deg=tilt_max_deg,
                                workspace_size=workspace_size, path_resolution=path_resolution,
                                n_laps=n_laps, duration_sec=duration_sec,
                                max_speed_min=max_speed_min, max_speed_max=max_speed_max,
                                max_accel_min=max_accel_min, max_accel_max=max_accel_max,
                                speed_margin=speed_margin, lookahead_dist=lookahead_dist,
                                output_folder=output_folder, seed=seed,
                                att_d_gain_scale=att_d_gain_scale,
                                perturb_prob=perturb_prob, perturb_magnitude=perturb_magnitude,
                                perturb_count=perturb_count, obs_pos_noise_std=obs_pos_noise_std,
                                clockwise=clockwise)
        total_steps += episode_steps
        per_shape_count[shape] += 1
        per_dir_count['cw' if clockwise else 'ccw'] += 1
        episode_idx += 1

    print(f"[INFO] done: {episode_idx} episodes, {total_steps} steps total (target was {target_steps})")
    for shape, count in per_shape_count.items():
        print(f"  {shape}: {count} episodes")
    print(f"  direction: {per_dir_count['ccw']} CCW / {per_dir_count['cw']} CW")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch-collect shape_dataset.py episodes up to a step budget')
    parser.add_argument('--target_steps',       default=DEFAULT_TARGET_STEPS, type=int, help='Stop launching new episodes once the total steps reach this; an in-progress episode always finishes (default: 200000)', metavar='')
    parser.add_argument('--shapes',             default=DEFAULT_SHAPES, type=str, nargs='+', help='Shapes to round-robin through (default: all)', metavar='', choices=sd.SHAPE_SIDES.keys())
    parser.add_argument('--seed_start',         default=DEFAULT_SEED_START, type=int, help='Seed of the first episode; increments by 1 each episode (default: 0)', metavar='')
    parser.add_argument('--drone',              default=sd.DEFAULT_DRONE,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--physics',            default=sd.DEFAULT_PHYSICS,   type=Physics,       help='Physics updates (default: PYB_DRAG)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=False,             type=str2bool,         help='Whether to use PyBullet GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=sd.DEFAULT_OBSTACLES, type=str2bool,      help='Whether to add obstacles (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=sd.DEFAULT_SIMULATION_FREQ_HZ, type=int,  help='Simulation frequency in Hz (default: 1000)', metavar='')
    parser.add_argument('--control_freq_hz',    default=sd.DEFAULT_CONTROL_FREQ_HZ,    type=int,  help='Control frequency in Hz (default: 100)', metavar='')
    parser.add_argument('--radius',             default=sd.DEFAULT_RADIUS,    type=float,         help='Base circumradius in meters, before jitter (default: 2.2)', metavar='')
    parser.add_argument('--side_jitter',        default=sd.DEFAULT_SIDE_JITTER, type=float,       help='Fractional per-side length variation (default: 0.3)', metavar='')
    parser.add_argument('--tilt_max_deg',       default=sd.DEFAULT_TILT_MAX_DEG, type=float,      help='Max random tilt in degrees (default: 30)', metavar='')
    parser.add_argument('--workspace_size',     default=sd.DEFAULT_WORKSPACE_SIZE, type=float,    help='Cubic workspace side length in meters (default: 5.0)', metavar='')
    parser.add_argument('--path_resolution',    default=sd.DEFAULT_PATH_RESOLUTION, type=int,     help='Waypoints sampled along the path (default: 3000)', metavar='')
    parser.add_argument('--n_laps',             default=sd.DEFAULT_N_LAPS, type=float,            help='Laps per episode when --duration_sec is not set (default: 3)', metavar='')
    parser.add_argument('--duration_sec',       default=sd.DEFAULT_DURATION_SEC, type=float,      help='Seconds per episode (default: auto = n_laps * time-optimal lap time)', metavar='')
    parser.add_argument('--max_speed_min',      default=sd.DEFAULT_MAX_SPEED_MIN, type=float,     help='Lower bound (m/s) episode max speed is drawn from (default: 2.0)', metavar='')
    parser.add_argument('--max_speed_max',      default=sd.DEFAULT_MAX_SPEED_MAX, type=float,     help='Upper bound (m/s) episode max speed is drawn from (default: 2.0)', metavar='')
    parser.add_argument('--max_accel_min',      default=sd.DEFAULT_MAX_ACCEL_MIN, type=float,     help='Lower bound (m/s^2) episode max acceleration is drawn from (default: 2.0)', metavar='')
    parser.add_argument('--max_accel_max',      default=sd.DEFAULT_MAX_ACCEL_MAX, type=float,     help='Upper bound (m/s^2) episode max acceleration is drawn from (default: 2.0)', metavar='')
    parser.add_argument('--speed_margin',       default=sd.DEFAULT_SPEED_MARGIN, type=float,      help='Fraction of max_speed/max_accel the planned profile targets (default: 0.7)', metavar='')
    parser.add_argument('--lookahead_dist',     default=sd.DEFAULT_LOOKAHEAD_DIST, type=float,    help='Pure-pursuit lookahead distance in meters (default: 0.3)', metavar='')
    parser.add_argument('--output_folder',      default=sd.DEFAULT_OUTPUT_FOLDER, type=str,       help='Folder where to save datasets (default: "results")', metavar='')
    parser.add_argument('--att_d_gain_scale',   default=sd.DEFAULT_ATT_D_GAIN_SCALE, type=float,  help='Scales DSLPIDControl attitude D-gain for every episode (default: 1.0 = unmodified)', metavar='')
    parser.add_argument('--perturb_prob',       default=sd.DEFAULT_PERTURB_PROB, type=float,      help='Probability each episode gets perturb_count random mid-episode position displacements, to record off-path recovery data (default: 0.0 = off)', metavar='')
    parser.add_argument('--perturb_magnitude',  default=sd.DEFAULT_PERTURB_MAGNITUDE, type=float, help='Max meters of each position displacement when perturbation fires (default: 1.5)', metavar='')
    parser.add_argument('--perturb_count',      default=sd.DEFAULT_PERTURB_COUNT, type=int,       help='Number of independent kicks within an episode that gets perturbed (default: 1)', metavar='')
    parser.add_argument('--obs_pos_noise_std',  default=sd.DEFAULT_OBS_POS_NOISE_STD, type=float, help='Meters of Gaussian noise added to the logged tx-x/ty-y/tz-z columns only (not reward, not control); default: 0.0 = off', metavar='')
    parser.add_argument('--direction',          default='both', type=str, choices=['both', 'ccw', 'cw'], help="Traversal direction: 'both' alternates each shape half CCW / half CW (default), 'ccw'/'cw' force one", metavar='')
    ARGS = parser.parse_args()

    collect(**vars(ARGS))
