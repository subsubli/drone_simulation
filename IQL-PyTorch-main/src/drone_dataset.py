"""Loads a shape_dataset.py CSV (or merged.csv) into the same dataset dict format
`main.py` expects from D4RL (`observations, actions, rewards, terminals, next_observations`).

CSV columns expected (see gym_pybullet_drones/examples/shape_dataset.py's docstring):
    [episode_id,] step, tx-x, ty-y, tz-z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz, ax, ay, az, reward, done

`episode_id` is optional -- present in merge_shape_dataset.py's merged.csv output, absent in
a single per-episode CSV (in which case the whole file is treated as one episode).
"""
import csv

import numpy as np

STATE_COLS = ['tx-x', 'ty-y', 'tz-z', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']
ACTION_COLS = ['ax', 'ay', 'az']
LOOKAHEAD_COLS = ['lx', 'ly', 'lz']


def load_drone_dataset(csv_file, reward_clip_min=None, pos_err_scale=None, include_prev_action=False,
                       include_lookahead=False):
    """`reward_clip_min`, if given, floors `rewards` at that value (reward is always <= 0,
    a negative distance, so there's no meaningful upper clip). Perturbation-recovery rows
    can have reward down around -3 vs. the -0.01 to -0.05 typical of normal tracking; that
    100x+ range destabilizes the discount=0.99 TD bootstrap (observed as V/Q loss not
    converging -- see project_drone_offline_rl memory). Clipping trades away fine-grained
    "how far off" signal beyond the clip point for a bounded, more stable target scale.

    `pos_err_scale`, if given, overrides the tx-x/ty-y/tz-z channels' normalization divisor
    with this fixed value (meters) instead of their empirical std. A rollout diagnostic
    found the trained policy's action was only weakly, non-monotonically sensitive to
    pos_err -- it leaned on velocity/quaternion instead, since those explain most of the
    variance in the (usually near-zero-error) training data. Dividing by a small fixed
    scale (e.g. 0.1) makes any real position error register as a much larger-magnitude
    input relative to the other channels, regardless of how narrow its empirical spread
    happens to be, forcing the network to give it more weight.

    `include_prev_action`, if True, appends the PREVIOUS step's action (target_vel) to the
    observation, growing it from 13 to 16 dims. Motivation: the plain 13-dim state has no
    path-progress / tangent-direction cue -- pos_err is only the perpendicular offset to the
    nearest path point -- so once on-path (pos_err~0) the policy has no signal for which way
    to travel and stalls (diagnosed via a progress metric: it covers only 6-27% of the path
    and completes ~0 laps). The pure-pursuit action IS the look-ahead/progress direction, but
    the current action can't go in the state (that's the label); the PREVIOUS action can
    (standard last-action-in-state trick, no leakage) and carries the recent travel direction
    so the policy can maintain heading. Fit from the CSV without recollecting; at deployment
    the policy's own previous output is fed back in (see evaluate_trained_policy.py). This is
    a proxy for a true look-ahead-point state feature (which would need recollection).
    """
    with open(csv_file, newline='') as f:
        rows = list(csv.DictReader(f))

    observations = np.array([[float(r[c]) for c in STATE_COLS] for r in rows], dtype=np.float32)
    actions = np.array([[float(r[c]) for c in ACTION_COLS] for r in rows], dtype=np.float32)
    rewards = np.array([float(r['reward']) for r in rows], dtype=np.float32)
    if include_lookahead:
        #### lx/ly/lz = vector from the drone to the pure-pursuit look-ahead point (points
        #### AHEAD along the path), the explicit progress/heading cue pos_err lacks. Appended
        #### to the observation. Needs a CSV recollected with these columns (new schema).
        lookahead = np.array([[float(r[c]) for c in LOOKAHEAD_COLS] for r in rows], dtype=np.float32)
        observations = np.concatenate([observations, lookahead], axis=1)
    if reward_clip_min is not None:
        rewards = np.maximum(rewards, reward_clip_min)
    terminals = np.array([r['done'] == 'True' for r in rows], dtype=np.float32)
    if 'episode_id' in rows[0]:
        episode_id = np.array([int(r['episode_id']) for r in rows])
    else:
        episode_id = np.zeros(len(rows), dtype=int)  # whole file == one episode

    if include_prev_action:
        #### prev_action[t] = action[t-1] within the same episode, 0 at each episode's first
        #### row. Appended to the observation so the policy sees its own recent heading.
        prev_action = np.zeros_like(actions)
        same_prev = episode_id[1:] == episode_id[:-1]
        prev_action[1:][same_prev] = actions[:-1][same_prev]
        observations = np.concatenate([observations, prev_action], axis=1)

    #### next_observations: shift by one row within each episode. The last row of each
    #### episode has no real next state -- left as a copy of its own observation, which is
    #### fine since `terminals` masks it out of the Q-learning bootstrap during training.
    next_observations = observations.copy()
    same_episode_next = episode_id[:-1] == episode_id[1:]
    next_observations[:-1][same_episode_next] = observations[1:][same_episode_next]

    #### The 13 state dims live on very different scales (meters vs. unit quaternion vs.
    #### m/s vs. rad/s), which starves an unnormalized MLP of useful gradient signal.
    #### Stats are fit on this training CSV only and must be saved and reused verbatim at
    #### inference time (see main.py / evaluate_trained_policy.py) -- never refit at eval.
    #### Computed on the RAW (pre-normalization) position-error columns, so the threshold
    #### stays in physical meters. Marks rows where the drone is meaningfully off-path --
    #### used by main.py to oversample the otherwise-rare (~5%) recovery transitions, whose
    #### behavior-cloning signal is otherwise swamped by the ~95% near-zero-error rows.
    offpath_mask = np.linalg.norm(observations[:, :3], axis=1) > 0.2

    obs_mean = observations.mean(axis=0)
    obs_std = observations.std(axis=0)
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std)
    if pos_err_scale is not None:
        obs_std[:3] = pos_err_scale
    observations = (observations - obs_mean) / obs_std
    next_observations = (next_observations - obs_mean) / obs_std

    #### Largest action magnitude actually flown -- used to bound the policy network's
    #### output (see src/policy.py's max_action) so it can never command something the
    #### expert data never demonstrated, regardless of how the state extrapolates.
    action_bound = float(np.abs(actions).max())

    return {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'terminals': terminals,
        'next_observations': next_observations,
    }, {'obs_mean': obs_mean, 'obs_std': obs_std, 'action_bound': action_bound,
        'offpath_mask': offpath_mask}
