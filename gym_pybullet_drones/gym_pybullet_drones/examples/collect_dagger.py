"""DAgger data collection: drive the drone with a TRAINED policy (so it visits the
off-path states the policy actually reaches), but log pure-pursuit's expert answer at
each of those states as the action label. Adding this to the training set teaches the
policy the correct recovery on its own state distribution -- the standard fix for the
closed-loop covariate-shift / BC-approximation-accumulation failure diagnosed for this
project (see project_drone_offline_rl memory).

Run in the `drones` env (needs pybullet + torch):
    KMP_DUPLICATE_LIB_OK=TRUE python collect_dagger.py --run-dir <RUN> \
        --shapes triangle --seed-start 0 --n-seeds 20 --output_folder dagger_iter1
"""
import argparse
import json
import os

import shape_dataset as sd
from evaluate_trained_policy import load_policy, load_normalization, make_policy_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, help='trained policy run dir (final.pt + config + npz)')
    parser.add_argument('--shapes', nargs='+', default=['triangle'])
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--n-seeds', type=int, default=20, help='seeds seed_start .. seed_start+n_seeds-1')
    parser.add_argument('--slew-max-accel', type=float, default=2.0,
                         help='slew cap on the policy driving the drone (match rollout); keeps the '
                              'on-policy states realistic instead of blown-up')
    parser.add_argument('--att_d_gain_scale', type=float, default=0.3)
    parser.add_argument('--perturb_prob', type=float, default=0.0,
                         help='probability an episode gets perturbation kicks DURING the DAgger '
                              'rollout -- makes the policy visit more off-path (incl. corner) '
                              'recovery states, which pure-pursuit then labels; 0 = off')
    parser.add_argument('--perturb_count', type=int, default=1)
    parser.add_argument('--perturb_magnitude', type=float, default=1.5)
    parser.add_argument('--output_folder', default='dagger_data')
    parser.add_argument('--direction', default='both', choices=['both', 'ccw', 'cw'],
                         help="Traversal direction for the DAgger rollouts: 'both' alternates by "
                              "seed parity (half CCW / half CW, default), 'ccw'/'cw' force one. "
                              "Match the initial dataset so the policy sees recovery states in both "
                              "directions it was trained on.")
    ARGS = parser.parse_args()

    with open(os.path.join(ARGS.run_dir, 'config.json')) as f:
        cfg = json.load(f)
    obs_mean, obs_std, action_bound = load_normalization(ARGS.run_dir)
    policy = load_policy(ARGS.run_dir, max_action=action_bound)

    n_eps = 0
    for seed in range(ARGS.seed_start, ARGS.seed_start + ARGS.n_seeds):
        #### 'both' toggles direction by seed parity -> each shape (inner loop) gets half its
        #### seeds CCW and half CW across a contiguous seed range, matching the initial dataset.
        if ARGS.direction == 'both':
            clockwise = (seed % 2 == 1)
        else:
            clockwise = (ARGS.direction == 'cw')
        for shape in ARGS.shapes:
            #### Fresh policy_fn per episode so the slew-limiter's internal prev-action state
            #### resets at each episode start (it's a stateful closure).
            policy_fn = make_policy_fn(policy, obs_mean, obs_std, slew_max_accel=ARGS.slew_max_accel,
                                        include_prev_action=bool(cfg.get('include_prev_action')),
                                        include_lookahead=bool(cfg.get('include_lookahead')))
            sd.run(shape=shape, seed=seed, gui=False, policy_fn=policy_fn, dagger_relabel=True,
                   att_d_gain_scale=ARGS.att_d_gain_scale, output_folder=ARGS.output_folder,
                   perturb_prob=ARGS.perturb_prob, perturb_count=ARGS.perturb_count,
                   perturb_magnitude=ARGS.perturb_magnitude, clockwise=clockwise)
            n_eps += 1
    print(f"[INFO] DAgger collection done: {n_eps} episodes -> {ARGS.output_folder}/shape_dataset/")


if __name__ == '__main__':
    main()
