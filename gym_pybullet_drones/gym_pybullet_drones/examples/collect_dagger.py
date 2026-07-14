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
    parser.add_argument('--output_folder', default='dagger_data')
    ARGS = parser.parse_args()

    obs_mean, obs_std, action_bound = load_normalization(ARGS.run_dir)
    policy = load_policy(ARGS.run_dir, max_action=action_bound)

    n_eps = 0
    for seed in range(ARGS.seed_start, ARGS.seed_start + ARGS.n_seeds):
        for shape in ARGS.shapes:
            #### Fresh policy_fn per episode so the slew-limiter's internal prev-action state
            #### resets at each episode start (it's a stateful closure).
            policy_fn = make_policy_fn(policy, obs_mean, obs_std, slew_max_accel=ARGS.slew_max_accel)
            sd.run(shape=shape, seed=seed, gui=False, policy_fn=policy_fn, dagger_relabel=True,
                   att_d_gain_scale=ARGS.att_d_gain_scale, output_folder=ARGS.output_folder)
            n_eps += 1
    print(f"[INFO] DAgger collection done: {n_eps} episodes -> {ARGS.output_folder}/shape_dataset/")


if __name__ == '__main__':
    main()
