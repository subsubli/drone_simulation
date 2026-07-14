"""Rolls a trained IQL-PyTorch-main policy through shape_dataset.py's PyBullet simulation
and compares tracking error / action smoothness against the pure-pursuit expert that
generated the training data, on held-out seeds (not used during training).

Needs both pybullet and torch importable in the same process, which triggers an OpenMP
runtime conflict in the `drones` conda env unless worked around. Run as:

    $ KMP_DUPLICATE_LIB_OK=TRUE python evaluate_trained_policy.py \\
        --run-dir /path/to/IQL-PyTorch-main/runs/merged/<timestamp_dir> \\
        --seed 500
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

import shape_dataset as sd

IQL_PYTORCH_DIR = '/Users/hanjakp/drone_simulation/IQL-PyTorch-main'
sys.path.insert(0, IQL_PYTORCH_DIR)
from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction

#### Must match drone_dataset.py's STATE_COLS/ACTION_COLS ordering exactly.
STATE_DIM = 13
ACTION_DIM = 3


def load_policy(run_dir, max_action):
    with open(os.path.join(run_dir, 'config.json')) as f:
        cfg = json.load(f)
    hidden_dim, n_hidden = cfg['hidden_dim'], cfg['n_hidden']
    #### state dim grows by 3 per enabled extra feature (lookahead lx/ly/lz, prev-action).
    state_dim = STATE_DIM + (3 if cfg.get('include_lookahead') else 0) \
                          + (ACTION_DIM if cfg.get('include_prev_action') else 0)
    #### max_action must match what main.py used at training time (src/policy.py's bound
    #### doesn't change the state_dict's shapes, so a mismatch would load silently wrong).
    if cfg['deterministic_policy']:
        policy = DeterministicPolicy(state_dim, ACTION_DIM, hidden_dim=hidden_dim, n_hidden=n_hidden,
                                      max_action=max_action)
    else:
        policy = GaussianPolicy(state_dim, ACTION_DIM, hidden_dim=hidden_dim, n_hidden=n_hidden,
                                 max_action=max_action)
    #### Reconstructs the full ImplicitQLearning module (qf/vf/policy) purely so the saved
    #### state_dict (which contains all three) loads cleanly -- only `.policy` is used below.
    iql = ImplicitQLearning(
        qf=TwinQ(state_dim, ACTION_DIM, hidden_dim=hidden_dim, n_hidden=n_hidden),
        vf=ValueFunction(state_dim, hidden_dim=hidden_dim, n_hidden=n_hidden),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-4),
        max_steps=1,
        tau=cfg['tau'], beta=cfg['beta'], discount=cfg['discount'], alpha=cfg['alpha'],
        smoothness_coef=cfg.get('smoothness_coef', 0.0),
    )
    iql.load_state_dict(torch.load(os.path.join(run_dir, 'final.pt'), map_location='cpu'))
    iql.eval()
    return iql.policy


def load_normalization(run_dir):
    npz = np.load(os.path.join(run_dir, 'obs_normalization.npz'))
    return npz['mean'].astype(np.float32), npz['std'].astype(np.float32), float(npz['action_bound'])


def make_policy_fn(policy, obs_mean, obs_std, slew_max_accel=None, control_freq_hz=100,
                    include_prev_action=False, include_lookahead=False):
    #### The training data's target_vel came from PurePursuitTracker, which slew-rate-limits
    #### its command (max |delta-v| per step = max_accel / control_freq). The MLP policy
    #### predicts each step's target_vel independently, so nothing enforces that continuity
    #### at rollout -- and a raw MLP command that jumps between steps is not physically
    #### trackable by the low-level PID, so the drone overshoots and diverges (diagnosed:
    #### commanded ~0.5 m/s but achieved ~2-6 m/s). Re-applying the SAME slew-rate cap to the
    #### policy's output at rollout restores the continuity the data had. None = off (raw).
    max_delta_v = None if slew_max_accel is None else slew_max_accel / control_freq_hz
    #### prev_raw = the policy's previous RAW output, fed back as the prev-action state feature
    #### (matches the training label, which is the raw pure-pursuit target_vel). prev_slew =
    #### the previous slew-limited command actually sent to the drone, for the slew continuity.
    prev_raw = np.zeros(3)
    prev_slew = np.zeros(3)

    def policy_fn(pos_err, state, lookahead=None):
        nonlocal prev_raw, prev_slew
        obs = np.concatenate([pos_err, state[3:7], state[10:13], state[13:16]]).astype(np.float32)
        #### Must match drone_dataset's concat order: lookahead first, then prev_action.
        if include_lookahead:
            obs = np.concatenate([obs, np.asarray(lookahead, dtype=np.float32)]).astype(np.float32)
        if include_prev_action:
            obs = np.concatenate([obs, prev_raw]).astype(np.float32)
        obs = (obs - obs_mean) / obs_std
        with torch.no_grad():
            raw = policy.act(torch.from_numpy(obs), deterministic=True).numpy()
        prev_raw = raw.copy()
        action = raw
        if max_delta_v is not None:
            delta = raw - prev_slew
            dmag = np.linalg.norm(delta)
            if dmag > max_delta_v:
                action = prev_slew + delta * (max_delta_v / dmag)
            prev_slew = action
        return action
    return policy_fn


def latest_csv(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    return max(files, key=os.path.getmtime)


def episode_metrics(csv_path):
    import csv as csv_module
    with open(csv_path, newline='') as f:
        rows = list(csv_module.DictReader(f))
    pos_err = np.array([[float(r['tx-x']), float(r['ty-y']), float(r['tz-z'])] for r in rows])
    actions = np.array([[float(r['ax']), float(r['ay']), float(r['az'])] for r in rows])
    tracking_error = np.linalg.norm(pos_err, axis=1)
    action_deltas = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    return {
        'mean_tracking_error': float(tracking_error.mean()),
        'max_tracking_error': float(tracking_error.max()),
        'mean_action_smoothness': float(action_deltas.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True,
                         help='IQL-PyTorch-main/runs/.../<timestamp> dir with final.pt + config.json')
    parser.add_argument('--shapes', nargs='+', default=list(sd.SHAPE_SIDES.keys()))
    parser.add_argument('--seed', type=int, default=500,
                         help='held-out seed not used during training')
    parser.add_argument('--output_folder', default='eval_rollout')
    parser.add_argument('--att_d_gain_scale', type=float, default=0.3)
    parser.add_argument('--slew_max_accel', type=float, default=None,
                         help='m/s^2 slew-rate cap applied to the policy output at rollout, matching '
                              'the training data collection (default None = off / raw MLP output)')
    ARGS = parser.parse_args()

    with open(os.path.join(ARGS.run_dir, 'config.json')) as f:
        cfg = json.load(f)
    obs_mean, obs_std, action_bound = load_normalization(ARGS.run_dir)
    policy = load_policy(ARGS.run_dir, max_action=action_bound)

    results = []
    for shape in ARGS.shapes:
        print(f"[INFO] === {shape} (seed={ARGS.seed}) ===")
        #### Fresh policy_fn PER SHAPE -- make_policy_fn holds slew/prev state in its closure,
        #### so reusing one across shapes leaks the previous episode's ending state into the
        #### next shape's start and corrupts it (was silently inflating square/circle error).
        policy_fn = make_policy_fn(policy, obs_mean, obs_std, slew_max_accel=ARGS.slew_max_accel,
                                    include_prev_action=bool(cfg.get('include_prev_action')),
                                    include_lookahead=bool(cfg.get('include_lookahead')))
        sd.run(shape=shape, seed=ARGS.seed, gui=False,
               output_folder=os.path.join(ARGS.output_folder, 'expert'),
               att_d_gain_scale=ARGS.att_d_gain_scale)
        sd.run(shape=shape, seed=ARGS.seed, gui=False,
               output_folder=os.path.join(ARGS.output_folder, 'policy'),
               att_d_gain_scale=ARGS.att_d_gain_scale, policy_fn=policy_fn)

        expert_m = episode_metrics(latest_csv(os.path.join(ARGS.output_folder, 'expert', 'shape_dataset')))
        policy_m = episode_metrics(latest_csv(os.path.join(ARGS.output_folder, 'policy', 'shape_dataset')))
        results.append((shape, expert_m, policy_m))
        print(f"  expert: {expert_m}")
        print(f"  policy: {policy_m}")

    print("\n=== Summary (mean tracking error / mean action smoothness, meters) ===")
    print(f"{'shape':<10} {'expert_err':>12} {'policy_err':>12} {'expert_smooth':>14} {'policy_smooth':>14}")
    for shape, e, p in results:
        print(f"{shape:<10} {e['mean_tracking_error']:>12.4f} {p['mean_tracking_error']:>12.4f} "
              f"{e['mean_action_smoothness']:>14.4f} {p['mean_action_smoothness']:>14.4f}")


if __name__ == '__main__':
    main()
