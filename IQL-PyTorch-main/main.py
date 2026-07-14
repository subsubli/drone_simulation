from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, DeterministicPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import set_seed, Log, sample_batch, torchify
from src.drone_dataset import load_drone_dataset


def get_env_and_dataset(log, env_name, max_episode_steps):
    import gym
    import d4rl
    from src.util import return_range

    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        log(f'Dataset returns have range [{min_ret}, {max_ret}]')
        dataset['rewards'] /= (max_ret - min_ret)
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.

    for k, v in dataset.items():
        dataset[k] = torchify(v)

    return env, dataset


def main(args):
    torch.set_num_threads(1)
    log_name = args.csv_file.stem if args.csv_file else args.env_name
    log = Log(Path(args.log_dir)/log_name, vars(args))
    log(f'Log dir: {log.dir}')

    #### Drone CSV mode: no live gym env, so periodic env-rollout evaluation is skipped --
    #### this only trains against the fixed offline dataset (use a held-out CSV to check
    #### generalization instead of a live-rollout return, if needed).
    #### --max-action lets the Tanh action bound be set manually (e.g. to the dataset's
    #### intended max_speed rather than whatever empirical per-component max happened to be
    #### flown); None (default) falls back to the dataset-derived bound in csv-file mode, or
    #### stays unbounded in D4RL mode (original behavior).
    max_action = args.max_action
    if args.csv_file:
        env = None
        raw_dataset, obs_norm = load_drone_dataset(args.csv_file, reward_clip_min=args.reward_clip_min,
                                                    pos_err_scale=args.pos_err_scale)
        dataset = {k: torchify(v) for k, v in raw_dataset.items()}
        offpath_idx = torch.from_numpy(np.where(obs_norm['offpath_mask'])[0]).to(
            dataset['observations'].device)
        log(f'off-path rows (|pos_err|>0.2): {len(offpath_idx)} / {len(dataset["observations"])} '
            f'({100*len(offpath_idx)/len(dataset["observations"]):.2f}%)')
        if max_action is None:
            max_action = obs_norm['action_bound']
        #### Saved alongside final.pt -- evaluation/deployment must normalize observations
        #### and bound actions with these exact stats (fit on this training CSV, or the CLI
        #### override actually used), not refit/recompute their own.
        np.savez(log.dir/'obs_normalization.npz', mean=obs_norm['obs_mean'], std=obs_norm['obs_std'],
                  action_bound=max_action)
    else:
        env, dataset = get_env_and_dataset(log, args.env_name, args.max_episode_steps)
        offpath_idx = None

    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]   # this assume continuous actions
    set_seed(args.seed, env=env)

    if args.deterministic_policy:
        policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
                                      max_action=max_action)
    else:
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
                                 max_action=max_action)

    def eval_policy():
        import d4rl
        from src.util import evaluate_policy
        eval_returns = np.array([evaluate_policy(env, policy, args.max_episode_steps) \
                                 for _ in range(args.n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
        log.row({
            'return mean': eval_returns.mean(),
            'return std': eval_returns.std(),
            'normalized return mean': normalized_returns.mean(),
            'normalized return std': normalized_returns.std(),
        })

    iql = ImplicitQLearning(
        qf=TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        vf=ValueFunction(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount,
        smoothness_coef=args.smoothness_coef,
    )

    #### Oversampling: force a fixed fraction of each batch to come from off-path recovery
    #### rows (else they're ~5% of data and their behavior-cloning signal is swamped). This
    #### is a diagnostic knob to tell a data-QUANTITY problem (this fixes it) from a data-
    #### DIVERSITY problem (only 2 kicks -> few distinct recovery directions; this won't).
    oversample = offpath_idx is not None and args.oversample_offpath_frac > 0 and len(offpath_idx) > 0
    def sample_batch_oversampled():
        n = len(dataset['observations'])
        device = dataset['observations'].device
        n_off = int(args.batch_size * args.oversample_offpath_frac)
        idx_uni = torch.randint(n, (args.batch_size - n_off,), device=device)
        idx_off = offpath_idx[torch.randint(len(offpath_idx), (n_off,), device=device)]
        idx = torch.cat([idx_uni, idx_off])
        return {k: v[idx] for k, v in dataset.items()}

    for step in trange(args.n_steps):
        batch = sample_batch_oversampled() if oversample else sample_batch(dataset, args.batch_size)
        losses = iql.update(**batch)
        if (step+1) % args.eval_period == 0:
            if env is not None:
                eval_policy()
            else:
                log.row({'step': step+1, **losses})

    torch.save(iql.state_dict(), log.dir/'final.pt')
    log.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env-name', default=None, help='D4RL env name (mutually exclusive with --csv-file)')
    parser.add_argument('--csv-file', type=Path, default=None, help='shape_dataset.py CSV (per-episode or merged.csv) to train on instead of a D4RL env')
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=10**6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.85)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--max-action', type=float, default=None,
                         help='manually overrides the policy\'s Tanh action bound (src/policy.py); '
                              'default None = dataset-derived (csv-file mode) or unbounded (D4RL mode)')
    parser.add_argument('--reward-clip-min', type=float, default=None,
                         help='floors csv-file-mode rewards at this value (reward is always <= 0); '
                              'bounds the discount-bootstrapped TD target scale when perturbation-'
                              'recovery rows have much larger |reward| than normal tracking rows; '
                              'default None = off')
    parser.add_argument('--pos-err-scale', type=float, default=None,
                         help='overrides the tx-x/ty-y/tz-z normalization divisor with this fixed '
                              'meters value instead of their empirical std, so real position error '
                              'registers as a larger-magnitude input relative to velocity/quaternion; '
                              'default None = use empirical std (original behavior)')
    parser.add_argument('--oversample-offpath-frac', type=float, default=0.0,
                         help='csv-file mode: fraction of each training batch drawn from off-path '
                              'recovery rows (|pos_err|>0.2m) instead of uniformly; 0 = off')
    parser.add_argument('--smoothness-coef', type=float, default=0.05,
                         help='penalizes the policy mean-action jump between consecutive states '
                              '(observations -> next_observations); 0 = off')
    parser.add_argument('--deterministic-policy', action='store_true')
    parser.add_argument('--eval-period', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    ARGS = parser.parse_args()
    if (ARGS.env_name is None) == (ARGS.csv_file is None):
        parser.error('Pass exactly one of --env-name or --csv-file')
    main(ARGS)