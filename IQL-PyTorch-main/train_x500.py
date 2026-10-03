"""IQL for the explicit X500 CTBR transitions in datasets/data_track.csv.gz.

Offline validation measures action fit, not closed-loop flight performance.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import Log, sample_batch, set_seed, torchify


def read_table(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f:
        return np.genfromtxt(f, delimiter=',', names=True, dtype=np.float32)


def load_data(path, seed, extra=()):
    """Validation flights are drawn from `path` only; `extra` CSVs (e.g. DAgger) always go to train."""
    table = read_table(path)
    base_flights = np.unique(table['flight']).astype(int)
    if extra:
        tables = [table]+[read_table(p) for p in extra]
        names = [n for n in table.dtype.names if all(n in t.dtype.names for t in tables)]
        dtype = [(n, np.float32) for n in names]
        table = np.concatenate([t[names].astype(dtype) for t in tables])
    def columns(prefix, count):
        return np.column_stack([table[f'{prefix}_{i}'] for i in range(count)])
    raw = dict(observations=columns('obs', 25), actions=columns('act', 4),
               next_observations=columns('next_obs', 25), rewards=table['reward'],
               terminals=table['terminal'])
    if not all(np.isfinite(v).all() for v in raw.values()):
        raise ValueError('Non-finite transitions')
    if np.abs(raw['actions']).max() > 1.00001:
        raise ValueError('CTBR actions must be in [-1, 1]')
    for name in ('terminal', 'timeout'):
        if not np.isin(table[name], [0, 1]).all():
            raise ValueError(f'{name} must be binary')
    flights = np.unique(table['flight']).astype(int)
    if len(base_flights) < 2:
        raise ValueError('Need at least two flights for held-out validation')
    val_flights = np.random.default_rng(seed).permutation(base_flights)[:max(1, round(len(base_flights)*0.2))]
    val_mask = np.isin(table['flight'], val_flights)
    train_mask = ~val_mask
    mean = raw['observations'][train_mask].mean(0, dtype=np.float64).astype(np.float32)
    std = raw['observations'][train_mask].std(0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1., std)
    # Keep provided next states, including at timeouts; only crashes mask bootstrap.
    for name in ('observations', 'next_observations'):
        raw[name] = (raw[name] - mean) / std
    train = {k: torchify(v[train_mask]) for k, v in raw.items()}
    val = {k: torchify(v[val_mask]) for k, v in raw.items()}
    metadata = dict(train_flights=sorted(set(flights.tolist())-set(val_flights.tolist())),
                    validation_flights=sorted(val_flights.tolist()),
                    train_rows=int(train_mask.sum()), validation_rows=int(val_mask.sum()),
                    terminals=int(table['terminal'].sum()), timeouts=int(table['timeout'].sum()),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    extra={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in extra})
    return train, val, mean, std, metadata


@torch.no_grad()
def validate(policy, data):
    errors = []
    for obs, actions in zip(data['observations'].split(2048), data['actions'].split(2048)):
        errors.append((policy(obs).mean-actions).square())
    mse = torch.cat(errors).mean(0).cpu().numpy()
    return {'action_mse': float(mse.mean()), 'action_rmse_per_dim': np.sqrt(mse).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv-file', type=Path, required=True)
    parser.add_argument('--log-dir', type=Path, default=Path('runs_x500'))
    parser.add_argument('--n-steps', type=int, default=300000)
    parser.add_argument('--eval-period', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split-seed', type=int, default=0, help='flight split; keep fixed across seeds')
    parser.add_argument('--tau', type=float, default=.85)
    parser.add_argument('--extra-csv', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    torch.set_num_threads(1)
    set_seed(args.seed)
    train, val, mean, std, metadata = load_data(args.csv_file, args.split_seed, args.extra_csv)
    obs_dim = train['observations'].shape[1]
    config = dict(vars(args), dataset_format='x500_ctbr', obs_dim=obs_dim, act_dim=4,
                  hidden_dim=256, n_hidden=2, beta=3., discount=.99,
                  smoothness_coef=.05, max_action=1., learning_rate=3e-4,
                  reward_clip_min=None, batch_size=256)
    log = Log(args.log_dir/'data_track', config)
    log(f'Log dir: {log.dir}; device={train["observations"].device}')
    log(str(metadata))
    (log.dir/'dataset_split.json').write_text(json.dumps(metadata, indent=2))
    np.savez(log.dir/'obs_normalization.npz', mean=mean, std=std, action_bound=1.)
    policy = GaussianPolicy(obs_dim, 4, hidden_dim=256, max_action=1.)
    iql = ImplicitQLearning(TwinQ(obs_dim, 4), ValueFunction(obs_dim), policy,
                           lambda p: torch.optim.Adam(p, lr=3e-4),
                           max_steps=args.n_steps, tau=args.tau, beta=3., smoothness_coef=.05)
    initial = validate(policy, val)
    baseline = float((val['actions']-train['actions'].mean(0)).square().mean())
    for step in range(1, args.n_steps+1):
        losses = iql.update(**sample_batch(train, 256))
        if not all(np.isfinite(v) for v in losses.values()):
            raise RuntimeError(f'Non-finite loss at step {step}: {losses}')
        if step % args.eval_period == 0 or step == args.n_steps:
            metrics = validate(policy, val)
            log.row(dict(step=step, **losses, val_action_mse=metrics['action_mse']))
    torch.save(iql.state_dict(), log.dir/'final.pt')
    report = dict(initial_validation=initial, final_validation=metrics,
                  train_mean_action_baseline_mse=baseline,
                  train_action_fit=validate(policy, train),
                  online_evaluation='Not run: compatible X500 environment absent')
    (log.dir/'evaluation.json').write_text(json.dumps(report, indent=2))
    log(str(report))
    log.close()


if __name__ == '__main__':
    main()
