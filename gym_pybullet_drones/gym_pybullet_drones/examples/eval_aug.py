"""Table-1-style evaluation for the augmentation experiment.

For a trained policy RUN dir, rolls out seeds 500-509 x {ccw,cw} on each shape, and per rollout
records BOTH net laps (closest-path-index progress, like progress_metric.py) and mean tracking
error (|pos_err|, like evaluate_trained_policy.py). Aggregates per shape: net laps mean+-std (min,
traverse-count where laps>=2.0) and distance mean+-std (max) -- exactly the Table 1 columns.

    conda activate drones
    KMP_DUPLICATE_LIB_OK=TRUE python eval_aug.py --run-dir <RUN> [--shapes triangle square pentagon circle]
"""
import os, sys, json, csv, argparse, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shape_dataset as sd
from evaluate_trained_policy import load_policy, load_normalization, make_policy_fn

N_IDX = 3000  # path discretization used by progress_metric


def net_laps(rec):
    rec = np.array(rec)
    if len(rec) < 2:
        return 0.0, 0.0
    cov = len(np.unique(rec)) / N_IDX
    d = np.diff(rec)
    d = np.where(d < -N_IDX / 2, d + N_IDX, np.where(d > N_IDX / 2, d - N_IDX, d))
    return cov, abs(d.sum() / N_IDX)      # magnitude (CW traverses path indices backward)


def rollout(shape, seed, cw, policy, mean, std, include_la, out):
    fn = make_policy_fn(policy, mean, std, slew_max_accel=2.0, include_lookahead=include_la)
    rec = []
    orig = sd.PurePursuitTracker.step
    def patched(self, cur_pos, _r=rec, _o=orig):
        tv, ci = _o(self, cur_pos); _r.append(ci); return tv, ci
    sd.PurePursuitTracker.step = patched
    try:
        sd.run(shape=shape, seed=seed, gui=False, policy_fn=fn, att_d_gain_scale=0.3,
               output_folder=out, clockwise=cw)
    finally:
        sd.PurePursuitTracker.step = orig
    cov, laps = net_laps(rec)
    csvf = max(glob.glob(os.path.join(out, 'shape_dataset', '*.csv')), key=os.path.getmtime)
    with open(csvf) as f:
        rows = list(csv.DictReader(f))
    pe = np.array([[float(r['tx-x']), float(r['ty-y']), float(r['tz-z'])] for r in rows])
    os.remove(csvf)                       # keep the folder from growing / stale-mtime races
    pen = np.linalg.norm(pe, axis=1)      # per-step |pos_err| over the whole rollout
    return laps, cov, float(pen.mean()), pen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--shapes', nargs='+', default=['triangle', 'square', 'pentagon', 'circle'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(500, 510)))
    ap.add_argument('--label', default=None)
    A = ap.parse_args()
    cfg = json.load(open(os.path.join(A.run_dir, 'config.json')))
    include_la = bool(cfg.get('include_lookahead'))
    mean, std, ab = load_normalization(A.run_dir)
    policy = load_policy(A.run_dir, max_action=ab)
    out = f'/tmp/evalaug_{os.getpid()}'
    os.makedirs(os.path.join(out, 'shape_dataset'), exist_ok=True)

    print(f"# eval {A.label or os.path.basename(A.run_dir.rstrip('/'))}  seeds {A.seeds[0]}-{A.seeds[-1]} x both dirs")
    print(f"{'shape':10} {'laps mean±std':>16} {'min':>5} {'trav':>6} {'dist mean±std':>16} "
          f"{'p90':>6} {'p99':>6} {'max':>7}")
    summary = {}
    for shape in A.shapes:
        laps, dists, pe_pool = [], [], []
        for seed in A.seeds:
            for cw in (False, True):
                l, cov, derr, pen = rollout(shape, seed, cw, policy, mean, std, include_la, out)
                laps.append(l); dists.append(derr); pe_pool.append(pen)
        laps, dists = np.array(laps), np.array(dists)
        pe_all = np.concatenate(pe_pool)                          # per-step |pos_err| pooled over all rollouts
        p90, p99, pmax = (float(np.percentile(pe_all, 90)), float(np.percentile(pe_all, 99)), float(pe_all.max()))
        trav = int((laps >= 2.0).sum())
        print(f"{shape:10} {laps.mean():>7.2f}±{laps.std():<7.2f} {laps.min():>5.2f} {trav:>3}/{len(laps)} "
              f"{dists.mean():>7.3f}±{dists.std():<7.3f} {p90:>6.3f} {p99:>6.3f} {pmax:>7.3f}")
        summary[shape] = dict(laps_mean=laps.mean(), laps_std=laps.std(), laps_min=laps.min(),
                              trav=trav, n=len(laps), dist_mean=dists.mean(), dist_std=dists.std(),
                              dist_p90=p90, dist_p99=p99, dist_max=pmax)
    total_trav = sum(s['trav'] for s in summary.values())
    total_n = sum(s['n'] for s in summary.values())
    print(f"# TOTAL traverse: {total_trav}/{total_n}")
    json.dump(summary, open(os.path.join(A.run_dir, 'eval_aug_summary.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
