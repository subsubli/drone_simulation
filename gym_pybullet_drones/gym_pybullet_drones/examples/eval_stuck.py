"""Diagnose WHY a policy's net laps are low: stuck (frozen at a corner) vs slow (advancing but
not looping enough) vs lost (diverged off the path). Records the closest-path-index trace and
tracking error per rollout, and reports coverage, net laps, how much the index still advances in
the LAST quarter of the episode (0 => frozen at end), and mean |pos_err| (large => off the path).

    conda activate drones
    KMP_DUPLICATE_LIB_OK=TRUE python eval_stuck.py --run-dir <RUN> --seeds 500 501 502
"""
import os, sys, json, csv, argparse, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shape_dataset as sd
from evaluate_trained_policy import load_policy, load_normalization, make_policy_fn

N_IDX = 3000


def wrapped(d):
    return np.where(d < -N_IDX / 2, d + N_IDX, np.where(d > N_IDX / 2, d - N_IDX, d))


def rollout(shape, seed, cw, policy, mean, std, include_la, out, att_d=0.3):
    fn = make_policy_fn(policy, mean, std, slew_max_accel=2.0, include_lookahead=include_la)
    rec = []
    orig = sd.PurePursuitTracker.step
    def patched(self, cur_pos, _r=rec, _o=orig):
        tv, ci = _o(self, cur_pos); _r.append(ci); return tv, ci
    sd.PurePursuitTracker.step = patched
    try:
        sd.run(shape=shape, seed=seed, gui=False, policy_fn=fn, att_d_gain_scale=att_d,
               output_folder=out, clockwise=cw)
    finally:
        sd.PurePursuitTracker.step = orig
    rec = np.array(rec); n = len(rec)
    cov = len(np.unique(rec)) / N_IDX
    laps = abs(wrapped(np.diff(rec)).sum() / N_IDX)
    q = n // 4
    last_adv = abs(wrapped(np.diff(rec[-q:])).sum()) / N_IDX if q > 1 else 0.0   # laps advanced in last 25%
    csvf = max(glob.glob(os.path.join(out, 'shape_dataset', '*.csv')), key=os.path.getmtime)
    with open(csvf) as f:
        rows = list(csv.DictReader(f))
    pe = np.linalg.norm([[float(r['tx-x']), float(r['ty-y']), float(r['tz-z'])] for r in rows], axis=1)
    os.remove(csvf)
    return laps, cov, last_adv, float(pe.mean())


def verdict(laps, cov, last_adv, dist):
    if laps >= 2.0:
        return 'COMPLETES'
    if dist > 0.8:
        return 'LOST(off-path)'            # meters off the path -> diverged, not tracing
    if last_adv < 0.02:
        return 'STUCK(frozen)'             # not advancing at the end
    return 'SLOW(advancing)'               # still moving forward, just not 3 laps in time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--shapes', nargs='+', default=['triangle', 'square', 'pentagon', 'circle'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[500, 501, 502])
    ap.add_argument('--label', default=None)
    ap.add_argument('--att-d-gain-scale', type=float, default=0.3)
    A = ap.parse_args()
    cfg = json.load(open(os.path.join(A.run_dir, 'config.json')))
    include_la = bool(cfg.get('include_lookahead'))
    mean, std, ab = load_normalization(A.run_dir)
    policy = load_policy(A.run_dir, max_action=ab)
    out = f'/tmp/evalstuck_{os.getpid()}'
    os.makedirs(os.path.join(out, 'shape_dataset'), exist_ok=True)
    print(f"# {A.label or os.path.basename(A.run_dir.rstrip('/'))}  seeds {A.seeds}  (both dirs)")
    print(f"{'shape':10} {'laps':>5} {'cov':>5} {'lastQ_adv':>10} {'dist':>6}   verdict (majority)")
    for shape in A.shapes:
        rows = [rollout(shape, s, cw, policy, mean, std, include_la, out, A.att_d_gain_scale)
                for s in A.seeds for cw in (False, True)]
        r = np.array(rows)
        vs = [verdict(*x) for x in rows]
        maj = max(set(vs), key=vs.count)
        print(f"{shape:10} {r[:,0].mean():>5.2f} {r[:,1].mean():>5.2f} {r[:,2].mean():>10.3f} "
              f"{r[:,3].mean():>6.2f}   {maj}  {dict((v, vs.count(v)) for v in set(vs))}")


if __name__ == '__main__':
    main()
