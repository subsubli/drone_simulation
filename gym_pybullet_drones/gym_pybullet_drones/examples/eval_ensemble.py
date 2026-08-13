"""Ensemble evaluation: average the RAW target_vel of N trained policies at each rollout step,
then apply ONE slew cap to the averaged command. Same net-laps/dist metrics & output format as
eval_aug.py. Motivation (§34e): 512+LN INIT has large training-seed variance (~±25/500), so
averaging several seeds'/taus' policies should reduce variance and may raise completion.

    conda activate drones
    KMP_DUPLICATE_LIB_OK=TRUE python eval_ensemble.py --run-dirs R1 R2 ... --shapes ... --direction both
"""
import os, sys, json, csv, argparse, glob
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shape_dataset as sd
from evaluate_trained_policy import load_policy, load_normalization
from eval_aug import net_laps

ATT = 1.0
N_IDX = 3000  # path discretization (matches eval_aug/progress_metric)


def make_ensemble_fn(policies, means, stds, include_la, slew_max_accel=2.0, control_freq_hz=100):
    """Average raw MLP target_vel across policies, then slew-cap the mean (shared continuity)."""
    max_delta_v = None if slew_max_accel is None else slew_max_accel / control_freq_hz
    prev_slew = np.zeros(3)

    def fn(pos_err, state, lookahead=None):
        nonlocal prev_slew
        base = np.concatenate([pos_err, state[3:7], state[10:13], state[13:16]]).astype(np.float32)
        if include_la:
            base = np.concatenate([base, np.asarray(lookahead, dtype=np.float32)]).astype(np.float32)
        raws = []
        for pol, m, s in zip(policies, means, stds):
            obs = ((base - m) / s).astype(np.float32)
            with torch.no_grad():
                raws.append(pol.act(torch.from_numpy(obs), deterministic=True).numpy())
        raw = np.mean(raws, axis=0)                       # action-space averaging
        action = raw
        if max_delta_v is not None:
            delta = raw - prev_slew
            dmag = np.linalg.norm(delta)
            if dmag > max_delta_v:
                action = prev_slew + delta * (max_delta_v / dmag)
            prev_slew = action
        return action
    return fn


def rollout(shape, seed, cw, policies, means, stds, include_la, out, att_d, n_laps=None, stop_at=None):
    fn = make_ensemble_fn(policies, means, stds, include_la)
    rec = []
    orig = sd.PurePursuitTracker.step
    def patched(self, cur_pos, _r=rec, _o=orig):
        tv, ci = _o(self, cur_pos); _r.append(ci); return tv, ci
    sd.PurePursuitTracker.step = patched
    kw = {} if n_laps is None else {'n_laps': n_laps}
    if stop_at is not None:                     # early-stop once net laps reaches stop_at (saves time)
        st = {'prev': None, 'cum': 0.0}
        def stop_fn(ci):
            if st['prev'] is not None:
                d = ci - st['prev']
                d = d + N_IDX if d < -N_IDX/2 else (d - N_IDX if d > N_IDX/2 else d)
                st['cum'] += d
            st['prev'] = ci
            return abs(st['cum']) / N_IDX >= stop_at
        kw['stop_fn'] = stop_fn
    try:
        sd.run(shape=shape, seed=seed, gui=False, policy_fn=fn, att_d_gain_scale=att_d,
               output_folder=out, clockwise=cw, **kw)
    finally:
        sd.PurePursuitTracker.step = orig
    cov, laps = net_laps(rec)
    csvf = max(glob.glob(os.path.join(out, 'shape_dataset', '*.csv')), key=os.path.getmtime)
    with open(csvf) as f:
        rows = list(csv.DictReader(f))
    pe = np.array([[float(r['tx-x']), float(r['ty-y']), float(r['tz-z'])] for r in rows])
    os.remove(csvf)
    pen = np.linalg.norm(pe, axis=1)
    return laps, cov, float(pen.mean()), pen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dirs', nargs='+', required=True, help='N run dirs to ensemble')
    ap.add_argument('--shapes', nargs='+', default=['triangle', 'square', 'pentagon', 'circle', 'star'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(500, 550)))
    ap.add_argument('--label', default=None)
    ap.add_argument('--att-d-gain-scale', type=float, default=1.0)
    ap.add_argument('--direction', default='both', choices=['both', 'cw', 'ccw'])
    ap.add_argument('--n-laps', type=float, default=None,
                    help='time budget in lap-times (default None=normal 3). Set >3 to remove the speed '
                         'confound and measure completion as "reaches the full 3.0-lap target given time".')
    ap.add_argument('--stop-at', type=float, default=None,
                    help='early-stop each rollout once it reaches this many net laps (saves time; '
                         'e.g. 3.0 with --n-laps 8 to just check "can it reach 3 laps given time").')
    A = ap.parse_args()
    DIRS = {'both': (False, True), 'ccw': (False,), 'cw': (True,)}[A.direction]

    policies, means, stds = [], [], []
    include_la = None
    for rd in A.run_dirs:
        cfg = json.load(open(os.path.join(rd, 'config.json')))
        la = bool(cfg.get('include_lookahead'))
        include_la = la if include_la is None else include_la
        m, s, ab = load_normalization(rd)
        policies.append(load_policy(rd, max_action=ab))
        means.append(m); stds.append(s)
    out = f'/tmp/evalens_{os.getpid()}'
    os.makedirs(os.path.join(out, 'shape_dataset'), exist_ok=True)

    print(f"# ensemble N={len(policies)} {A.label or ''}  seeds {A.seeds[0]}-{A.seeds[-1]} x {A.direction}")
    print(f"# n_laps(time budget)={A.n_laps or 3}")
    print(f"{'shape':10} {'laps mean±std':>16} {'min':>5} {'t2.0':>6} {'t3.0':>6} {'dist mean±std':>16} "
          f"{'d_cmp':>7} {'p90':>6} {'p99':>6} {'max':>7}")
    THRESH = [2.0, 3.0]
    tot = {t: 0 for t in THRESH}
    n_all = 0
    for shape in A.shapes:
        laps, dists, pe_pool = [], [], []
        for seed in A.seeds:
            for cw in DIRS:
                l, cov, derr, pen = rollout(shape, seed, cw, policies, means, stds, include_la, out, A.att_d_gain_scale, A.n_laps, A.stop_at)
                laps.append(l); dists.append(derr); pe_pool.append(pen)
        laps, dists = np.array(laps), np.array(dists)
        pe_all = np.concatenate(pe_pool)
        p90, p99, pmax = np.percentile(pe_all, 90), np.percentile(pe_all, 99), pe_all.max()
        n_all += len(laps)
        for t in THRESH:
            tot[t] += int((laps >= t).sum())
        t20 = int((laps >= 2.0).sum()); t30 = int((laps >= 3.0).sum())
        comp = dists[laps >= 2.0]                              # completing-only cruise error
        cstr = f"{comp.mean():.3f}" if len(comp) else "  -  "
        print(f"{shape:10} {laps.mean():>7.2f}±{laps.std():<7.2f} {laps.min():>5.2f} {t20:>3}/{len(laps)} {t30:>3}/{len(laps)} "
              f"{dists.mean():>7.3f}±{dists.std():<7.3f} {cstr:>7} {p90:>6.3f} {p99:>6.3f} {pmax:>7.3f}")
    for t in THRESH:
        print(f"# TOTAL traverse @>= {t} lap: {tot[t]}/{n_all} ({100*tot[t]/n_all:.1f}%)")


if __name__ == '__main__':
    main()
