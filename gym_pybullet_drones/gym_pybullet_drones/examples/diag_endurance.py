"""Endurance test: give the ensemble policy LOTS of time (n_laps-worth) and check whether the
slow-but-tracking rollouts EVENTUALLY reach 3.0 laps while STAYING on the path -- i.e. is it a
complete tracker that's merely slow, or does it eventually escape? Reports per rollout:
  laps        = total net laps achieved in the extended window
  reach3      = did it hit >= 3.0 laps
  pe_mean/max = |pos_err| over the WHOLE extended run (bounded => never escaped)
  pe_lastT    = mean |pos_err| in the final third (rising/large => late escape)
  end_adv     = laps still gained in the final 10% (still looping at the end)
"""
import os, sys, json, glob, csv, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shape_dataset as sd
from evaluate_trained_policy import load_policy, load_normalization
from eval_ensemble import make_ensemble_fn
N_IDX = 3000

def netlaps(rec):
    rec = np.array(rec)
    if len(rec) < 2: return 0.0
    d = np.diff(rec); d = np.where(d < -N_IDX/2, d+N_IDX, np.where(d > N_IDX/2, d-N_IDX, d))
    return abs(d.sum()/N_IDX)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dirs', nargs='+', required=True)
    ap.add_argument('--shapes', nargs='+', default=['circle', 'triangle'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(500,504)))
    ap.add_argument('--n-laps', type=float, default=9.0, help='time budget in lap-times (default 9 = 3x normal)')
    ap.add_argument('--att-d-gain-scale', type=float, default=1.0)
    A = ap.parse_args()
    pols, means, stds = [], [], []
    la = False
    for rd in A.run_dirs:
        cfg = json.load(open(os.path.join(rd,'config.json'))); la = bool(cfg.get('include_lookahead'))
        m,s,ab = load_normalization(rd); pols.append(load_policy(rd, max_action=ab)); means.append(m); stds.append(s)
    out = f'/tmp/diagend_{os.getpid()}'; os.makedirs(os.path.join(out,'shape_dataset'), exist_ok=True)
    n3 = 0; ntot = 0
    print(f"# endurance: n_laps(time)={A.n_laps}  (normal=3)")
    print(f"{'shape/seed/dir':16} {'laps':>6} {'reach3':>6} {'pe_mean':>7} {'pe_max':>7} {'pe_lastT':>8} {'end_adv':>7}")
    for shape in A.shapes:
        for seed in A.seeds:
            for cw in (False, True):
                fn = make_ensemble_fn(pols, means, stds, la)
                rec = []
                orig = sd.PurePursuitTracker.step
                def patched(self, cur_pos, _r=rec, _o=orig):
                    tv, ci = _o(self, cur_pos); _r.append(ci); return tv, ci
                sd.PurePursuitTracker.step = patched
                try:
                    sd.run(shape=shape, seed=seed, gui=False, policy_fn=fn, att_d_gain_scale=A.att_d_gain_scale,
                           output_folder=out, clockwise=cw, n_laps=A.n_laps)
                finally:
                    sd.PurePursuitTracker.step = orig
                laps = netlaps(rec)
                e = max(1, len(rec)//10)
                end_adv = netlaps(rec[-e:])
                csvf = max(glob.glob(os.path.join(out,'shape_dataset','*.csv')), key=os.path.getmtime)
                rows = list(csv.DictReader(open(csvf))); os.remove(csvf)
                pe = np.array([np.linalg.norm([float(r['tx-x']),float(r['ty-y']),float(r['tz-z'])]) for r in rows])
                t3 = pe[2*len(pe)//3:].mean()
                reach3 = laps >= 3.0
                n3 += int(reach3); ntot += 1
                d = 'cw' if cw else 'ccw'
                print(f"{shape[:8]:8}/{seed}/{d:3} {laps:>6.2f} {str(reach3):>6} {pe.mean():>7.3f} {pe.max():>7.3f} {t3:>8.3f} {end_adv:>7.2f}")
    print(f"# reached 3.0 laps (given time): {n3}/{ntot}")

if __name__ == '__main__':
    main()
