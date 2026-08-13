"""Diagnose sub-2-lap ensemble rollouts: SLOW (tracks whole path, just <2 laps in the window)
vs ESCAPE (follows ~1 lap then leaves the path). Records the closest-path-index trace and
per-step |pos_err| for each rollout and reports:
  cov          = unique path indices visited / N_IDX  (1.0 = traced the whole path)
  laps         = net laps
  lastQ_adv    = laps gained in the FINAL QUARTER of the episode (~0 => frozen/stopped)
  pe_firsthalf / pe_lasthalf = mean |pos_err| in first vs last half (rising late => escaping)
  pe_max_t     = fraction into the episode where |pos_err| peaks (late peak => escape)
"""
import os, sys, json, glob, csv, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shape_dataset as sd
from evaluate_trained_policy import load_policy, load_normalization
from eval_ensemble import make_ensemble_fn
N_IDX = 3000

def net_from(rec):
    rec = np.array(rec)
    if len(rec) < 2: return 0.0, 0.0
    cov = len(np.unique(rec)) / N_IDX
    d = np.diff(rec); d = np.where(d < -N_IDX/2, d+N_IDX, np.where(d > N_IDX/2, d-N_IDX, d))
    return cov, abs(d.sum()/N_IDX)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dirs', nargs='+', required=True)
    ap.add_argument('--shapes', nargs='+', default=['circle'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(500,506)))
    ap.add_argument('--att-d-gain-scale', type=float, default=1.0)
    A = ap.parse_args()
    pols, means, stds = [], [], []
    la = False
    for rd in A.run_dirs:
        cfg = json.load(open(os.path.join(rd,'config.json'))); la = bool(cfg.get('include_lookahead'))
        m,s,ab = load_normalization(rd); pols.append(load_policy(rd, max_action=ab)); means.append(m); stds.append(s)
    out = f'/tmp/diagens_{os.getpid()}'; os.makedirs(os.path.join(out,'shape_dataset'), exist_ok=True)
    print(f"{'shape/seed/dir':16} {'laps':>5} {'cov':>5} {'lastQ_adv':>9} {'pe_1st':>7} {'pe_2nd':>7} {'pe_max':>7} {'peak@':>6} verdict")
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
                           output_folder=out, clockwise=cw)
                finally:
                    sd.PurePursuitTracker.step = orig
                cov, laps = net_from(rec)
                q = max(1, len(rec)//4)
                _, lastq = net_from(rec[-q:])
                csvf = max(glob.glob(os.path.join(out,'shape_dataset','*.csv')), key=os.path.getmtime)
                rows = list(csv.DictReader(open(csvf))); os.remove(csvf)
                pe = np.array([np.linalg.norm([float(r['tx-x']),float(r['ty-y']),float(r['tz-z'])]) for r in rows])
                h = len(pe)//2
                pe1, pe2 = pe[:h].mean(), pe[h:].mean()
                peak_t = int(pe.argmax())/len(pe)
                # verdict
                if laps >= 2.0: v = "COMPLETE"
                elif cov > 0.9 and lastq > 0.15: v = "SLOW(tracking)"
                elif pe2 > 2*pe1 + 0.3 or peak_t > 0.6 and pe.max() > 1.0: v = "ESCAPE(late)"
                elif lastq < 0.05: v = "STUCK(frozen)"
                else: v = "SLOW-ish"
                d = 'cw' if cw else 'ccw'
                print(f"{shape[:8]:8}/{seed}/{d:3} {laps:>5.2f} {cov:>5.2f} {lastq:>9.2f} {pe1:>7.3f} {pe2:>7.3f} {pe.max():>7.3f} {peak_t:>6.2f} {v}")

if __name__ == '__main__':
    main()
