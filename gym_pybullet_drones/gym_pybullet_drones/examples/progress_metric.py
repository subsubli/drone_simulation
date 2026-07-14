"""Progress-aware metric: path coverage + net laps. mean|pos_err| misses this entirely.
Handles the prev-action (16-dim) policy via evaluate_trained_policy's own loaders."""
import sys, json, os
sys.path.insert(0, '/Users/hanjakp/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples')
import numpy as np
import shape_dataset as sd
from evaluate_trained_policy import load_policy, load_normalization, make_policy_fn

RUN = sys.argv[1] if len(sys.argv) > 1 else \
    '/Users/hanjakp/drone_simulation/IQL-PyTorch-main/runs/merged/07-14-26_21.31.07_lwyv_multishape'
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 500
cfg = json.load(open(os.path.join(RUN, 'config.json')))
include_prev = bool(cfg.get('include_prev_action'))
include_la = bool(cfg.get('include_lookahead'))
mean, std, ab = load_normalization(RUN)
policy = load_policy(RUN, max_action=ab)
print(f"RUN={os.path.basename(RUN)}  include_prev_action={include_prev}  include_lookahead={include_la}")
print(f"{'shape':10s} {'coverage':>9s} {'net laps':>9s}  verdict")
for shape in ['triangle', 'square', 'pentagon', 'circle']:
    fn = make_policy_fn(policy, mean, std, slew_max_accel=2.0,
                        include_prev_action=include_prev, include_lookahead=include_la)
    orig = sd.PurePursuitTracker.step
    rec = []
    def patched(self, cur_pos, _r=rec, _o=orig):
        tv, ci = _o(self, cur_pos); _r.append(ci); return tv, ci
    sd.PurePursuitTracker.step = patched
    try:
        sd.run(shape=shape, seed=SEED, gui=False, policy_fn=fn, att_d_gain_scale=0.3, output_folder='/tmp/_prog_junk')
    finally:
        sd.PurePursuitTracker.step = orig
    rec = np.array(rec); N = 3000
    cov = len(np.unique(rec)) / N
    d = np.diff(rec); d = np.where(d < -N/2, d + N, np.where(d > N/2, d - N, d))
    laps = d.sum() / N
    verdict = 'TRAVERSES' if cov > 0.8 else ('partial' if cov > 0.3 else 'STUCK')
    print(f"{shape:10s} {cov*100:>8.1f}% {laps:>9.2f}  {verdict}")
