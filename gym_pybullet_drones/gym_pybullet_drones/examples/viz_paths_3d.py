"""Interactive 3D window (rotatable with the mouse): a trained policy tracing all 4 shapes,
2x2, target path vs flown path. Run in the `drones` env.

    KMP_DUPLICATE_LIB_OK=TRUE python viz_paths_3d.py <RUN_DIR> [seed]

RUN_DIR is an IQL-PyTorch-main/runs/.../<timestamp> dir (has final.pt + config.json + npz).
"""
import sys, json, os
import numpy as np
import matplotlib.pyplot as plt   # default (GUI) backend so plt.show() opens a window
import shape_dataset as sd
from evaluate_trained_policy import load_policy, load_normalization, make_policy_fn

RUN = sys.argv[1] if len(sys.argv) > 1 else None
if RUN is None:
    sys.exit("usage: python viz_paths_3d.py <RUN_DIR> [seed]")
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 500

cfg = json.load(open(os.path.join(RUN, 'config.json')))
inc_prev = bool(cfg.get('include_prev_action')); inc_la = bool(cfg.get('include_lookahead'))
mean, std, ab = load_normalization(RUN)
policy = load_policy(RUN, max_action=ab)

fig = plt.figure(figsize=(13, 11))
for k, shape in enumerate(['triangle', 'square', 'pentagon', 'circle']):
    actual, target = [], []
    #### Fresh policy_fn PER shape -- make_policy_fn holds slew/prev state in its closure.
    base = make_policy_fn(policy, mean, std, slew_max_accel=2.0,
                          include_prev_action=inc_prev, include_lookahead=inc_la)
    def wrapped(pos_err, state, lookahead=None, _a=actual, _t=target, _b=base):
        _a.append(state[0:3].copy()); _t.append(state[0:3] + np.asarray(pos_err))
        return _b(pos_err, state, lookahead)
    sd.run(shape=shape, seed=SEED, gui=False, policy_fn=wrapped, att_d_gain_scale=0.3,
           output_folder='/tmp/_viz_paths_3d_junk')
    actual = np.array(actual); target = np.array(target)
    ax = fig.add_subplot(2, 2, k + 1, projection='3d')
    ax.plot(target[:, 0], target[:, 1], target[:, 2], 'k--', lw=1.2, label='target path')
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], 'b-', lw=1.2, label='policy flown')
    allp = np.vstack([target, actual]); mid = (allp.min(0) + allp.max(0)) / 2
    half = max((allp.max(0) - allp.min(0)).max(), 1e-6) / 2
    ax.set_xlim(mid[0]-half, mid[0]+half); ax.set_ylim(mid[1]-half, mid[1]+half); ax.set_zlim(mid[2]-half, mid[2]+half)
    ax.set_box_aspect((1, 1, 1)); ax.set_title(shape)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    if k == 0:
        ax.legend(fontsize=8)
fig.suptitle(f'{os.path.basename(RUN)} -- all 4 shapes, seed {SEED}', fontsize=12)
fig.tight_layout()
plt.show()
