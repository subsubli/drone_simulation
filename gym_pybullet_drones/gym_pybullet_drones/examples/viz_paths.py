"""Use shape_dataset's own plot_path (which draws the FULL target path) + top-down view,
saved to PNG, to check whether the policy actually traverses the whole shape or gets stuck."""
import sys, json, os
sys.path.insert(0, '/Users/hanjakp/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shape_dataset as sd
from evaluate_trained_policy import load_policy, load_normalization, make_policy_fn

RUN = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: python viz_paths.py <RUN_DIR> [seed]")
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 500
cfg = json.load(open(os.path.join(RUN, 'config.json')))
inc_prev = bool(cfg.get('include_prev_action')); inc_la = bool(cfg.get('include_lookahead'))
mean, std, ab = load_normalization(RUN)
policy = load_policy(RUN, max_action=ab)

big = plt.figure(figsize=(14, 12))
for k, shape in enumerate(['triangle', 'square', 'pentagon', 'circle']):
    fn = make_policy_fn(policy, mean, std, slew_max_accel=2.0,
                        include_prev_action=inc_prev, include_lookahead=inc_la)
    sd.run(shape=shape, seed=SEED, gui=False, policy_fn=fn, plot_path=True,
           att_d_gain_scale=0.3, output_folder='/tmp/_viz_junk3')
    src = plt.gcf().axes[0]                 # the axes plot_path just drew (full target + actual)
    tgt_line, act_line = src.lines[0], src.lines[1]
    ax = big.add_subplot(2, 2, k + 1, projection='3d')
    ax.plot(*tgt_line.get_data_3d(), 'k--', lw=1.2, label='FULL target path')
    ax.plot(*act_line.get_data_3d(), 'b-', lw=1.0, alpha=0.8, label='policy flown')
    import numpy as np
    allp = np.vstack([np.array(tgt_line.get_data_3d()).T, np.array(act_line.get_data_3d()).T])
    mid = (allp.min(0) + allp.max(0)) / 2; half = max((allp.max(0) - allp.min(0)).max(), 1e-6) / 2
    ax.set_xlim(mid[0]-half, mid[0]+half); ax.set_ylim(mid[1]-half, mid[1]+half); ax.set_zlim(mid[2]-half, mid[2]+half)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=90, azim=-90)   # top-down
    ax.set_title(f'{shape} (top-down)'); ax.set_xlabel('X'); ax.set_ylabel('Y')
    if k == 0: ax.legend(fontsize=8)
    plt.close(plt.get_fignums()[0] if plt.get_fignums()[0] != big.number else plt.get_fignums()[-1])

big.suptitle('Does the policy traverse the WHOLE shape? (full target path vs flown, top-down)', fontsize=13)
big.tight_layout()
OUT = 'policy_paths.png'
big.savefig(OUT, dpi=110); print(f"saved {os.path.abspath(OUT)}")
