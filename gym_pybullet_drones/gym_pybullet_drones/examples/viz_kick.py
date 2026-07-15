"""Visualize ONE expert episode that has perturbation kicks: target path vs the drone's
flown path (top-down), with the kick points marked. Shows what a kick + recovery looks like.

    python viz_kick.py [shape=square] [seed=1] [mag=0.3] [count=2]
"""
import sys, os, glob, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shape_dataset as sd

shape = sys.argv[1] if len(sys.argv) > 1 else 'square'
seed  = int(sys.argv[2]) if len(sys.argv) > 2 else 1
mag   = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
count = int(sys.argv[4]) if len(sys.argv) > 4 else 2
kk    = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0   # adaptive_lookahead_k
out = '/tmp/_kickviz'
os.system(f'rm -rf {out}')

#### plot_path=True makes shape_dataset draw (full target path, flown path) into a figure.
sd.run(shape=shape, seed=seed, gui=False, plot_path=True,
       perturb_prob=1.0, perturb_count=count, perturb_magnitude=mag,
       att_d_gain_scale=0.3, clockwise=(seed % 2 == 1), adaptive_lookahead_k=kk, output_folder=out)
src = plt.gcf().axes[0]
tgt = np.array(src.lines[0].get_data_3d()).T   # full target path (absolute)
act = np.array(src.lines[1].get_data_3d()).T   # drone flown path (absolute), one point per step

#### Kick steps = where |pos_err| jumps (the kick teleports the drone off-path in one step).
f = glob.glob(f'{out}/shape_dataset/*.csv')[0]
rows = list(csv.reader(open(f))); ix = {n: i for i, n in enumerate(rows[0])}
perr = np.array([[float(r[ix[c]]) for c in ['tx-x', 'ty-y', 'tz-z']] for r in rows[1:]])
d = np.linalg.norm(perr, axis=1)
kicks = (np.where(np.diff(d) > mag * 0.5)[0] + 1).tolist()

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111)
ax.plot(tgt[:, 0], tgt[:, 1], 'k--', lw=1.0, label='target path')
ax.plot(act[:, 0], act[:, 1], 'b-', lw=1.0, alpha=0.8, label='drone flown')
for j in kicks:
    if j < len(act):
        ax.plot(act[j, 0], act[j, 1], 'r*', ms=16, zorder=5)
ax.plot([], [], 'r*', ms=12, label=f'kick ({len(kicks)}x, mag {mag}m)')
ax.set_aspect('equal'); ax.legend(fontsize=9)
ax.set_title(f'{shape} seed{seed} — expert with {count}x kick perturbation (top-down)')
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
fig.tight_layout()
fig.savefig('kick_viz.png', dpi=110)
print(f"saved {os.path.abspath('kick_viz.png')}")
print(f"kicks at steps {kicks}, |pos_err| max={d.max():.2f}m")
