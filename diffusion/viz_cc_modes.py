"""One figure: the class-conditional generator's mode separation (§17).

|pos_err| distribution (log-x) for real soft data vs the cc pool split into its
two learned modes (on-path = window max<=0.2m, off-path = window max>0.2m).
Shows: on-path is a tight precise-tracking bulk with a SHARP cutoff (no smeared
tail), off-path is a cleanly SEPARATE recovery cluster ~1m, and real spans both
with a long heavy tail the generator compresses. Saves cc_mode_separation.png.
"""
import os, csv, glob, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THR = 0.2
EX = os.path.expanduser('~/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples')
REAL = f'{EX}/data_soft/shape_dataset'
CCPOOL = 'gen_pool_cc/shape_dataset'


def pe_of_files(files):
    """|pos_err| per row (flat) and per-window max, for a list of per-episode CSVs."""
    flat, wmax = [], []
    for fn in files:
        with open(fn) as f:
            rows = [(float(r['tx-x']), float(r['ty-y']), float(r['tz-z'])) for r in csv.DictReader(f)]
        a = np.linalg.norm(rows, axis=1)
        flat.append(a); wmax.append(a.max())
    return np.concatenate(flat), np.array(wmax)


def sample(folder, n, seed=0):
    fs = sorted(glob.glob(os.path.join(folder, '*.csv')))
    random.Random(seed).shuffle(fs)
    return fs[:n]


# real: sample episodes; cc: sample windows then split into the two modes by window-max
real_pe, _ = pe_of_files(sample(REAL, 120))
cc_files = sample(CCPOOL, 4000)
on_files, off_files = [], []
for fn in cc_files:
    with open(fn) as f:
        rows = [(float(r['tx-x']), float(r['ty-y']), float(r['tz-z'])) for r in csv.DictReader(f)]
    (off_files if np.linalg.norm(rows, axis=1).max() > THR else on_files).append(fn)
cc_on, _ = pe_of_files(on_files)
cc_off, _ = pe_of_files(off_files)

print(f"real rows {len(real_pe)}, cc on-path windows {len(on_files)} ({len(cc_on)} rows), "
      f"off-path windows {len(off_files)} ({len(cc_off)} rows)")

bins = np.logspace(np.log10(2e-4), np.log10(50), 90)
fig, ax = plt.subplots(figsize=(9, 5.2))
for pe, color, label in [
    (real_pe, '#444444', f'real soft  (median {np.median(real_pe)*1000:.0f}mm, max {real_pe.max():.1f}m)'),
    (cc_on,  '#1f77b4', f'cc on-path  (median {np.median(cc_on)*1000:.0f}mm, max {cc_on.max():.2f}m)'),
    (cc_off, '#d62728', f'cc off-path  (median {np.median(cc_off):.2f}m, max {cc_off.max():.2f}m)'),
]:
    ax.hist(pe, bins=bins, density=True, histtype='step', linewidth=2.2, color=color, label=label)
    ax.hist(pe, bins=bins, density=True, alpha=0.10, color=color)

ax.axvline(THR, color='k', ls='--', lw=1, alpha=0.6)
ax.text(THR*1.07, ax.get_ylim()[1]*0.92, '0.2m\non/off\nthreshold', fontsize=8, va='top')
ax.set_xscale('log')
ax.set_xlabel('|pos_err|  (m, log scale)')
ax.set_ylabel('density')
ax.set_title('Class-conditional generator: clean mode separation (§17)\n'
             'on-path = tight precise-tracking bulk, sharp cutoff (no smeared tail);  '
             'off-path = separate ~1m recovery cluster')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
out = 'cc_mode_separation.png'
fig.savefig(out, dpi=140)
print(f'saved {out}')
