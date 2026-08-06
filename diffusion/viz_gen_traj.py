"""Visualize ONE generated trajectory window vs a REAL window (same reconstruction).

The state is PATH-RELATIVE (pos_err + look-ahead), so there is no absolute world position. We
reconstruct the flown path by integrating velocity (dt = 1/hz); the reference it tracked is
flown + pos_err. Left = generated window, right = a real window from the soft merged CSV, both
with TRUE equal aspect so the shape isn't distorted. This is the honest side-by-side: does a
generated 0.64s segment look like a real one?

    conda activate drones
    python viz_gen_traj.py gen_traj_quality/shape_dataset/gentraj-00000.csv \
        --real ../gym_pybullet_drones/gym_pybullet_drones/examples/data_soft/merged1.5M_soft.csv
"""
import argparse, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WANT = ['tx-x', 'ty-y', 'tz-z', 'vx', 'vy', 'vz']            # pos_err(3) + vel(3); skip str cols


def load_gen(csv_file):
    with open(csv_file) as f:
        rd = csv.reader(f); hdr = next(rd); ix = {n: i for i, n in enumerate(hdr)}
        col = [ix[c] for c in WANT]
        a = np.array([[float(r[j]) for j in col] for r in rd])
    return a[:, 0:3], a[:, 3:6]


def load_real_window(csv_file, H, target_pe=0.023, tol=0.015, max_scan=400000):
    """Find a REPRESENTATIVE real H-window: precise-tracking (mean |pos_err| near target_pe),
    matching the generated one — not a rare heavy-tail deviation window. Slides within episodes."""
    with open(csv_file) as f:
        rd = csv.reader(f); hdr = next(rd); ix = {n: i for i, n in enumerate(hdr)}
        ecol = ix['episode_id']; col = [ix[c] for c in WANT]
        cur = []; cur_e = None; scanned = 0; best = None; best_d = 1e9
        def scan(ep):
            nonlocal best, best_d
            a = np.array(ep)
            for i in range(0, len(a) - H, H // 2):           # stride H/2
                w = a[i:i + H]
                m = np.linalg.norm(w[:, 0:3], axis=1).mean()
                d = abs(m - target_pe)
                if d < best_d:
                    best_d, best = d, w
                if d <= tol:
                    return True
            return False
        for r in rd:
            e = int(float(r[ecol]))
            if cur_e is not None and e != cur_e:
                if scan(cur):
                    cur = None; break
                cur = []
            cur.append([float(r[j]) for j in col]); cur_e = e
            scanned += 1
            if scanned >= max_scan:
                break
        if cur:
            scan(cur)
    w = best
    print(f"[real] picked window mean|pos_err|={np.linalg.norm(w[:,0:3],axis=1).mean():.3f} (target {target_pe})")
    return w[:, 0:3], w[:, 3:6]


def reconstruct(pos_err, vel, dt):
    flown = np.cumsum(vel * dt, axis=0); flown -= flown[0]   # integrate velocity, start at origin
    ref = flown + pos_err                                    # reference = flown + tracking error
    return flown, ref, np.linalg.norm(pos_err, axis=1)


def set_equal(ax, pts):
    """True equal aspect: box proportional to data ranges (no axis distortion)."""
    r = np.ptp(pts, axis=0); r[r < 1e-6] = 1e-6
    ax.set_box_aspect(r)


def draw(ax, flown, ref, err, title):
    ax.plot(*ref.T, '-', color='tab:blue', lw=2, label='reference (flown+pos_err)')
    ax.plot(*flown.T, '-', color='tab:orange', lw=2, label='flown (∫vel)')
    ax.scatter(*flown[0], color='k', s=40); ax.scatter(*flown[-1], color='r', s=40)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    set_equal(ax, np.vstack([flown, ref]))
    ax.set_title(f'{title}\n|pos_err| mean {err.mean():.3f} max {err.max():.3f} m', fontsize=10)
    ax.legend(fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--real', default=None, help='soft merged CSV to draw a real window alongside')
    ap.add_argument('--hz', type=int, default=100)
    ap.add_argument('--out', default=None)
    A = ap.parse_args()
    dt = 1.0 / A.hz
    pe_g, v_g = load_gen(A.csv); H = len(pe_g)
    fg, rg, eg = reconstruct(pe_g, v_g, dt)

    ncol = 2 if A.real else 1
    fig = plt.figure(figsize=(6.5 * ncol, 5.5))
    ax = fig.add_subplot(1, ncol, 1, projection='3d')
    draw(ax, fg, rg, eg, f'GENERATED ({H} steps, {H*dt:.2f}s)')
    if A.real:
        pe_r, v_r = load_real_window(A.real, H)
        fr, rr, er = reconstruct(pe_r, v_r, dt)
        ax2 = fig.add_subplot(1, ncol, 2, projection='3d')
        draw(ax2, fr, rr, er, f'REAL ({H} steps, {H*dt:.2f}s)')

    out = A.out or A.csv.rsplit('/', 1)[-1].replace('.csv', '_vs_real.png')
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"[viz] -> {out}  GEN flown span {np.ptp(fg,0).round(3).tolist()} |pe| {eg.mean():.3f}")


if __name__ == '__main__':
    main()
