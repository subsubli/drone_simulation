"""Minimal data-identified X500 CTBR simulator + closed-loop eval (policy.onnx vs PD expert).

Dynamics are fitted by least squares on the TRAIN flights of data_track.csv.gz:
  body rate:  w[t+1] = w[t] + sum_l b_l * sp[t-l] + c * w[t]   (per axis, l = 0..5)
  velocity:   v[t+1] = v[t] + dt * (kT * T[t-2] * R[:,2] - g_eff * z - D * v)   (D per axis, only with --drag)
No EKF noise, no wind, no motor saturation beyond the [-1,1] action clip.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

DT, LAGS, T_LAG = .01, 6, 2
YAW_DATA = np.deg2rad(-94.)  # data median heading (PX4 start yaw); yaw 0 is OOD for the policy
YAW0 = YAW_DATA  # start heading of the current episode (rollout() may jitter it, see --yaw-span)
YAW_SPAN = 0.
T_MAX, RATE_MAX = 34.19, np.deg2rad([220., 220., 200.])


def load(csv):
    with gzip.open(csv, 'rt') as f:
        t = np.genfromtxt(f, delimiter=',', names=True, dtype=np.float64)
    col = lambda p, n: np.column_stack([t[f'{p}_{i}'] for i in range(n)])
    return t, col('obs', 25), col('act', 4), col('next_obs', 25)


def lagged(x, lag, flight):
    """x[t-lag] within the same flight, mask of rows where that exists."""
    ok = np.r_[np.zeros(lag, bool), flight[lag:] == flight[:len(flight)-lag]] if lag else np.ones(len(x), bool)
    return np.roll(x, lag, 0), ok


def features(t, o, a, no):
    """Regression design shared by fit() and heldout_r2(): per rate axis (X, y), and velocity (X, y)."""
    f = t['flight']
    sp, w = a[:, 1:]*RATE_MAX, o[:, 15:18]
    ok = lagged(sp, LAGS-1, f)[1]
    rate = [(np.column_stack([lagged(sp[:, ax], l, f)[0] for l in range(LAGS)] + [w[:, ax]]), no[:, 15+ax]-w[:, ax])
            for ax in range(3)]
    T = lagged((a[:, 0]+1)/2*T_MAX, T_LAG, f)[0]
    z, v = o[:, [5, 8, 11]], o[:, 12:15]  # body z axis in world = column 2 of row-major R
    n = len(o)
    Xv = np.zeros((n, 3, 5))  # columns: kT, g, drag_x, drag_y, drag_z  (a = kT*T*z - g*zhat - D*v)
    Xv[:, :, 0], Xv[:, 2, 1] = T[:, None]*z, -1.
    for ax in range(3):
        Xv[:, ax, 2+ax] = -v[:, ax]
    return ok, rate, (Xv, (no[:, 12:15]-v)/DT)


def fit(t, o, a, no, rows, drag=False):
    ok, rate, (Xv, yv) = features(t, o, a, no)
    ok &= rows
    b_rate = np.array([np.linalg.lstsq(X[ok], y[ok], rcond=None)[0] for X, y in rate])
    cols = 5 if drag else 2
    th = np.linalg.lstsq(Xv[ok][:, :, :cols].reshape(-1, cols), yv[ok].ravel(), rcond=None)[0]
    th = np.r_[th, np.zeros(5-cols)]
    return dict(rate=b_rate, kT=th[0], g=th[1], drag=th[2:])


def heldout_r2(model, t, o, a, no, rows):
    """R2 of the GIVEN model's one-step predictions on `rows` (no refit). [wx, wy, wz, v]."""
    ok, rate, (Xv, yv) = features(t, o, a, no)
    ok &= rows
    r2 = lambda y, p: 1-((y-p)**2).sum()/((y-y.mean())**2).sum()
    out = [r2(y[ok], X[ok]@b) for (X, y), b in zip(rate, model['rate'])]
    th = np.r_[model['kT'], model['g'], model['drag']]
    return out+[r2(yv[ok].ravel(), (Xv[ok]@th).ravel())]


def replay(model, t, o, a, rows, horizons=(1, 10, 50, 100), every=50):
    """Open-loop multi-step check: start the sim at a real state, feed the LOGGED actions, compare
    h steps later with the real state. Baseline 'hold' = assume the state does not change."""
    f, P = t['flight'], np.column_stack([t['pos_x'], t['pos_y'], t['pos_z']])
    H = max(horizons)
    err = {h: [] for h in horizons}
    for s in np.where(rows)[0][::every]:
        if s < LAGS or s+H >= len(o) or f[s-LAGS] != f[s] or f[s+H] != f[s]:
            continue
        sim = X500Sim(model, P[s], o[s, 12:15])
        sim.R, sim.w = o[s, 3:12].reshape(3, 3).copy(), o[s, 15:18].copy()
        sim.hist = [a[s-1-l].copy() for l in range(LAGS)]
        for k in range(H):
            sim.step(a[s+k])
            if k+1 in err:
                e = s+k+1
                Rt = o[e, 3:12].reshape(3, 3)
                ang = np.degrees(np.arccos(np.clip((np.trace(Rt.T@sim.R)-1)/2, -1, 1)))
                ang0 = np.degrees(np.arccos(np.clip((np.trace(Rt.T@o[s, 3:12].reshape(3, 3))-1)/2, -1, 1)))
                err[k+1].append([np.linalg.norm(sim.p-P[e]), np.linalg.norm(sim.v-o[e, 12:15]),
                                 np.linalg.norm(sim.w-o[e, 15:18]), ang,
                                 np.linalg.norm(P[s]+o[s, 12:15]*(k+1)*DT-P[e]),  # hold-velocity baseline
                                 np.linalg.norm(o[s, 12:15]-o[e, 12:15]), np.linalg.norm(o[s, 15:18]-o[e, 15:18]), ang0])
    return {h: np.sqrt(np.mean(np.square(v), 0)) for h, v in err.items()}, len(err[H])


def so3_exp(v):
    th = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if th < 1e-9:
        return np.eye(3)+K
    return np.eye(3)+np.sin(th)/th*K+(1-np.cos(th))/th**2*K@K


class X500Sim:
    def __init__(self, model, p, v):
        self.m, self.p, self.v = model, p.copy(), v.copy()
        c, s = np.cos(YAW0), np.sin(YAW0)
        self.R, self.w = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]]), np.zeros(3)
        hover = 2*self.m['g']/self.m['kT']/T_MAX-1
        self.hist = [np.array([hover, 0, 0, 0])]*LAGS  # newest first

    def step(self, a):
        a = np.clip(a, -1, 1)
        self.hist = [a]+self.hist[:-1]
        sp = np.array([h[1:]*RATE_MAX for h in self.hist])  # (LAGS,3)
        R, w = self.R, self.w
        T = (self.hist[T_LAG][0]+1)/2*T_MAX
        b = self.m['rate']
        self.w = w+(b[:, :LAGS]*sp.T).sum(1)+b[:, LAGS]*w
        self.v = self.v+DT*(self.m['kT']*T*R[:, 2]-[0, 0, self.m['g']]-self.m['drag']*self.v)
        self.p = self.p+DT*self.v
        self.R = R@so3_exp(w*DT)

    def obs(self, target, target_vel):
        return np.r_[target-self.p, self.R.ravel(), self.v, self.w, self.hist[0], target_vel].astype(np.float32)

    def crashed(self):
        return self.p[2] < .15 or self.p[2] > 6 or abs(self.p[0]) > 5 or abs(self.p[1]) > 5 or self.R[2, 2] < np.cos(np.deg2rad(75))


def trajectory(rng):
    """Circle or figure-8, radius 0.8-1.5 m, period 6-10 s, vertical wiggle 0-0.3 m (DATASET.md §7)."""
    r, per, h, eight = rng.uniform(.8, 1.5), rng.uniform(6, 10), rng.uniform(0, .3), rng.random() < .5
    sgn, om, z0 = rng.choice([-1, 1]), 2*np.pi/per, 2.3
    def at(t):
        s = om*t
        if eight:
            p = [r*np.sin(s), sgn*r*np.sin(2*s)/2, z0+h*np.sin(s)]
            v = [r*om*np.cos(s), sgn*r*om*np.cos(2*s), h*om*np.cos(s)]
        else:
            p = [r*np.cos(s)-r, sgn*r*np.sin(s), z0+h*np.sin(s)]
            v = [-r*om*np.sin(s), sgn*r*om*np.cos(s), h*om*np.cos(s)]
        return np.array(p), np.array(v)
    return at, ('eight' if eight else 'circle'), per


def pd_expert(model, sim, tgt, tvel, kp=4., kd=3.5, katt=8.):
    """Geometric PD (gains from DATASET.md §7), yaw held at YAW0."""
    acc = kp*(tgt-sim.p)+kd*(tvel-sim.v)+[0, 0, model['g']]
    T = acc@sim.R[:, 2]/model['kT']
    zd = acc/np.linalg.norm(acc)
    yd = np.cross(zd, [np.cos(YAW0), np.sin(YAW0), 0]); yd /= np.linalg.norm(yd)
    Rd = np.column_stack([np.cross(yd, zd), yd, zd])
    E = Rd.T@sim.R-sim.R.T@Rd
    w = -katt*.5*np.array([E[2, 1], E[0, 2], E[1, 0]])
    return np.r_[2*T/T_MAX-1, w/RATE_MAX]


def load_policy(run_dirs):
    """Deterministic action of one run, or the action-average of several (ensemble)."""
    import torch
    from src.policy import GaussianPolicy
    nets = []
    for d in run_dirs:
        cfg = json.loads((d/'config.json').read_text())
        pol = GaussianPolicy(25, 4, hidden_dim=cfg['hidden_dim'], n_hidden=cfg['n_hidden'], max_action=1.)
        st = torch.load(d/'final.pt', map_location='cpu', weights_only=True)
        pol.load_state_dict({k.removeprefix('policy.'): v for k, v in st.items() if k.startswith('policy.')})
        norm = np.load(d/'obs_normalization.npz')
        nets.append((pol.eval(), torch.from_numpy(norm['mean']), torch.from_numpy(norm['std'])))
    @torch.no_grad()
    def act(sim, tgt, tvel):
        x = torch.from_numpy(sim.obs(tgt, tvel))[None]
        return np.mean([p((x-m)/s).mean[0].numpy() for p, m, s in nets], 0).astype(np.float64)
    return act


def reward(ob, a, crashed):
    """DATASET.md §5, evaluated on the current obs (matches the logged reward)."""
    ve, dact = ob[22:25]-ob[12:15], a-ob[18:22]
    return -(ob[0:3]@ob[0:3]+.05*ve@ve+.02*ob[15:18]@ob[15:18]+.5*dact@dact)-10.*crashed


def rollout(model, act_fn, seed, seconds, log=None, flight=0):
    """Fly act_fn. If `log` is a list, append DAgger rows: PD label at every visited state, with
    next_obs from a CLONE of the sim stepped by that label (so the logged action explains the transition)."""
    import copy
    global YAW0
    YAW0 = YAW_DATA+np.deg2rad(np.random.default_rng(seed+10**6).uniform(-YAW_SPAN, YAW_SPAN))  # own rng: same trajectories
    rng = np.random.default_rng(seed)
    at, shape, per = trajectory(rng)
    p0, v0 = at(0.)
    sim = X500Sim(model, p0, v0)
    errs = []
    steps = int(seconds/DT)
    for k in range(steps):
        tgt, tvel = at(k*DT)
        if log is not None:
            ob, lab = sim.obs(tgt, tvel), np.clip(pd_expert(model, sim, tgt, tvel), -1, 1)
            twin = copy.deepcopy(sim); twin.step(lab)
            nxt = twin.obs(*at((k+1)*DT)); dead = twin.crashed()
            log.append(np.r_[ob, lab, reward(ob, lab, dead), nxt, dead, k == steps-1, flight, k*DT])
        sim.step(act_fn(sim, tgt, tvel))
        errs.append(np.linalg.norm(at((k+1)*DT)[0]-sim.p))
        if sim.crashed():
            if log is not None and not log[-1][55]:
                log[-1][56] = 1.  # policy crashed here, label's twin did not: cut as timeout, not terminal
            return dict(seed=seed, shape=shape, crashed=True, t=(k+1)*DT, err=float(np.mean(errs)), max=float(np.max(errs)))
    return dict(seed=seed, shape=shape, crashed=False, t=seconds, err=float(np.mean(errs)), max=float(np.max(errs)))


def summary(name, rs):
    for shape in ('circle', 'eight', 'all'):
        s = [r for r in rs if shape in ('all', r['shape'])]
        ok = [r for r in s if not r['crashed']]
        if not s:
            continue
        print(f"{name:7s} {shape:6s} n={len(s):3d} crash={len(s)-len(ok):3d}  "
              f"mean_err={np.mean([r['err'] for r in s]):.3f} m  "
              f"survivor_err={np.mean([r['err'] for r in ok]) if ok else float('nan'):.3f} m  "
              f"max={np.max([r['max'] for r in s]):.2f} m  mean_t={np.mean([r['t'] for r in s]):.1f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dirs', type=Path, nargs='+', help='one run, or several = action-averaged ensemble')
    ap.add_argument('--csv-file', type=Path, default=Path('../datasets/data_track.csv.gz'))
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--seconds', type=float, default=20.)
    ap.add_argument('--out', type=Path, help='eval json (default: <first run>/sim_eval.json)')
    ap.add_argument('--collect', type=Path, help='write DAgger CSV here instead of evaluating')
    ap.add_argument('--drag', action='store_true', help='linear drag; off: held-out R2 +0.004 only, replay worse, z-drag fits <0')
    ap.add_argument('--validate', action='store_true', help='only print sim validation (held-out R2, replay)')
    ap.add_argument('--yaw-span', type=float, default=0., help='start yaw = -94 + U(-span, span) deg per episode')
    ap.add_argument('--seed-base', type=int, default=1000, help='eval 1000; use another range to collect')
    args = ap.parse_args()
    global YAW_SPAN
    YAW_SPAN = args.yaw_span
    t, o, a, no = load(args.csv_file)
    split = json.loads((args.run_dirs[0]/'dataset_split.json').read_text())
    val = np.isin(t['flight'], split['validation_flights'])
    model = fit(t, o, a, no, ~val, drag=args.drag)
    print('identified kT=%.4f g=%.3f drag=%s hover_a0=%.3f' % (model['kT'], model['g'], model['drag'].round(4), 2*model['g']/model['kT']/T_MAX-1))
    print('rate coeffs (lags0..5, w):\n', model['rate'].round(4))
    print('one-step R2 [wx,wy,wz,v]  train', np.round(heldout_r2(model, t, o, a, no, ~val), 3),
          ' HELD-OUT val (train-fit, no refit)', np.round(heldout_r2(model, t, o, a, no, val), 3))
    if args.validate:
        for name, m in (('no-drag', fit(t, o, a, no, ~val, drag=False)), ('drag', fit(t, o, a, no, ~val))):
            print(f'[{name}] held-out one-step R2', np.round(heldout_r2(m, t, o, a, no, val), 4), 'drag', m['drag'].round(4))
            rm, n = replay(m, t, o, a, val)
            print(f'[{name}] open-loop replay on val flights (n={n} starts), RMSE: pos m | vel m/s | rate rad/s | att deg   || hold-baseline')
            for h, r in rm.items():
                print(f'   {h:3d} steps ({h*DT:.2f}s): {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:5.2f}   || {r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} | {r[7]:5.2f}')
        return
    policy = load_policy(args.run_dirs)
    seeds = range(args.seed_base, args.seed_base+args.episodes)

    if args.collect:
        rows, res = [], []
        first = int(t['flight'].max())+1000  # flight ids disjoint from the real data
        for i, s in enumerate(seeds):
            res.append(rollout(model, policy, s, args.seconds, rows, first+i))
        summary('driver', res)
        head = ([f'obs_{i}' for i in range(25)]+[f'act_{i}' for i in range(4)]+['reward']
                + [f'next_obs_{i}' for i in range(25)]+['terminal', 'timeout', 'flight', 't'])
        with gzip.open(args.collect, 'wt') as f:
            np.savetxt(f, np.array(rows), delimiter=',', header=','.join(head), comments='', fmt='%.7g')
        print(f'{len(rows)} rows, {len(res)} flights, terminals={int(np.array(rows)[:, 55].sum())} -> {args.collect}')
        return

    expert = lambda sim, tgt, tvel: pd_expert(model, sim, tgt, tvel)
    res = {name: [rollout(model, fn, s, args.seconds) for s in seeds] for name, fn in (('PD', expert), ('IQL', policy))}
    for name, rs in res.items():
        summary(name, rs)
    (args.out or args.run_dirs[0]/'sim_eval.json').write_text(json.dumps(dict(
        runs=[str(d) for d in args.run_dirs],
        model=dict(kT=model['kT'], g=model['g'], drag=model['drag'].tolist(), rate=model['rate'].tolist(),
                   heldout_r2=heldout_r2(model, t, o, a, no, val)),
        episodes=args.episodes, seconds=args.seconds, yaw_span=args.yaw_span, results=res), indent=1))


if __name__ == '__main__':
    main()
