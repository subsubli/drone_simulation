"""Transition diffusion (SynthER-style) for the drone offline-RL CSV.

Learns the joint (s, a, r, s') distribution of a merged CSV with a small DDPM, then samples
synthetic transitions and writes them as **2-row mini-episodes** (row0 = s,a,r,done=False ;
row1 = s',0,0,done=True). That way the EXISTING pipeline consumes them unchanged:
merge_shape_dataset.py concatenates the per-file CSVs (re-assigning episode_id), and
drone_dataset.py reads each 2-row episode back as one (s, a, r, s') transition.

    conda activate iql
    python transition_diffusion.py --csv data_soft/merged1.5M_soft.csv \
        --steps 30000 --n-gen 200000 --per-file 20000 --out gen_diffusion

Then (separately) merge: python merge_shape_dataset.py --input_folder gen_diffusion/shape_dataset ...
"""
import os, argparse, csv
import numpy as np
import torch
import torch.nn as nn

STATE = ['tx-x', 'ty-y', 'tz-z', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'lx', 'ly', 'lz']
ACT = ['ax', 'ay', 'az']
SD, AD = len(STATE), len(ACT)          # 16, 3
DIM = SD + AD + 1 + SD                  # [s(16) | a(3) | r(1) | s'(16)] = 36
QOFF = (0, SD + AD + 1)                 # quaternion block starts (s' block offset = 20); quat = [off+3:off+7]


def load_transitions(csv_file, limit=None):
    """Extract (s_t, a_t, r_t, s_{t+1}) for consecutive steps within the SAME episode."""
    with open(csv_file) as f:
        rd = csv.reader(f); hdr = next(rd); ix = {n: i for i, n in enumerate(hdr)}
        sc = [ix[c] for c in STATE]; ac = [ix[c] for c in ACT]; rc = ix['reward']
        ec = ix.get('episode_id')
        prev = None; prev_e = None; out = []
        for r in rd:
            s = np.array([float(r[j]) for j in sc], dtype=np.float32)
            a = np.array([float(r[j]) for j in ac], dtype=np.float32)
            rew = float(r[rc]); e = int(r[ec]) if ec is not None else 0
            if prev is not None and prev_e == e:
                out.append(np.concatenate([prev[0], prev[1], [prev[2]], s]))
            prev = (s, a, rew); prev_e = e
            if limit and len(out) >= limit:
                break
    return np.array(out, dtype=np.float32)


class Denoiser(nn.Module):
    def __init__(self, dim, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + 1, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, dim))

    def forward(self, x, t):
        return self.net(torch.cat([x, (t.float() / 1000.0)[:, None]], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--n-gen', type=int, default=200000)
    ap.add_argument('--per-file', type=int, default=20000)
    ap.add_argument('--out', default='gen_diffusion')
    ap.add_argument('--T', type=int, default=1000)
    ap.add_argument('--limit', type=int, default=None, help='cap transitions loaded (quick test)')
    A = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(1)

    X = load_transitions(A.csv, A.limit)
    print(f"[data] {len(X)} transitions, dim {X.shape[1]} (device {dev})")
    mean = X.mean(0); std = X.std(0) + 1e-6
    Xt = torch.tensor((X - mean) / std, dtype=torch.float32)

    model = Denoiser(DIM).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    betas = torch.linspace(1e-4, 0.02, A.T).to(dev)
    abar = torch.cumprod(1 - betas, 0)

    for step in range(A.steps):
        idx = torch.randint(0, len(Xt), (256,))
        x0 = Xt[idx].to(dev); t = torch.randint(0, A.T, (256,), device=dev)
        noise = torch.randn_like(x0)
        ab = abar[t][:, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        loss = ((model(xt, t) - noise) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            print(f"[train] step {step:>6} loss {loss.item():.4f}")

    @torch.no_grad()
    def sample(n):
        x = torch.randn(n, DIM, device=dev)
        for t in reversed(range(A.T)):
            tt = torch.full((n,), t, device=dev)
            pred = model(x, tt); a = 1 - betas[t]; ab = abar[t]
            x = (x - (betas[t] / (1 - ab).sqrt()) * pred) / a.sqrt()
            if t > 0:
                x = x + betas[t].sqrt() * torch.randn_like(x)
        return x.cpu().numpy()

    os.makedirs(f'{A.out}/shape_dataset', exist_ok=True)
    HDR = ['episode_id', 'step'] + STATE + ACT + ['reward', 'done']
    written = 0; fi = 0
    while written < A.n_gen:
        k = min(A.per_file, A.n_gen - written)
        g = sample(k) * std + mean
        #### Re-normalize both quaternion blocks so generated attitudes are valid unit quats.
        for off in QOFF:
            q = g[:, off + 3:off + 7]
            g[:, off + 3:off + 7] = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        fn = f'{A.out}/shape_dataset/gen-{fi:04d}.csv'
        with open(fn, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(HDR)
            for e in range(k):
                s = g[e, 0:SD]; a = g[e, SD:SD + AD]; r = g[e, SD + AD]
                sp = g[e, SD + AD + 1:SD + AD + 1 + SD]
                w.writerow([e, 0, *s, *a, r, False])          # row0: (s, a, r), not terminal
                w.writerow([e, 1, *sp, 0.0, 0.0, 0.0, 0.0, True])  # row1: s' as next state, terminal
        written += k; fi += 1
        print(f"[gen] wrote {fn} ({k} transitions)")
    print(f"[done] {written} synthetic transitions in {fi} files -> {A.out}/shape_dataset/")


if __name__ == '__main__':
    main()
