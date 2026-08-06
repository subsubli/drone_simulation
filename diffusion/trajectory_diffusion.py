"""Trajectory diffusion (Diffuser-lite) for the drone offline-RL CSV.

Instead of individual transitions (which forced fake terminal rows), this learns fixed-length
H-step TRAJECTORY windows [ (s,a) per step ] with a DDPM and generates whole episodes. Each
generated window is written as ONE per-episode CSV (H rows), so the existing merge_shape_dataset.py
(file == one episode) and drone_dataset.py consume them unchanged — no fake transitions, and
`reward` is recomputed from the generated pos_err, `done` is only the last step.

    conda activate iql
    python trajectory_diffusion.py --csv data_soft/merged1.5M_soft.csv \
        --horizon 64 --steps 30000 --n-gen 500 --out gen_traj

Then (separately): python merge_shape_dataset.py --input_folder gen_traj/shape_dataset ...
"""
import os, argparse, csv
import numpy as np
import torch
import torch.nn as nn

STATE = ['tx-x', 'ty-y', 'tz-z', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'lx', 'ly', 'lz']
ACT = ['ax', 'ay', 'az']
SD, AD = len(STATE), len(ACT)          # 16, 3
FD = SD + AD                           # per-step features = 19 (state + action)
QUAT = slice(3, 7)                     # quaternion within the state block of each step


def load_windows(csv_file, H, stride, limit=None):
    """Slice each episode into H-step windows of [state(16) | action(3)] = (H, 19)."""
    with open(csv_file) as f:
        rd = csv.reader(f); hdr = next(rd); ix = {n: i for i, n in enumerate(hdr)}
        sc = [ix[c] for c in STATE]; ac = [ix[c] for c in ACT]; ec = ix.get('episode_id')
        cur = []; cur_e = None; wins = []
        def flush(ep):
            arr = np.array(ep, dtype=np.float32)               # (L, 19)
            for i in range(0, len(arr) - H + 1, stride):
                wins.append(arr[i:i + H])
        for r in rd:
            feat = [float(r[j]) for j in sc] + [float(r[j]) for j in ac]
            e = int(r[ec]) if ec is not None else 0
            if cur_e is not None and e != cur_e:
                flush(cur); cur = []
            cur.append(feat); cur_e = e
            if limit and len(wins) >= limit:
                break
        if cur:
            flush(cur)
    return np.array(wins, dtype=np.float32)                     # (N, H, 19)


class TemporalDenoiser(nn.Module):
    """1D conv over time so the model sees the sequence structure (not a flattened blob).

    depth controls how many residual conv blocks stack in the middle; ch is width. For the
    quality run we go wider (ch=128) and deeper (depth=6) than the fast CPU baseline (64/3)."""
    def __init__(self, feat, H, ch=128, depth=6):
        super().__init__()
        self.H = H
        self.inp = nn.Conv1d(feat, ch, 5, padding=2)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv1d(ch, ch, 5, padding=2), nn.GroupNorm(8, ch), nn.SiLU()) for _ in range(depth)])
        self.out = nn.Conv1d(ch, feat, 5, padding=2)
        self.temb = nn.Linear(1, ch)

    def forward(self, x, t):                                    # x: (B, H, feat)
        h = self.inp(x.transpose(1, 2))                        # (B, ch, H)
        h = h + self.temb((t.float() / 1000)[:, None])[:, :, None]
        for blk in self.blocks:                                # residual conv stack
            h = h + blk(h)
        return self.out(h).transpose(1, 2)                    # (B, H, feat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--horizon', type=int, default=64)
    ap.add_argument('--stride', type=int, default=32)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--n-gen', type=int, default=500)
    ap.add_argument('--out', default='gen_traj')
    ap.add_argument('--T', type=int, default=1000)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--control-freq', type=int, default=100)
    ap.add_argument('--ch', type=int, default=128)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--lr-min', type=float, default=1e-5)
    A = ap.parse_args()
    if torch.cuda.is_available():
        dev = 'cuda'
    elif torch.backends.mps.is_available():
        dev = 'mps'
    else:
        dev = 'cpu'; torch.set_num_threads(1)
    H = A.horizon

    X = load_windows(A.csv, H, A.stride, A.limit)             # (N, H, 19)
    print(f"[data] {len(X)} windows of ({H}, {FD}) (device {dev})")
    mean = X.reshape(-1, FD).mean(0); std = X.reshape(-1, FD).std(0) + 1e-6   # per-channel
    Xt = torch.tensor((X - mean) / std, dtype=torch.float32)

    model = TemporalDenoiser(FD, H, ch=A.ch, depth=A.depth).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] ch={A.ch} depth={A.depth} params={n_params/1e6:.2f}M batch={A.batch} ema={A.ema}")
    opt = torch.optim.Adam(model.parameters(), A.lr)
    #### warmup (linear 0->lr) then cosine decay lr->lr_min over the remaining steps
    def lr_at(step):
        if step < A.warmup:
            return A.lr * (step + 1) / A.warmup
        prog = (step - A.warmup) / max(1, A.steps - A.warmup)   # 0..1
        cos = 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))
        return A.lr_min + (A.lr - A.lr_min) * cos
    print(f"[lr] schedule: warmup {A.warmup} steps 0->{A.lr:g}, then cosine ->{A.lr_min:g} over {A.steps} steps")
    #### EMA weights — standard DDPM quality trick; sample from the smoothed shadow, not the raw weights
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    betas = torch.linspace(1e-4, 0.02, A.T).to(dev)
    abar = torch.cumprod(1 - betas, 0)

    for step in range(A.steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g['lr'] = lr
        idx = torch.randint(0, len(Xt), (A.batch,))
        x0 = Xt[idx].to(dev); t = torch.randint(0, A.T, (A.batch,), device=dev)
        noise = torch.randn_like(x0)
        ab = abar[t][:, None, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        loss = ((model(xt, t) - noise) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                                  # update EMA shadow
            for k, v in model.state_dict().items():
                ema[k].mul_(A.ema).add_(v.detach(), alpha=1 - A.ema)
        if step % 2000 == 0:
            print(f"[train] step {step:>6} lr {lr:.2e} loss {loss.item():.4f}", flush=True)

    model.load_state_dict(ema)                                  # sample from EMA weights
    model.eval()

    @torch.no_grad()
    def sample(n):
        x = torch.randn(n, H, FD, device=dev)
        for t in reversed(range(A.T)):
            tt = torch.full((n,), t, device=dev)
            pred = model(x, tt); a = 1 - betas[t]; ab = abar[t]
            x = (x - (betas[t] / (1 - ab).sqrt()) * pred) / a.sqrt()
            if t > 0:
                x = x + betas[t].sqrt() * torch.randn_like(x)
            x = x.clamp(-3.0, 3.0)   # keep the reverse process inside the normalized data range
        return x.cpu().numpy()

    os.makedirs(f'{A.out}/shape_dataset', exist_ok=True)
    #### Same column layout as a real per-episode CSV (NO episode_id — merge adds it).
    HDR = ['step'] + STATE + ACT + ['reward', 'done']
    made = 0
    allG = []                                                  # keep generated (unnormalized) for stats
    while made < A.n_gen:
        k = min(200, A.n_gen - made)
        G = sample(k) * std + mean                             # (k, H, 19)
        for w in range(k):
            traj = G[w]                                        # (H, 19)
            #### renormalize the quaternion each step so attitudes are valid unit quats
            q = traj[:, QUAT]; traj[:, QUAT] = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
            s = traj[:, :SD]; act = traj[:, SD:SD + AD]
            reward = -np.linalg.norm(s[:, 0:3], axis=1)        # reward = -|pos_err|, recomputed
            fn = f'{A.out}/shape_dataset/gentraj-{made + w:05d}.csv'
            with open(fn, 'w', newline='') as f:
                wr = csv.writer(f); wr.writerow(HDR)
                for i in range(H):
                    wr.writerow([i, *s[i], *act[i], reward[i], i == H - 1])
        allG.append(G)
        made += k
        print(f"[gen] wrote {made}/{A.n_gen} trajectory CSVs", flush=True)
    print(f"[done] {made} trajectory episodes (H={H}) -> {A.out}/shape_dataset/ (one CSV each)")

    #### ---- quality report: generated vs real, on the same channels ----
    G = np.concatenate(allG, 0).reshape(-1, FD)                # (n_gen*H, 19)
    R = X.reshape(-1, FD)                                      # real windows, same feature layout
    def blk(a, sl): return np.linalg.norm(a[:, sl], axis=1)
    def line(tag, a):
        pe, ve, ac = blk(a, slice(0, 3)), blk(a, slice(7, 10)), blk(a, slice(16, 19))
        qn = blk(a, QUAT)
        print(f"  {tag:4} pos_err|mean|={pe.mean():.3f} (med {np.median(pe):.4f} max {pe.max():.2f})  "
              f"vel={ve.mean():.2f}  act|mean|={ac.mean():.2f} (max {ac.max():.2f})  quat_norm={qn.mean():.3f}")
    print("[quality] generated vs real (pos_err/vel/action magnitudes, quat norm):")
    line('GEN', G); line('REAL', R)


if __name__ == '__main__':
    main()
