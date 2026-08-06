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
    def __init__(self, feat, H, ch=128, depth=6, dilated=False):
        super().__init__()
        self.H = H
        self.inp = nn.Conv1d(feat, ch, 5, padding=2)
        #### dilated=True grows dilation 1,2,4,8,16,... so the receptive field spans the whole window
        #### (kernel 5, depth 6 -> RF ~33 < H=64 when undilated; dilation lifts RF past 64).
        self.blocks = nn.ModuleList()
        for i in range(depth):
            d = (2 ** (i % 5)) if dilated else 1
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d), nn.GroupNorm(8, ch), nn.SiLU()))
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
    ap.add_argument('--csv', default=None, help='training CSV (required unless --load)')
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
    ap.add_argument('--lambda-cons', type=float, default=0.1,
                    help='weight of the pos_err<->vel physical-consistency loss (0 disables)')
    ap.add_argument('--act-cap', type=float, default=0.0,
                    help='cap generated |action| at this m/s; <=0 uses the real data max')
    ap.add_argument('--pe-asinh', type=float, default=0.0,
                    help='asinh(pos_err/scale) transform to de-tail pos_err before normalizing; 0=off, e.g. 0.05')
    ap.add_argument('--dilated', action='store_true',
                    help='dilated conv blocks (1,2,4,8,16) to expand the temporal receptive field past H')
    ap.add_argument('--sampler', choices=['ddpm', 'ddim'], default='ddpm')
    ap.add_argument('--ddim-steps', type=int, default=100, help='reverse steps for DDIM (subsampled from T)')
    ap.add_argument('--eta', type=float, default=0.0, help='DDIM stochasticity (0=deterministic)')
    ap.add_argument('--load', default=None, help='load a saved model.pt and RESAMPLE only (skip training)')
    ap.add_argument('--gen-batch', type=int, default=200, help='windows sampled per reverse-diffusion pass')
    A = ap.parse_args()
    if not A.load and not A.csv:
        ap.error('--csv is required unless --load is given')
    if torch.cuda.is_available():
        dev = 'cuda'
    elif torch.backends.mps.is_available():
        dev = 'mps'
    else:
        dev = 'cpu'; torch.set_num_threads(1)
    H = A.horizon

    PE_COLS = [0, 1, 2]

    def make_to_phys(mean, std, c):
        def to_phys(arr):                                     # normalized(+transformed) -> physical units
            a = arr * std + mean
            if c > 0:
                a[..., PE_COLS] = c * np.sinh(a[..., PE_COLS])  # invert asinh
            return a
        return to_phys

    if A.load:
        #### RESAMPLE-ONLY: rebuild the model from a saved checkpoint, skip data-load + training entirely.
        ck = torch.load(A.load, map_location=dev, weights_only=False)
        sa = ck['args']
        mean, std, act_cap = ck['mean'], ck['std'], ck['act_cap']
        c = sa.get('pe_asinh', 0.0); H = ck['H']; A.T = sa.get('T', A.T)
        model = TemporalDenoiser(FD, H, ch=sa['ch'], depth=sa['depth'], dilated=sa.get('dilated', False)).to(dev)
        model.load_state_dict(ck['ema']); model.eval()
        betas = torch.linspace(1e-4, 0.02, A.T).to(dev); abar = torch.cumprod(1 - betas, 0)
        to_phys = make_to_phys(mean, std, c); X = None
        os.makedirs(f'{A.out}/shape_dataset', exist_ok=True)
        print(f"[load] {A.load}: ch={sa['ch']} depth={sa['depth']} dilated={sa.get('dilated', False)} "
              f"asinh={c} act_cap={act_cap:.3f} T={A.T} -> resample {A.n_gen} via {A.sampler}")
        return _generate(A, dev, model, betas, abar, to_phys, act_cap, H, X)

    X = load_windows(A.csv, H, A.stride, A.limit)             # (N, H, 19) PHYSICAL units
    print(f"[data] {len(X)} windows of ({H}, {FD}) (device {dev})")
    #### asinh(pos_err/scale): linear near 0 (keeps the ~6mm precise-tracking bulk resolvable) but
    #### log-compresses the heavy tail (max ~5.7m) so per-channel std isn't set by rare deviations.
    PE_COLS = [0, 1, 2]; c = A.pe_asinh
    Xtr = X.copy()
    if c > 0:
        Xtr[..., PE_COLS] = np.arcsinh(Xtr[..., PE_COLS] / c)
        print(f"[norm] pos_err asinh(x/{c}) transform ON")
    mean = Xtr.reshape(-1, FD).mean(0); std = Xtr.reshape(-1, FD).std(0) + 1e-6   # per-channel
    Xt = torch.tensor((Xtr - mean) / std, dtype=torch.float32)
    mean_t = torch.tensor(mean, device=dev); std_t = torch.tensor(std, device=dev)  # for physical-space losses
    dt = 1.0 / A.control_freq                                  # step time for the consistency loss
    PE = slice(0, 3); VEL = slice(7, 10)                       # pos_err and velocity channels within a step
    #### physical action cap = the real data's max target-speed magnitude (the slew/max_speed clip, ~1.4)
    act_cap = float(np.linalg.norm(X.reshape(-1, FD)[:, SD:SD + AD], axis=1).max()) if A.act_cap <= 0 else A.act_cap
    print(f"[clamp] action magnitude capped at real max = {act_cap:.3f} m/s")

    model = TemporalDenoiser(FD, H, ch=A.ch, depth=A.depth, dilated=A.dilated).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] ch={A.ch} depth={A.depth} dilated={A.dilated} params={n_params/1e6:.2f}M batch={A.batch} ema={A.ema}")
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
        eps_pred = model(xt, t)
        loss_diff = ((eps_pred - noise) ** 2).mean()
        #### physical pos_err<->vel consistency on the PREDICTED x0 (denormalized).
        #### pos_err = target - pos  =>  d(pos_err)/dt = v_target - vel  =>  v_target = d(pos_err)/dt + vel.
        #### the path (target) is smooth, so penalize v_target's step-to-step change (target accel).
        #### jitter in pos_err makes d(pos_err)/dt explode -> this term forces pos_err & vel to agree.
        loss_cons = torch.tensor(0.0, device=dev)
        if A.lambda_cons > 0:
            x0_pred = (xt - (1 - ab).sqrt() * eps_pred) / ab.sqrt()   # DDPM x0 estimate (normalized)
            x0_pred = x0_pred.clamp(-3.0, 3.0)                        # bound before denorm/sinh (avoids inf at high t)
            phys = x0_pred * std_t + mean_t                           # de-normalize (pe still asinh-space)
            pe, vel = phys[..., PE], phys[..., VEL]                   # (B,H,3) each
            if c > 0:
                pe = c * torch.sinh(pe)                               # invert asinh -> physical meters
            #### everything in per-step METERS: target displacement d_tgt = Δpos_err + vel·dt
            d_tgt = (pe[:, 1:] - pe[:, :-1]) + vel[:, :-1] * dt       # (B,H-1,3)
            acc_tgt = d_tgt[:, 1:] - d_tgt[:, :-1]                    # target accel (2nd diff) (B,H-2,3)
            cons_per = (acc_tgt ** 2).mean(dim=(1, 2))               # (B,)
            loss_cons = (abar[t] * cons_per).mean()                  # weight by x0 confidence (low noise)
        loss = loss_diff + A.lambda_cons * loss_cons
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                                  # update EMA shadow
            for k, v in model.state_dict().items():
                ema[k].mul_(A.ema).add_(v.detach(), alpha=1 - A.ema)
        if step % 2000 == 0:
            print(f"[train] step {step:>6} lr {lr:.2e} loss {loss.item():.4f} "
                  f"(diff {loss_diff.item():.4f} cons {loss_cons.item():.4f})", flush=True)

    model.load_state_dict(ema)                                  # sample from EMA weights
    model.eval()
    os.makedirs(f'{A.out}/shape_dataset', exist_ok=True)
    torch.save({'ema': ema, 'mean': mean, 'std': std, 'act_cap': act_cap,
                'args': vars(A), 'FD': FD, 'H': H}, f'{A.out}/model.pt')  # resample without retraining
    to_phys = make_to_phys(mean, std, c)
    return _generate(A, dev, model, betas, abar, to_phys, act_cap, H, X)


def _generate(A, dev, model, betas, abar, to_phys, act_cap, H, X):
    """Sample trajectories and write one per-episode CSV each (shared by train and --load paths)."""
    @torch.no_grad()
    def ddpm_sample(n):                                          # ancestral: injects noise each step
        x = torch.randn(n, H, FD, device=dev)
        for t in reversed(range(A.T)):
            tt = torch.full((n,), t, device=dev)
            pred = model(x, tt); a = 1 - betas[t]; ab = abar[t]
            x = (x - (betas[t] / (1 - ab).sqrt()) * pred) / a.sqrt()
            if t > 0:
                x = x + betas[t].sqrt() * torch.randn_like(x)
            x = x.clamp(-3.0, 3.0)   # keep the reverse process inside the normalized data range
        return x.cpu().numpy()

    @torch.no_grad()
    def ddim_sample(n, steps, eta=0.0):                          # DDIM: eta=0 -> deterministic, no noise
        seq = np.linspace(0, A.T - 1, steps, dtype=int)[::-1]    # subsampled descending timesteps
        one = torch.tensor(1.0, device=dev)
        x = torch.randn(n, H, FD, device=dev)
        for i, t in enumerate(seq):
            tt = torch.full((n,), int(t), device=dev)
            eps = model(x, tt); ab = abar[int(t)]
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-3.0, 3.0)   # predicted clean x0
            t_prev = int(seq[i + 1]) if i + 1 < len(seq) else -1
            ab_prev = abar[t_prev] if t_prev >= 0 else one
            sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).clamp(min=0).sqrt() if eta > 0 else 0.0
            coef = (1 - ab_prev - (sigma ** 2 if eta > 0 else 0.0)).clamp(min=0).sqrt()   # dir toward x_t
            x = ab_prev.sqrt() * x0 + coef * eps
            if eta > 0 and t_prev >= 0:
                x = x + sigma * torch.randn_like(x)
        return x.cpu().numpy()

    def sample(n):
        return ddim_sample(n, A.ddim_steps, A.eta) if A.sampler == 'ddim' else ddpm_sample(n)

    def pe_jerk(Gphys):                                          # (n,H,FD) -> mean pos_err 2nd-diff
        pe = Gphys[:, :, 0:3]
        return float(np.linalg.norm(pe[:, 2:] - 2 * pe[:, 1:-1] + pe[:, :-2], axis=2).mean())
    cmp = 64
    print(f"[sampler] pe_jerk  DDPM={pe_jerk(to_phys(ddpm_sample(cmp))):.5f}  "
          f"DDIM({A.ddim_steps},eta={A.eta})={pe_jerk(to_phys(ddim_sample(cmp, A.ddim_steps, A.eta))):.5f}  "
          f"(real≈0.00135)", flush=True)

    #### Same column layout as a real per-episode CSV (NO episode_id — merge adds it).
    HDR = ['step'] + STATE + ACT + ['reward', 'done']
    made = 0
    allG = []                                                  # keep generated (physical) for stats
    while made < A.n_gen:
        k = min(A.gen_batch, A.n_gen - made)
        G = to_phys(sample(k))                                 # (k, H, 19) physical (asinh inverted)
        for w in range(k):
            traj = G[w]                                        # (H, 19)
            #### renormalize the quaternion each step so attitudes are valid unit quats
            q = traj[:, QUAT]; traj[:, QUAT] = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
            s = traj[:, :SD]; act = traj[:, SD:SD + AD]
            #### clamp action magnitude to the real physical max (preserve direction, scale down only)
            an = np.linalg.norm(act, axis=1, keepdims=True)
            act = act * np.minimum(1.0, act_cap / (an + 1e-8))
            reward = -np.linalg.norm(s[:, 0:3], axis=1)        # reward = -|pos_err|, recomputed
            fn = f'{A.out}/shape_dataset/gentraj-{made + w:05d}.csv'
            with open(fn, 'w', newline='') as f:
                wr = csv.writer(f); wr.writerow(HDR)
                for i in range(H):
                    wr.writerow([i, *s[i], *act[i], reward[i], i == H - 1])
        if len(allG) < 20:                                     # cap stats memory for very large gen runs
            allG.append(G)
        made += k
        print(f"[gen] wrote {made}/{A.n_gen} trajectory CSVs", flush=True)
    print(f"[done] {made} trajectory episodes (H={H}) -> {A.out}/shape_dataset/ (one CSV each)")

    #### ---- quality report: generated (vs real if available) ----
    G = np.concatenate(allG, 0).reshape(-1, FD)
    gn = np.linalg.norm(G[:, SD:SD + AD], axis=1, keepdims=True)   # apply the same action cap as written
    G[:, SD:SD + AD] *= np.minimum(1.0, act_cap / (gn + 1e-8))
    def blk(a, sl): return np.linalg.norm(a[:, sl], axis=1)
    def line(tag, a):
        pe, ve, ac = blk(a, slice(0, 3)), blk(a, slice(7, 10)), blk(a, slice(16, 19))
        print(f"  {tag:4} pos_err|mean|={pe.mean():.3f} (med {np.median(pe):.4f} max {pe.max():.2f})  "
              f"vel={ve.mean():.2f}  act|mean|={ac.mean():.2f} (max {ac.max():.2f})  quat_norm={blk(a, QUAT).mean():.3f}")
    print("[quality] generated vs real (pos_err/vel/action magnitudes, quat norm):")
    line('GEN', G)
    if X is not None:
        line('REAL', X.reshape(-1, FD))


if __name__ == '__main__':
    main()
