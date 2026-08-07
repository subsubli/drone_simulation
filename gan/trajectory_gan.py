"""Trajectory GAN (WGAN-GP) for the drone offline-RL CSV — a drop-in alternative to
`../diffusion/trajectory_diffusion.py` with the SAME data pipeline, window format, asinh
normalization, class-conditional (on/off-path) setup, checkpoint/--load resample, and per-episode
CSV output, so `build_mix.py` / `merge_shape_dataset.py` / `eval_aug.py` consume its pool unchanged.

Only the generator swaps: instead of a denoising diffusion reverse process, a conditional GAN maps
latent noise (+ on/off-path class) directly to an H-step [state(16)|action(3)] window. The
generator/discriminator reuse the same 1D temporal-conv backbone as the diffusion denoiser.
Discriminator Lipschitz control is selectable via --d-reg: 'sn' = spectral norm + hinge loss
(default; cheaper — no 2nd-order gradient, n_critic 1), 'gp' = WGAN gradient penalty (n_critic ~5).

    conda activate iql
    python trajectory_gan.py --csv ../gym_pybullet_drones/gym_pybullet_drones/examples/data_soft/merged1.5M_soft.csv \
        --class-cond --offpath-batch-frac 0.5 --pe-asinh 0.05 --lambda-cons 0.1 \
        --ch 128 --depth 6 --steps 50000 --n-gen 500 --out gen_gan
    # resample-only from a saved checkpoint (like diffusion's --load):
    python trajectory_gan.py --load gen_gan/model.pt --n-gen 24000 --out gen_pool_gan
"""
import os, argparse, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sn(module, on):
    """Spectral-normalize a layer (Miyato) when `on`, else return it unchanged."""
    return nn.utils.spectral_norm(module) if on else module

STATE = ['tx-x', 'ty-y', 'tz-z', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'lx', 'ly', 'lz']
ACT = ['ax', 'ay', 'az']
SD, AD = len(STATE), len(ACT)          # 16, 3
FD = SD + AD                           # per-step features = 19 (state + action)
QUAT = slice(3, 7)                     # quaternion within the state block of each step

#### class-conditional generation, identical semantics to the diffusion version:
#### 0=on-path (precise-tracking window, max|pos_err|<=thr), 1=off-path (contains a recovery excursion).
#### GANs condition both G and D directly (no null class / classifier-free guidance).
CLS_ON, CLS_OFF = 0, 1
N_CLASSES = 2


def load_windows(csv_file, H, stride, limit=None):
    """Slice each episode into H-step windows of [state(16) | action(3)] = (H, 19). (Same as diffusion.)"""
    with open(csv_file) as f:
        rd = csv.reader(f); hdr = next(rd); ix = {n: i for i, n in enumerate(hdr)}
        sc = [ix[c] for c in STATE]; ac = [ix[c] for c in ACT]; ec = ix.get('episode_id')
        cur = []; cur_e = None; wins = []
        def flush(ep):
            arr = np.array(ep, dtype=np.float32)
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
    return np.array(wins, dtype=np.float32)


class ResBlocks(nn.Module):
    """Shared 1D temporal-conv residual stack (same backbone as the diffusion denoiser).
    `sn=True` spectral-normalizes the convs — used on the discriminator side only."""
    def __init__(self, ch, depth, dilated=False, sn=False):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            d = (2 ** (i % 5)) if dilated else 1
            self.blocks.append(nn.Sequential(
                _sn(nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d), sn), nn.GroupNorm(8, ch), nn.SiLU()))

    def forward(self, h):
        for blk in self.blocks:
            h = h + blk(h)
        return h


class Generator(nn.Module):
    """latent z (+ class) -> (H, feat) window. z is projected to a (ch, H) feature map, the class
    embedding is broadcast-added exactly like the diffusion timestep embedding, then a temporal-conv
    residual stack shapes it and a final conv maps to `feat` channels."""
    def __init__(self, feat, H, latent, ch=128, depth=6, dilated=False, n_classes=0):
        super().__init__()
        self.H = H; self.ch = ch; self.n_classes = n_classes
        self.fc = nn.Linear(latent, ch * H)
        self.res = ResBlocks(ch, depth, dilated)
        self.out = nn.Conv1d(ch, feat, 5, padding=2)
        if n_classes > 0:
            self.cemb = nn.Embedding(n_classes, ch)

    def forward(self, z, c=None):
        h = self.fc(z).view(z.shape[0], self.ch, self.H)          # (B, ch, H)
        if self.n_classes > 0 and c is not None:
            h = h + self.cemb(c)[:, :, None]
        h = self.res(h)
        return self.out(h).transpose(1, 2)                        # (B, H, feat)


class Discriminator(nn.Module):
    """(H, feat) window -> critic score. Conv stack + global average pool, then a linear head plus a
    projection term <embed(c), pooled_features> (Miyato projection discriminator) for conditioning.
    GroupNorm (not BatchNorm) so the WGAN-GP gradient penalty stays well-defined. `sn=True`
    spectral-normalizes every layer (the alternative Lipschitz control, used with the hinge loss)."""
    def __init__(self, feat, ch=128, depth=6, dilated=False, n_classes=0, sn=False):
        super().__init__()
        self.n_classes = n_classes
        self.inp = _sn(nn.Conv1d(feat, ch, 5, padding=2), sn)
        self.res = ResBlocks(ch, depth, dilated, sn=sn)
        self.head = _sn(nn.Linear(ch, 1), sn)
        if n_classes > 0:
            self.cproj = _sn(nn.Embedding(n_classes, ch), sn)

    def forward(self, x, c=None):
        h = self.inp(x.transpose(1, 2))                           # (B, ch, H)
        h = self.res(h)
        h = h.mean(dim=2)                                         # global average pool -> (B, ch)
        out = self.head(h)                                       # (B, 1)
        if self.n_classes > 0 and c is not None:
            out = out + (self.cproj(c) * h).sum(dim=1, keepdim=True)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None, help='training CSV (required unless --load)')
    ap.add_argument('--horizon', type=int, default=64)
    ap.add_argument('--stride', type=int, default=32)
    ap.add_argument('--steps', type=int, default=50000, help='generator update steps')
    ap.add_argument('--n-gen', type=int, default=500)
    ap.add_argument('--out', default='gen_gan')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--control-freq', type=int, default=100)
    ap.add_argument('--ch', type=int, default=128)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--dilated', action='store_true')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--latent-dim', type=int, default=128)
    ap.add_argument('--lr-g', type=float, default=1e-4)
    ap.add_argument('--lr-d', type=float, default=1e-4)
    ap.add_argument('--beta1', type=float, default=0.0, help='Adam beta1 (WGAN-GP default 0.0)')
    ap.add_argument('--beta2', type=float, default=0.9)
    ap.add_argument('--d-reg', choices=['sn', 'gp'], default='sn',
                    help="discriminator Lipschitz control: 'sn'=spectral norm + hinge loss (cheaper, "
                         "no 2nd-order grad); 'gp'=WGAN gradient penalty")
    ap.add_argument('--n-critic', type=int, default=1, help='D updates per outer step (SN-hinge: 1; WGAN-GP: ~5)')
    ap.add_argument('--g-steps', type=int, default=1,
                    help='G updates per outer step; >1 gives the generator more updates than D when D over-powers G')
    #### dynamic D-skip: when D is winning too easily, stop updating it until G catches up ####
    ap.add_argument('--dynamic-d', action='store_true',
                    help='skip the D update on steps where D is already winning (EMA win-rate > --d-win-thr)')
    ap.add_argument('--d-win-thr', type=float, default=0.9,
                    help='D win-rate above which D updates are skipped (only with --dynamic-d)')
    ap.add_argument('--d-win-ema', type=float, default=0.99, help='EMA decay for the tracked D win-rate')
    ap.add_argument('--gp-weight', type=float, default=10.0, help='WGAN gradient-penalty weight (only if --d-reg gp)')
    ap.add_argument('--ema', type=float, default=0.999, help='EMA on the generator weights for sampling')
    ap.add_argument('--lambda-cons', type=float, default=0.1,
                    help='weight of the pos_err<->vel physical-consistency loss on generated windows (0 disables)')
    ap.add_argument('--act-cap', type=float, default=0.0,
                    help='cap generated |action| at this m/s; <=0 uses the real data max')
    ap.add_argument('--pe-asinh', type=float, default=0.0,
                    help='asinh(pos_err/scale) transform before normalizing; 0=off, e.g. 0.05')
    #### class-conditional generation (same knobs/semantics as the diffusion version) ####
    ap.add_argument('--class-cond', action='store_true')
    ap.add_argument('--pe-offpath-thr', type=float, default=0.2,
                    help='|pos_err| (m) above which a window is labeled off-path (matches tables 7/13/14)')
    ap.add_argument('--offpath-batch-frac', type=float, default=-1.0,
                    help='oversample so this fraction of each real batch is off-path (e.g. 0.5); <0 = natural ratio')
    ap.add_argument('--gen-offpath-frac', type=float, default=-1.0,
                    help='fraction of generated windows drawn from the off-path mode; <0 = match the data')
    ap.add_argument('--load', default=None, help='load a saved model.pt and RESAMPLE only (skip training)')
    ap.add_argument('--gen-batch', type=int, default=500, help='windows generated per forward pass')
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
        def to_phys(arr):                                         # normalized(+transformed) -> physical units
            a = arr * std + mean
            if c > 0:
                a[..., PE_COLS] = c * np.sinh(a[..., PE_COLS])    # invert asinh
            return a
        return to_phys

    if A.load:
        #### RESAMPLE-ONLY: rebuild the generator from a saved checkpoint, skip data-load + training.
        ck = torch.load(A.load, map_location=dev, weights_only=False)
        sa = ck['args']
        mean, std, act_cap = ck['mean'], ck['std'], ck['act_cap']
        c = sa.get('pe_asinh', 0.0); H = ck['H']
        class_cond = sa.get('class_cond', False)
        n_cls = N_CLASSES if class_cond else 0
        data_frac = ck.get('offpath_frac', 0.1)
        G = Generator(FD, H, sa['latent_dim'], ch=sa['ch'], depth=sa['depth'],
                      dilated=sa.get('dilated', False), n_classes=n_cls).to(dev)
        G.load_state_dict(ck['ema']); G.eval()
        to_phys = make_to_phys(mean, std, c); X = None
        os.makedirs(f'{A.out}/shape_dataset', exist_ok=True)
        print(f"[load] {A.load}: ch={sa['ch']} depth={sa['depth']} latent={sa['latent_dim']} "
              f"asinh={c} act_cap={act_cap:.3f} class_cond={class_cond} -> resample {A.n_gen}")
        return _generate(A, dev, G, to_phys, act_cap, H, X, sa['latent_dim'], class_cond, data_frac)

    X = load_windows(A.csv, H, A.stride, A.limit)                 # (N, H, 19) PHYSICAL units
    print(f"[data] {len(X)} windows of ({H}, {FD}) (device {dev})")
    c = A.pe_asinh
    Xtr = X.copy()
    if c > 0:
        Xtr[..., PE_COLS] = np.arcsinh(Xtr[..., PE_COLS] / c)
        print(f"[norm] pos_err asinh(x/{c}) transform ON")
    mean = Xtr.reshape(-1, FD).mean(0); std = Xtr.reshape(-1, FD).std(0) + 1e-6
    Xt = torch.tensor((Xtr - mean) / std, dtype=torch.float32)
    mean_t = torch.tensor(mean, device=dev); std_t = torch.tensor(std, device=dev)
    dt = 1.0 / A.control_freq
    PE = slice(0, 3); VEL = slice(7, 10)
    act_cap = float(np.linalg.norm(X.reshape(-1, FD)[:, SD:SD + AD], axis=1).max()) if A.act_cap <= 0 else A.act_cap
    print(f"[clamp] action magnitude capped at real max = {act_cap:.3f} m/s")

    #### per-window on/off-path label + optional class-balanced sampling (identical to diffusion).
    pe_win_max = np.linalg.norm(X[..., 0:3], axis=2).max(axis=1)
    labels = np.where(pe_win_max > A.pe_offpath_thr, CLS_OFF, CLS_ON).astype(np.int64)
    data_frac = float((labels == CLS_OFF).mean())
    labels_t = torch.tensor(labels)
    samp_w = None
    n_on, n_off = int((labels == CLS_ON).sum()), int((labels == CLS_OFF).sum())
    if A.class_cond and A.offpath_batch_frac >= 0 and 0 < n_off < len(labels):
        f = A.offpath_batch_frac
        w = np.where(labels == CLS_OFF, f / n_off, (1 - f) / max(1, n_on)).astype(np.float64)
        samp_w = torch.tensor(w / w.sum())
    if A.class_cond:
        bal = f"{A.offpath_batch_frac:.2f} (oversampled)" if samp_w is not None else "natural (no oversample)"
        print(f"[class] threshold |pos_err|>{A.pe_offpath_thr}m: {data_frac*100:.1f}% of WINDOWS off-path "
              f"-> on-path={n_on} off-path={n_off}; batch off-path frac = {bal}")

    n_cls = N_CLASSES if A.class_cond else 0
    use_sn = (A.d_reg == 'sn')
    G = Generator(FD, H, A.latent_dim, ch=A.ch, depth=A.depth, dilated=A.dilated, n_classes=n_cls).to(dev)
    D = Discriminator(FD, ch=A.ch, depth=A.depth, dilated=A.dilated, n_classes=n_cls, sn=use_sn).to(dev)
    npar = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    reg = 'spectral-norm+hinge' if use_sn else f'wgan-gp(gp={A.gp_weight})'
    print(f"[model] G={npar(G):.2f}M D={npar(D):.2f}M ch={A.ch} depth={A.depth} latent={A.latent_dim} "
          f"class_cond={A.class_cond} batch={A.batch} n_critic={A.n_critic} g_steps={A.g_steps} "
          f"lr_g={A.lr_g:g} lr_d={A.lr_d:g} d_reg={reg}")
    optG = torch.optim.Adam(G.parameters(), A.lr_g, betas=(A.beta1, A.beta2))
    optD = torch.optim.Adam(D.parameters(), A.lr_d, betas=(A.beta1, A.beta2))
    ema = {k: v.detach().clone() for k, v in G.state_dict().items()}

    def draw(n):                                                  # real batch (+ labels) with optional balancing
        idx = (torch.multinomial(samp_w, n, replacement=True) if samp_w is not None
               else torch.randint(0, len(Xt), (n,)))
        cb = labels_t[idx].to(dev) if A.class_cond else None
        return Xt[idx].to(dev), cb

    def gp_of(real, fake, cb):                                    # WGAN gradient penalty
        eps = torch.rand(real.shape[0], 1, 1, device=dev)
        xi = (eps * real + (1 - eps) * fake).requires_grad_(True)
        d = D(xi, cb)
        g = torch.autograd.grad(d.sum(), xi, create_graph=True)[0]
        return ((g.reshape(g.shape[0], -1).norm(2, dim=1) - 1) ** 2).mean()

    def cons_loss(gnorm):                                         # pos_err<->vel physical consistency on generated x
        x = gnorm.clamp(-3.0, 3.0)
        phys = x * std_t + mean_t
        pe, vel = phys[..., PE], phys[..., VEL]
        if c > 0:
            pe = c * torch.sinh(pe)
        d_tgt = (pe[:, 1:] - pe[:, :-1]) + vel[:, :-1] * dt
        acc_tgt = d_tgt[:, 1:] - d_tgt[:, :-1]
        return (acc_tgt ** 2).mean()

    dwin = 0.5; n_dskip = 0                                       # tracked D win-rate + skip counter
    lossD = torch.tensor(0.0, device=dev)
    for step in range(A.steps):
        #### --- n_critic discriminator updates (dynamically skipped if D already dominates) ---
        skip_d = A.dynamic_d and dwin > A.d_win_thr
        if skip_d:
            n_dskip += 1
            with torch.no_grad():                                # still measure win-rate so dwin decays -> D resumes
                real, cb = draw(A.batch)
                fake = G(torch.randn(A.batch, A.latent_dim, device=dev), cb)
                win = 0.5 * ((D(real, cb) > 0).float().mean() + (D(fake, cb) < 0).float().mean())
                dwin = A.d_win_ema * dwin + (1 - A.d_win_ema) * win.item()
        else:
            for _ in range(A.n_critic):
                real, cb = draw(A.batch)
                z = torch.randn(A.batch, A.latent_dim, device=dev)
                fake = G(z, cb).detach()
                dr, df = D(real, cb), D(fake, cb)
                win = 0.5 * ((dr > 0).float().mean() + (df < 0).float().mean())   # D classification accuracy
                dwin = A.d_win_ema * dwin + (1 - A.d_win_ema) * win.item()
                if use_sn:                                        # spectral-norm path: hinge loss, no GP
                    lossD = F.relu(1.0 - dr).mean() + F.relu(1.0 + df).mean()
                else:                                             # WGAN-GP path
                    lossD = df.mean() - dr.mean() + A.gp_weight * gp_of(real, fake, cb)
                optD.zero_grad(); lossD.backward(); optD.step()
        #### --- g_steps generator updates (>1 rebalances when D over-powers G) ---
        for _ in range(A.g_steps):
            _, cb = draw(A.batch)
            z = torch.randn(A.batch, A.latent_dim, device=dev)
            fake = G(z, cb)
            lossG = -D(fake, cb).mean()
            lc = cons_loss(fake) if A.lambda_cons > 0 else torch.tensor(0.0, device=dev)
            optG.zero_grad(); (lossG + A.lambda_cons * lc).backward(); optG.step()
        with torch.no_grad():
            for k, v in G.state_dict().items():
                ema[k].mul_(A.ema).add_(v.detach(), alpha=1 - A.ema)
        if step % 2000 == 0:
            dtag = f" d_win {dwin:.2f} skip {n_dskip}" if A.dynamic_d else ""
            print(f"[train] step {step:>6} lossD {lossD.item():+.4f} lossG {lossG.item():+.4f} "
                  f"cons {lc.item():.4f}{dtag}", flush=True)

    G.load_state_dict(ema); G.eval()                              # sample from EMA weights
    os.makedirs(f'{A.out}/shape_dataset', exist_ok=True)
    torch.save({'ema': ema, 'mean': mean, 'std': std, 'act_cap': act_cap,
                'args': vars(A), 'FD': FD, 'H': H, 'offpath_frac': data_frac}, f'{A.out}/model.pt')
    to_phys = make_to_phys(mean, std, c)
    return _generate(A, dev, G, to_phys, act_cap, H, X, A.latent_dim, A.class_cond, data_frac)


def _generate(A, dev, G, to_phys, act_cap, H, X, latent, class_cond=False, data_frac=0.1):
    """Sample windows and write one per-episode CSV each (identical output to the diffusion version)."""
    @torch.no_grad()
    def sample(n, cls=None):
        z = torch.randn(n, latent, device=dev)
        cb = torch.full((n,), cls, device=dev, dtype=torch.long) if (class_cond and cls is not None) else None
        #### clamp the normalized output to a generous range so a spiky G can't blow the asinh sinh
        #### inverse up to inf/nan (real asinh-normalized data sits well within +-8); safety net only.
        return G(z, cb).clamp(-8.0, 8.0).cpu().numpy()

    def pe_jerk(Gphys):
        pe = Gphys[:, :, 0:3]
        return float(np.linalg.norm(pe[:, 2:] - 2 * pe[:, 1:-1] + pe[:, :-2], axis=2).mean())
    jcls = CLS_ON if class_cond else None
    print(f"[sampler] pe_jerk = {pe_jerk(to_phys(sample(64, jcls))):.5f}  (real≈0.00135)", flush=True)

    if class_cond:
        frac = data_frac if A.gen_offpath_frac < 0 else A.gen_offpath_frac
        n_off = int(round(A.n_gen * frac)); n_on = A.n_gen - n_off
        plan = [(CLS_ON, n_on), (CLS_OFF, n_off)]
        print(f"[gen] class-conditional: {n_on} on-path + {n_off} off-path (frac {frac:.3f})", flush=True)
    else:
        plan = [(None, A.n_gen)]

    HDR = ['step'] + STATE + ACT + ['reward', 'done']
    made = 0; allG = []
    for cls, count in plan:
        got = 0
        while got < count:
            k = min(A.gen_batch, count - got)
            Gp = to_phys(sample(k, cls))                          # (k, H, 19) physical
            for w in range(k):
                traj = Gp[w]
                q = traj[:, QUAT]; traj[:, QUAT] = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
                s = traj[:, :SD]; act = traj[:, SD:SD + AD]
                an = np.linalg.norm(act, axis=1, keepdims=True)
                act = act * np.minimum(1.0, act_cap / (an + 1e-8))
                reward = -np.linalg.norm(s[:, 0:3], axis=1)
                fn = f'{A.out}/shape_dataset/gentraj-{made:05d}.csv'
                with open(fn, 'w', newline='') as f:
                    wr = csv.writer(f); wr.writerow(HDR)
                    for i in range(H):
                        wr.writerow([i, *s[i], *act[i], reward[i], i == H - 1])
                made += 1
            if len(allG) < 20:
                allG.append(Gp)
            got += k
            print(f"[gen] wrote {made}/{A.n_gen} trajectory CSVs", flush=True)
    print(f"[done] {made} trajectory episodes (H={H}) -> {A.out}/shape_dataset/ (one CSV each)")

    #### ---- quality report (generated vs real if available) ----
    Gq = np.concatenate(allG, 0).reshape(-1, FD)
    gn = np.linalg.norm(Gq[:, SD:SD + AD], axis=1, keepdims=True)
    Gq[:, SD:SD + AD] *= np.minimum(1.0, act_cap / (gn + 1e-8))
    def blk(a, sl): return np.linalg.norm(a[:, sl], axis=1)
    def line(tag, a):
        pe, ve, ac = blk(a, slice(0, 3)), blk(a, slice(7, 10)), blk(a, slice(16, 19))
        print(f"  {tag:4} pos_err|mean|={pe.mean():.3f} (med {np.median(pe):.4f} max {pe.max():.2f})  "
              f"vel={ve.mean():.2f}  act|mean|={ac.mean():.2f} (max {ac.max():.2f})  quat_norm={blk(a, QUAT).mean():.3f}")
    print("[quality] generated vs real (pos_err/vel/action magnitudes, quat norm):")
    line('GEN', Gq)
    if X is not None:
        line('REAL', X.reshape(-1, FD))


if __name__ == '__main__':
    main()
