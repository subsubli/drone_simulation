"""Small X650 transition DDPM probe; generated samples are not flight-validated."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from transition_diffusion import Denoiser


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zip', default='x650_real_data.zip')
    p.add_argument('--out', default='diffusion/x650_probe_run')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--samples', type=int, default=2048)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with zipfile.ZipFile(args.zip) as z:
        names = [n for n in z.namelist() if n.endswith('.npz')]
        assert len(names) == 1
        d = dict(np.load(io.BytesIO(z.read(names[0])), allow_pickle=False))
    s, a, sn = (d[k] for k in ('observations', 'actions', 'next_observations'))
    # Model small changes separately to avoid losing the 20 ms transition scale.
    x = np.concatenate([s, a, sn - s], axis=1)
    assert x.shape[1] == 36 and np.isfinite(x).all()
    train = d['episode_id'] != 2
    mean, std = x[train].mean(0), np.maximum(x[train].std(0), 1e-4)
    xt = torch.from_numpy((x[train] - mean) / std)
    xv = torch.from_numpy((x[~train] - mean) / std)
    model = Denoiser(36, h=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    T = 1000
    beta = torch.linspace(1e-4, .02, T)
    abar = torch.cumprod(1 - beta, 0)
    # Fixed evaluation noise enables checkpoint comparisons, not a control metric.
    rng = torch.Generator().manual_seed(1024)
    ev = []
    for data in (xt, xv):
        ix = torch.randint(len(data), (2048,), generator=rng)
        t = torch.randint(T, (2048,), generator=rng)
        eps = torch.randn(2048, 36, generator=rng)
        ab = abar[t, None]
        ev.append((ab.sqrt() * data[ix] + (1-ab).sqrt()*eps, t, eps))
    history = []
    for step in range(1, args.steps + 1):
        x0 = xt[torch.randint(len(xt), (128,))]
        t = torch.randint(T, (128,))
        eps = torch.randn_like(x0)
        ab = abar[t, None]
        loss = (model(ab.sqrt()*x0 + (1-ab).sqrt()*eps, t)-eps).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 500 == 0 or step == args.steps:
            with torch.no_grad():
                scores = [float((model(y,t)-e).square().mean()) for y,t,e in ev]
            row = dict(step=step, train_noise_mse=scores[0], heldout_noise_mse=scores[1])
            history.append(row)
            print(json.dumps(row), flush=True)
    torch.save(dict(model=model.state_dict(), mean=mean, std=std,
                    args=vars(args), representation='s16,a4,delta_s16'), out/'model.pt')
    chunks = []
    with torch.no_grad():
        for start in range(0, args.samples, 256):
            y = torch.randn(min(256, args.samples-start), 36)
            for ti in reversed(range(T)):
                t = torch.full((len(y),), ti)
                y = (y-beta[ti]/(1-abar[ti]).sqrt()*model(y,t))/(1-beta[ti]).sqrt()
                if ti:
                    variance = beta[ti]*(1-abar[ti-1])/(1-abar[ti])
                    y += variance.sqrt()*torch.randn_like(y)
            chunks.append(y.numpy()*std+mean)
    gen = np.concatenate(chunks)
    # Keep raw outputs: normalization/clipping must not hide generation failures.
    gs, ga = gen[:, :16], gen[:, 16:20]
    gsn = gs + gen[:, 20:]
    reward = -(np.sum(gsn[:,:3]**2,1)+.1*np.sum(gsn[:,3:6]**2,1)+.01*np.sum(ga[:,:3]**2,1))
    np.savez_compressed(out/'raw_generated.npz', observations=gs, actions=ga,
                        next_observations=gsn, rewards=reward)

    def stats(v):
        state, act, delta = v[:,:16], v[:,16:20], v[:,20:]
        return dict(position_error_quantiles=np.quantile(np.linalg.norm(state[:,:3],axis=1),[.5,.95,1]).tolist(),
                    quaternion_norm_quantiles=np.quantile(np.linalg.norm(state[:,9:13],axis=1),[0,.5,1]).tolist(),
                    next_quaternion_norm_quantiles=np.quantile(np.linalg.norm((state+delta)[:,9:13],axis=1),[0,.5,1]).tolist(),
                    action_min=act.min(0).tolist(), action_max=act.max(0).tolist(),
                    delta_std=delta.std(0).tolist(), finite=bool(np.isfinite(v).all()))

    # Nearest training-row distance in standardized joint transition space.
    def nearest(v):
        distances = []
        for block in np.array_split((v-mean)/std, max(1,len(v)//64)):
            ds = torch.cdist(torch.from_numpy(block), xt,
                             compute_mode='donot_use_mm_for_euclid_dist')
            distances.extend((ds.min(1).values / 6).tolist())
        return np.quantile(distances,[.05,.5,.95]).tolist()

    report = dict(config=vars(args), zip_sha256=hashlib.sha256(Path(args.zip).read_bytes()).hexdigest(),
                  parameters=sum(p.numel() for p in model.parameters()),
                  train_rows=int(train.sum()), heldout_rows=int((~train).sum()),
                  scope='All airborne phases including Mission and RTL; no policy training or flight evaluation.',
                  checkpoint='Fixed final step, not selected on heldout flight.', history=history,
                  train=stats(x[train]), heldout=stats(x[~train]), generated=stats(gen),
                  nearest_train_rms_standardized=dict(generated=nearest(gen),heldout=nearest(x[~train])))
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('parameters','train_rows','heldout_rows','nearest_train_rms_standardized')}), flush=True)


if __name__ == '__main__':
    main()
