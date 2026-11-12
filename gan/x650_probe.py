"""Small spectral-normalized hinge GAN on X650 joint transitions."""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch import nn


def network(n_in, n_out, width, spectral=False):
    layers = []
    for i, o in zip([n_in, width, width, width], [width, width, width, n_out]):
        layer = nn.Linear(i, o)
        layers.append(nn.utils.spectral_norm(layer) if spectral else layer)
        layers.append(nn.LeakyReLU(.2))
    return nn.Sequential(*layers[:-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default='x650_real_data.zip')
    ap.add_argument('--out', default='gan/x650_probe_run')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--samples', type=int, default=2048)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    with zipfile.ZipFile(args.zip) as z:
        names = [n for n in z.namelist() if n.endswith('.npz')]
        assert len(names) == 1
        d = dict(np.load(io.BytesIO(z.read(names[0])), allow_pickle=False))
    s, a, sn = (d[k] for k in ('observations', 'actions', 'next_observations'))
    x = np.concatenate([s, a, sn-s], 1)
    assert x.shape[1] == 36 and np.isfinite(x).all()
    train = d['episode_id'] != 2
    mean, std = x[train].mean(0), np.maximum(x[train].std(0), 1e-4)
    xt = torch.from_numpy((x[train]-mean)/std)
    gen, disc = network(32,36,args.hidden), network(36,1,args.hidden,True)
    go = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.,.9))
    do = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.,.9))
    history = []
    start = time.monotonic()
    for step in range(1,args.steps+1):
        real = xt[torch.randint(len(xt),(128,))]
        fake = gen(torch.randn(128,32)).detach()
        dl = torch.relu(1-disc(real)).mean()+torch.relu(1+disc(fake)).mean()
        do.zero_grad(); dl.backward(); do.step()
        disc.requires_grad_(False)
        gl = -disc(gen(torch.randn(128,32))).mean()
        go.zero_grad(); gl.backward(); go.step()
        disc.requires_grad_(True)
        if step == 1 or step % 500 == 0 or step == args.steps:
            row = dict(step=step, discriminator_loss=float(dl.detach()), generator_loss=float(gl.detach()))
            history.append(row)
            print(json.dumps(row),flush=True)
    elapsed = time.monotonic()-start
    gen.eval(); disc.eval()
    with torch.no_grad():
        samples = gen(torch.randn(args.samples,32)).numpy()*std+mean
    gs, ga = samples[:,:16], samples[:,16:20]
    gsn = gs+samples[:,20:]
    reward = -(np.sum(gsn[:,:3]**2,1)+.1*np.sum(gsn[:,3:6]**2,1)+.01*np.sum(ga[:,:3]**2,1))
    np.savez_compressed(out/'raw_generated.npz', observations=gs, actions=ga,
                        next_observations=gsn, rewards=reward)
    torch.save(dict(generator=gen.state_dict(), discriminator=disc.state_dict(),
                    mean=mean,std=std,args=vars(args),representation='s16,a4,delta_s16'),out/'model.pt')

    def stats(v):
        state, act, delta = v[:,:16], v[:,16:20], v[:,20:]
        return dict(position_error_quantiles=np.quantile(np.linalg.norm(state[:,:3],axis=1),[.5,.95,1]).tolist(),
                    quaternion_norm_quantiles=np.quantile(np.linalg.norm(state[:,9:13],axis=1),[0,.5,1]).tolist(),
                    next_quaternion_norm_quantiles=np.quantile(np.linalg.norm((state+delta)[:,9:13],axis=1),[0,.5,1]).tolist(),
                    action_min=act.min(0).tolist(),action_max=act.max(0).tolist(),
                    delta_std=delta.std(0).tolist(),finite=bool(np.isfinite(v).all()),
                    thrust_outside_0_1_fraction=float(((act[:,3]<0)|(act[:,3]>1)).mean()))

    def nearest(v):
        vals = []
        for block in np.array_split((v-mean)/std,max(1,len(v)//64)):
            ds = torch.cdist(torch.from_numpy(block),xt,compute_mode='donot_use_mm_for_euclid_dist')
            vals.extend((ds.min(1).values/6).tolist())
        return np.quantile(vals,[.05,.5,.95]).tolist()

    report = dict(config=vars(args), zip_sha256=hashlib.sha256(Path(args.zip).read_bytes()).hexdigest(),
                  generator_parameters=sum(p.numel() for p in gen.parameters()),
                  discriminator_parameters=sum(p.numel() for p in disc.parameters()),
                  training_seconds=elapsed,train_rows=int(train.sum()),heldout_rows=int((~train).sum()),
                  algorithm='Spectral-normalized discriminator, hinge loss, one D and G update per step, latent32, batch128, Adam2e-4 betas(0,.9)',
                  scope='All airborne phases including Mission and RTL. No clipping, quaternion projection, policy learning or flight evaluation.',
                  checkpoint='Fixed final step, not selected on heldout flight.', history=history,
                  train=stats(x[train]),heldout=stats(x[~train]),generated=stats(samples),
                  nearest_train_rms_standardized=dict(generated=nearest(samples),heldout=nearest(x[~train])))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(training_seconds=elapsed,generated=report['generated'])),flush=True)


if __name__ == '__main__':
    main()
