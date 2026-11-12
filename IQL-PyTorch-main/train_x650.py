"""X650 IQL ensemble, optionally aggregating explicit simulator DAgger transitions."""
import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from src import util
from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy
from src.value_functions import TwinQ, ValueFunction
from train_x500 import validate


ACTION_CENTER = np.array([0, 0, 0, .46], dtype=np.float32)
ACTION_SCALE = np.array([1.5, 1.5, 1.5, .34], dtype=np.float32)


def load_x650(path):
    with zipfile.ZipFile(path) as z:
        files = [n for n in z.namelist() if n.endswith('.npz')]
        assert len(files) == 1
        return dict(np.load(io.BytesIO(z.read(files[0])), allow_pickle=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', type=Path, default=Path('../x650_real_data.zip'))
    ap.add_argument('--out', type=Path, default=Path('runs_x650/ensemble128_20k'))
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--extra-npz',type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    util.DEFAULT_DEVICE = torch.device('cpu')
    util.USE_LAYERNORM = False
    d = load_x650(args.zip)
    mask = d['episode_id'] != 2
    original_train_rows=int(mask.sum())
    if args.extra_npz:
        extra=dict(np.load(args.extra_npz,allow_pickle=False))
        n=len(extra['observations'])
        for k in ('observations','actions','next_observations','rewards','terminals'):
            d[k]=np.concatenate([d[k],extra[k]])
        mask=np.r_[mask,np.ones(n,dtype=bool)]
    else:
        n=0
    mean = d['observations'][mask].mean(0)
    std = d['observations'][mask].std(0)
    std = np.where(std < 1e-6, 1., std)
    raw = {k: d[k].copy() for k in ('observations','actions','next_observations','rewards','terminals')}
    for k in ('observations','next_observations'):
        raw[k] = (raw[k]-mean)/std
    raw['actions'] = (raw['actions']-ACTION_CENTER)/ACTION_SCALE
    assert np.abs(raw['actions']).max() <= 1.00001
    assert all(np.isfinite(v).all() for v in raw.values())
    train = {k:torch.from_numpy(v[mask]) for k,v in raw.items()}
    val = {k:torch.from_numpy(v[~mask]) for k,v in raw.items()}
    np.savez(args.out/'normalization.npz', mean=mean,std=std,
             action_center=ACTION_CENTER,action_scale=ACTION_SCALE,
             hover_thrust=np.median(d['hover_thrust'][d['episode_id']!=2]))
    cfg = dict(hidden_dim=128,n_hidden=2,layernorm=False,steps=args.steps,beta=3.,
               smoothness_coef=.05,discount=.99,target_alpha=.005,learning_rate=3e-4,
               batch_size=256,reward_clip=None,train_flights=[0,1],validation_flights=[2],
               train_rows=int(mask.sum()),validation_rows=int((~mask).sum()),
               zip_sha256=hashlib.sha256(args.zip.read_bytes()).hexdigest(),
               action_center=ACTION_CENTER.tolist(),action_scale=ACTION_SCALE.tolist(),
               phases='All airborne: Mission and RTL',device='cpu',torch_version=torch.__version__,
               checkpoints='5k/10k/15k/20k; final or validation-minimum selection recorded by evaluator',
               terminals='Use provided terminals only; timeouts bootstrap from provided next state',
               policy_schedule='CosineAnnealingLR over full training length',dagger=bool(args.extra_npz),
               original_train_rows=original_train_rows,extra_rows=n,
               extra_path=str(args.extra_npz) if args.extra_npz else None,
               extra_sha256=hashlib.sha256(args.extra_npz.read_bytes()).hexdigest() if args.extra_npz else None,
               initialization='From scratch; normalization fit on combined training data')
    (args.out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    reports=[]
    for seed in (0,1):
        for tau in (.85,.90,.95):
            util.set_seed(seed)
            name=f's{seed}_t{tau:.2f}'
            folder=args.out/name
            folder.mkdir()
            policy=GaussianPolicy(16,4,hidden_dim=128,n_hidden=2,max_action=1.)
            iql=ImplicitQLearning(TwinQ(16,4,128,2),ValueFunction(16,128,2),policy,
                                 lambda p:torch.optim.Adam(p,lr=3e-4),max_steps=args.steps,
                                 tau=tau,beta=3.,discount=.99,alpha=.005,smoothness_coef=.05)
            start=time.monotonic()
            initial=validate(policy,val)
            hist=[]
            for step in range(1,args.steps+1):
                losses=iql.update(**util.sample_batch(train,256))
                if not all(np.isfinite(v) for v in losses.values()):
                    raise RuntimeError(f'{name} step{step}: {losses}')
                if step%5000==0 or step==args.steps:
                    metrics=validate(policy,val)
                    row=dict(step=step,**losses,validation=metrics)
                    hist.append(row)
                    torch.save(iql.state_dict(),folder/f'step_{step}.pt')
                    (folder/'history.json').write_text(json.dumps(hist,indent=2)+'\n')
                    print(json.dumps(dict(member=name,step=step,val_mse=metrics['action_mse'],elapsed=time.monotonic()-start)),flush=True)
            report=dict(member=name,seed=seed,tau=tau,initial_validation=initial,
                        final_validation=metrics,train_fit=validate(policy,train),
                        seconds=time.monotonic()-start,parameters=sum(p.numel() for p in policy.parameters()))
            (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            reports.append(report)
            (args.out/'training_report.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    main()
