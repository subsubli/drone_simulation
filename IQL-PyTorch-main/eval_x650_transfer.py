"""Frozen X650 policies in the existing X500 data-identified surrogate, not Gazebo."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import torch

import x500_sim as simlib
from src.policy import GaussianPolicy
from train_x650 import load_x650, ACTION_CENTER, ACTION_SCALE

C = np.diag([1.,-1.,-1.])  # NWU/FLU <-> NED/FRD, both proper rotations


def observation(sim,tgt,tvel):
    rot=C@sim.R@C
    q=Rotation.from_matrix(rot).as_quat()[[3,0,1,2]]
    if q[0]<0:
        q=-q
    return np.r_[C@(sim.p-tgt),C@(sim.v-tvel),C@sim.v,q,C@sim.w].astype(np.float32)


def action(raw,model,hover,mode='hover_ratio'):
    thrust = raw[3]/hover * (model['g']/model['kT']) if mode=='hover_ratio' else raw[3]*simlib.T_MAX
    return np.r_[2*thrust/simlib.T_MAX-1,(C@raw[:3])/simlib.RATE_MAX]


def check_adapter(d,model,hover):
    errors=[]
    for i in np.linspace(0,len(d['observations'])-1,100,dtype=int):
        raw=d['raw_observations'][i]
        r=Rotation.from_quat(raw[[7,8,9,6]]).as_matrix()
        obj=SimpleNamespace(p=C@raw[:3],v=C@raw[3:6],R=C@r@C,w=C@raw[10:13])
        errors.append(float(np.max(np.abs(observation(obj,C@d['goals'][i,:3],C@d['goals'][i,3:])-d['observations'][i]))))
    assert max(errors)<1e-5, max(errors)
    a=action(np.array([.1,.2,.3,hover]),model,hover)
    np.testing.assert_allclose(a[1:]*simlib.RATE_MAX,[.1,-.2,-.3],atol=1e-8)
    np.testing.assert_allclose((a[0]+1)/2*simlib.T_MAX,model['g']/model['kT'])
    return dict(observation_roundtrip_max_error=max(errors),rate_sign_and_hover_mapping='passed')


def actor(policies,mean,std,model,hover,mode):
    center=torch.from_numpy(ACTION_CENTER)
    scale=torch.from_numpy(ACTION_SCALE)
    m,sd=torch.from_numpy(mean),torch.from_numpy(std)
    cache=None
    calls=0
    stats=dict(steps=0,clipped_steps=0,max_abs_standardized_observation=0.)
    @torch.no_grad()
    def act(sim,tgt,tvel):
        nonlocal cache,calls
        if calls%2==0:
            x=(torch.from_numpy(observation(sim,tgt,tvel))-m)/sd
            stats['max_abs_standardized_observation']=max(stats['max_abs_standardized_observation'],float(x.abs().max()))
            outputs=[]
            for p in policies:
                # Exact deterministic mean of GaussianPolicy; avoids distribution construction.
                outputs.append(torch.tanh(p.net(x[None]))[0])
            raw=(torch.stack(outputs).mean(0)*scale+center).numpy()
            cache=action(raw,model,hover,mode)
            stats['clipped_steps']+=int(np.any(np.abs(cache)>1))
            stats['steps']+=1
        calls+=1
        return np.clip(cache,-1,1)
    return act,stats


def aggregate(rows):
    ok=[r for r in rows if not r['crashed']]
    return dict(episodes=len(rows),crashes=len(rows)-len(ok),mean_seconds=float(np.mean([r['t'] for r in rows])),
                survivor_mean_error=float(np.mean([r['err'] for r in ok])) if ok else None,
                all_episode_mean_error=float(np.mean([r['err'] for r in rows])),
                max_error=float(max(r['max'] for r in rows)),
                mean_net_laps=float(np.mean([r['net_laps'] for r in rows])),
                mean_target_laps_budget=float(np.mean([r['target_laps_budget'] for r in rows])),
                mean_progress_ratio_budget=float(np.mean([r['progress_ratio_budget'] for r in rows])),
                projection_rejections=sum(r['progress_projection_rejections'] for r in rows))


def heldout_r2(model,t,o,a,no,val):
    ok,rate,(xv,yv)=simlib.features(t,o,a,no)
    ok &= val
    def r2(y,p):
        return float(1-((y-p)**2).sum()/((y-y.mean())**2).sum())
    scores=[r2(y[ok],np.sum(x[ok]*b,axis=1)) for (x,y),b in zip(rate,model['rate'])]
    th=np.r_[model['kT'],model['g'],model['drag']]
    return scores+[r2(yv[ok].ravel(),np.sum(xv[ok]*th,axis=2).ravel())]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs',type=Path,default=Path('runs_x650/ensemble128_20k'))
    ap.add_argument('--zip',type=Path,default=Path('../x650_real_data.zip'))
    ap.add_argument('--csv',type=Path,default=Path('../datasets/data_track.csv.gz'))
    ap.add_argument('--episodes',type=int,default=50)
    ap.add_argument('--seconds',type=float,default=20.)
    ap.add_argument('--checkpoint',choices=['final','best-validation'],default='final')
    ap.add_argument('--ensemble-only',action='store_true')
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    output=args.out or args.runs/('x500_transfer_eval_best_validation.json' if args.checkpoint=='best-validation' else 'x500_transfer_eval.json')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}; specify --out')
    torch.set_num_threads(1)
    cfg=json.loads((args.runs/'config.json').read_text())
    norm=dict(np.load(args.runs/'normalization.npz'))
    t,o,a,no=simlib.load(args.csv)
    # Match the historical X500 training-flight split, not the X650 split.
    val=np.isin(t['flight'],[2,3,4,18,20,34,35])
    model=simlib.fit(t,o,a,no,~val,drag=False)
    d=load_x650(args.zip)
    hover=float(norm['hover_thrust'])
    checks=check_adapter(d,model,hover)
    policies=[]
    selection=[]
    for seed in (0,1):
        for tau in (.85,.90,.95):
            name=f's{seed}_t{tau:.2f}'
            p=GaussianPolicy(16,4,128,2,max_action=1.)
            history=json.loads((args.runs/name/'history.json').read_text())
            chosen=min(history,key=lambda r:(r['validation']['action_mse'],r['step'])) if args.checkpoint=='best-validation' else next(r for r in history if r['step']==cfg['steps'])
            checkpoint=args.runs/name/f'step_{chosen["step"]}.pt'
            selection.append(dict(member=name,step=chosen['step'],validation_mse=chosen['validation']['action_mse'],path=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
            st=torch.load(checkpoint,map_location='cpu',weights_only=True)
            p.load_state_dict({k.removeprefix('policy.'):v for k,v in st.items() if k.startswith('policy.')})
            policies.append((name,p.eval()))
    # Secondary heading condition: first recorded attitude of the two training flights.
    yaws=[]
    for e in (0,1):
        q=d['observations'][np.flatnonzero(d['episode_id']==e)[0],9:13]
        r=C@Rotation.from_quat(q[[1,2,3,0]]).as_matrix()@C
        yaws.append(np.arctan2(r[1,0],r[0,0]))
    matched_yaw=float(np.arctan2(np.sin(yaws).mean(),np.cos(yaws).mean()))
    variants=[('PD',None,'hover_ratio',np.deg2rad(-94.))]
    variants += [(n,[p],'hover_ratio',np.deg2rad(-94.)) for n,p in policies]
    variants += [('ensemble6',[p for _,p in policies],'hover_ratio',np.deg2rad(-94.)),
                 ('ensemble6_training_heading',[p for _,p in policies],'hover_ratio',matched_yaw),
                 ('ensemble6_direct_thrust',[p for _,p in policies],'direct',np.deg2rad(-94.))]
    if args.ensemble_only:
        variants=[v for v in variants if v[0].startswith('ensemble6')]
    report=dict(protocol='Frozen policies using X650 observation/action contract; X500 identified surrogate, no Gazebo/hardware or DAgger during evaluation',
                training_dagger=cfg.get('dagger',False),training_extra_rows=cfg.get('extra_rows',0),
                training_config_sha256=hashlib.sha256((args.runs/'config.json').read_bytes()).hexdigest(),
                simulator_csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                simulator_model={k:np.asarray(v).tolist() for k,v in model.items()},
                heldout_r2=heldout_r2(model,t,o,a,no,val),adapter_checks=checks,
                x650_hover=hover,matched_yaw_deg=float(np.rad2deg(matched_yaw)),
                thrust_mapping='Main: T=(u/u_hover_X650)*(g/kT)_X500; sensitivity: T=u*T_MAX_X500. Both are interface assumptions, not identified X650 thrust curves.',
                controller_hz=50,simulator_hz=100,seed_base=1000,seconds=args.seconds,
                action_average='Equal average of deterministic normalized policy means, then common affine unscale',
                checkpoint=args.checkpoint,selection=selection,
                selection_scope='Minimum heldout flight2 action MSE among saved checkpoints; flight2 is validation, not independent test' if args.checkpoint=='best-validation' else 'Fixed final checkpoint',results={})
    for name,ps,mode,yaw in variants:
        simlib.YAW_DATA=yaw
        rows=[]
        clipping=0
        commands=0
        maxz=0.
        for seed in range(1000,1000+args.episodes):
            if ps is None:
                fn=lambda sim,tgt,tvel:simlib.pd_expert(model,sim,tgt,tvel)
                stats=None
            else:
                fn,stats=actor(ps,norm['mean'],norm['std'],model,hover,mode)
            rows.append(simlib.rollout(model,fn,seed,args.seconds))
            if stats:
                clipping+=stats['clipped_steps'];commands+=stats['steps']
                maxz=max(maxz,stats['max_abs_standardized_observation'])
        result=dict(summary=aggregate(rows),episodes=rows,yaw_deg=float(np.rad2deg(yaw)),
                    thrust_mode=mode,clipped_commands=clipping,commands=commands,max_abs_standardized_observation=maxz)
        report['results'][name]=result
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(name=name,**result['summary'])),flush=True)


if __name__=='__main__':
    main()
