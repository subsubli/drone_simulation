"""One 10-episode X500-surrogate DAgger pass in the X650 observation/action contract."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.policy import GaussianPolicy
import x500_sim as simlib
from eval_x650_transfer import observation,action,actor,C,check_adapter
from train_x650 import ACTION_CENTER,ACTION_SCALE,load_x650


def main():
    torch.set_num_threads(1)
    runs=Path('runs_x650/ensemble128_20k')
    out=Path('runs_x650/dagger10_data')
    out.mkdir(parents=True,exist_ok=False)
    norm=dict(np.load(runs/'normalization.npz'))
    hover=float(norm['hover_thrust'])
    src=json.loads((runs/'x500_transfer_eval_best_validation.json').read_text())
    model={k:np.array(v) for k,v in src['simulator_model'].items()}
    check_adapter(load_x650(Path('../x650_real_data.zip')),model,hover)
    policies=[]
    for c in src['selection']:
        p=GaussianPolicy(16,4,128,2,max_action=1.)
        st=torch.load(c['path'],weights_only=True,map_location='cpu')
        p.load_state_dict({k.removeprefix('policy.'):v for k,v in st.items() if k.startswith('policy.')})
        policies.append(p.eval())
    data={k:[] for k in ('observations','actions','next_observations','rewards','terminals','timeouts','episode_id','timestamps')}
    episodes=[]; clipped=0;max_roundtrip=0.
    simlib.YAW_DATA=np.deg2rad(-94.);simlib.YAW0=simlib.YAW_DATA
    for eid,seed in enumerate(range(2000,2010)):
        at,shape,_=simlib.trajectory(np.random.default_rng(seed))
        sim=simlib.X500Sim(model,*at(0.))
        driver,_=actor(policies,norm['mean'],norm['std'],model,hover,'hover_ratio')
        errs=[]
        for k in range(1000):
            tgt,tvel=at(k*.02)
            obs=observation(sim,tgt,tvel)
            expert=np.clip(simlib.pd_expert(model,sim,tgt,tvel),-1,1)
            thrust=(expert[0]+1)/2*simlib.T_MAX
            raw=np.r_[C@(expert[1:]*simlib.RATE_MAX),thrust/(model['g']/model['kT'])*hover]
            label=np.clip(raw,ACTION_CENTER-ACTION_SCALE,ACTION_CENTER+ACTION_SCALE)
            clipped+=int(np.any(np.abs(label-raw)>1e-7))
            applied=action(label,model,hover)
            assert np.abs(applied).max()<=1.00001
            inverse=np.r_[C@(applied[1:]*simlib.RATE_MAX),(applied[0]+1)/2*simlib.T_MAX/(model['g']/model['kT'])*hover]
            max_roundtrip=max(max_roundtrip,float(np.abs(inverse-label).max()))
            twin=copy.deepcopy(sim)
            terminal=False
            for _ in range(2):
                twin.step(applied)
                terminal |= bool(twin.crashed())
            nxt=observation(twin,*at((k+1)*.02))
            rew=-(nxt[:3]@nxt[:3]+.1*(nxt[3:6]@nxt[3:6])+.01*(label[:3]@label[:3]))
            dead=False
            for j in range(2):
                sim.step(driver(sim,*at(k*.02+j*.01)))
                dead |= bool(sim.crashed())
            errs.append(float(np.linalg.norm(at((k+1)*.02)[0]-sim.p)))
            values=(obs,label,nxt,rew,terminal,bool((dead or k==999) and not terminal),eid,k*.02)
            for key,value in zip(data,values):data[key].append(value)
            if dead:break
        episodes.append(dict(seed=seed,shape=shape,rows=k+1,seconds=(k+1)*.02,failed=dead,mean_error=float(np.mean(errs))))
    arr={k:np.asarray(v,dtype=np.int32 if k=='episode_id' else np.float32) for k,v in data.items()}
    assert all(np.isfinite(v).all() for v in arr.values())
    assert len(episodes)==10
    np.savez_compressed(out/'transitions.npz',**arr)
    report=dict(episodes=episodes,rows=len(arr['rewards']),clipped_expert_labels=clipped,
                label_terminal_count=int(arr['terminals'].sum()),timeout_count=int(arr['timeouts'].sum()),
                action_roundtrip_max_error=max_roundtrip,driver_selection=src['selection'],
                simulator_model=src['simulator_model'],source_eval_sha256=hashlib.sha256((runs/'x500_transfer_eval_best_validation.json').read_bytes()).hexdigest(),
                protocol='10 learner-driven episodes seeds2000..2009, max20s each; stop when learner fails. No resets or expert rescue within episode.',
                transitions='Clone learner state and history; apply bounded expert action for two 100Hz substeps. Next state and reward correspond to expert action, not learner action.',
                reward='Original X650 next-state tracking reward, no added crash penalty; bootstrap masked on expert-twin failure.',
                action_limits='X650 policy bounds rates +/-1.5rad/s, thrust 0.12 to 0.8; expert labels clipped BEFORE twin execution.',
                data_source='X500 identified surrogate, not real X650; no hardware or Gazebo execution')
    (out/'collection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
