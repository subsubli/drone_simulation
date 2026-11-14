"""Re-evaluate saved X500 policies with geometric net laps; no training or selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import x500_sim as sim
from eval_x650_transfer import aggregate, heldout_r2


def main():
    torch.set_num_threads(1)
    root=Path('runs_x500')
    output=root/'net_laps_20261115.json'
    if output.exists():
        raise FileExistsError(output)
    def one(pattern):
        paths=list(root.glob(pattern))
        assert len(paths)==1,(pattern,paths)
        return paths[0].parent
    base=one('data_track/*/final.pt')
    ensemble=[base]+[one(f'ens/{member}/data_track/*/final.pt') for member in
                         ('s1_t0.85','s0_t0.90','s1_t0.90','s0_t0.95','s1_t0.95')]
    groups={'base':[base],'ensemble6':ensemble}
    groups.update({f'DAgger{i}':[one(f'dagg{i}/data_track/*/final.pt')] for i in (1,2,3)})
    t,o,a,no=sim.load(Path('../datasets/data_track.csv.gz'))
    val=np.isin(t['flight'],[2,3,4,18,20,34,35])
    model=sim.fit(t,o,a,no,~val,drag=False)
    report=dict(protocol='Existing X500 identified surrogate, seeds1000..1049,20s,yaw-94,100Hz; saved weights unchanged',
                metric='Signed unwrapped local polyline projection; 2048 segments; reference-forward direction positive; no elapsed-time substitution',
                heldout_r2=heldout_r2(model,t,o,a,no,val),results={})
    for name,runs in [('PD',None)]+list(groups.items()):
        actor=(lambda s,p,v:sim.pd_expert(model,s,p,v)) if runs is None else sim.load_policy(runs)
        rows=[sim.rollout(model,actor,seed,20.) for seed in range(1000,1050)]
        report['results'][name]=dict(summary=aggregate(rows),episodes=rows,
             by_shape={shape:aggregate([r for r in rows if r['shape']==shape]) for shape in ('circle','eight')},
             checkpoints={} if runs is None else {str(r/'final.pt'):hashlib.sha256((r/'final.pt').read_bytes()).hexdigest() for r in runs})
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(name,json.dumps(report['results'][name]['summary']),flush=True)


if __name__=='__main__':main()
