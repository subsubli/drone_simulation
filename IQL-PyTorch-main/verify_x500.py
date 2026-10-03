import gzip, json, numpy as np, onnxruntime as ort
run='runs_x500/data_track/10-03-26_14.53.22_vtkg'
with gzip.open('../datasets/data_track.csv.gz','rt') as f:
    t=np.genfromtxt(f,delimiter=',',names=True,dtype=np.float32)
col=lambda p,n:np.column_stack([t[f'{p}_{i}'] for i in range(n)])
obs,act=col('obs',25),col('act',4)
sp=json.load(open(run+'/dataset_split.json'))
s=ort.InferenceSession(run+'/policy.onnx',providers=['CPUExecutionProvider'])
pred=s.run(['actions'],{'observations':obs})[0]
val=np.isin(t['flight'],sp['validation_flights']); tr=~val
tm=act[tr].mean(0)
def mse(m,p=pred): return float(((p[m]-act[m])**2).mean())
prev=obs[:,18:22]
print('rows',len(obs),'val',val.sum(),'train',tr.sum())
print('MSE val/train model',mse(val),mse(tr))
print('baseline train-mean act', float(((act[val]-tm)**2).mean()))
print('baseline copy prev action (obs[18:22])', float(((act[val]-prev[val])**2).mean()))
print('baseline model+prev avg', float(((act[val]-(pred[val]+prev[val])/2)**2).mean()))
print('RMSE/dim val',np.sqrt(((pred[val]-act[val])**2).mean(0)).round(4).tolist())
print('RMSE/dim copy-prev',np.sqrt(((prev[val]-act[val])**2).mean(0)).round(4).tolist())
print('act std val',act[val].std(0).round(4).tolist(),'pred std',pred[val].std(0).round(4).tolist())
print('corr per dim',[round(float(np.corrcoef(pred[val][:,i],act[val][:,i])[0,1]),3) for i in range(4)])
print('thrust act mean/pred mean',float(act[val][:,0].mean()),float(pred[val][:,0].mean()),'hover 0.185')
print('per val flight: n, model, mean-baseline, copy-prev, crash?')
for f in sp['validation_flights']:
    m=t['flight']==f
    print(f,int(m.sum()),round(mse(m),5),round(float(((act[m]-tm)**2).mean()),5),round(float(((act[m]-prev[m])**2).mean()),5),int(t['terminal'][m].sum()))
# near-crash rows (last 50 steps before terminal) and high-error rows
term=np.where(t['terminal']==1)[0]; near=np.zeros(len(t),bool)
for i in term: near[max(0,i-50):i+1]=True
nm=near&val
print('val rows within 50 steps of crash',int(nm.sum()),'mse',mse(nm) if nm.any() else None,'rest',mse(val&~near))
err=np.linalg.norm(obs[:,0:3],axis=1)
for lo,hi in [(0,.1),(.1,.3),(.3,1),(1,99)]:
    m=val&(err>=lo)&(err<hi); print(f'pos_err [{lo},{hi}) n={m.sum()} mse={mse(m):.5f} base={float(((act[m]-tm)**2).mean()):.5f}')
# bounds / saturation
print('pred max abs',float(np.abs(pred).max()),'finite',bool(np.isfinite(pred).all()))
# ONNX single vs batch
print('onnx batch1 vs full diff',float(np.abs(s.run(['actions'],{'observations':obs[val][:1]})[0]-pred[val][:1]).max()))
# determinism & input checks: raw obs
print('thrust N pred mean',float((pred[val][:,0].mean()+1)/2*34.19))
# smoothness of predicted actions vs logged (consecutive diff, same flight)
same=(t['flight'][1:]==t['flight'][:-1])&val[1:]
print('mean |Δa| logged',float(np.abs(np.diff(act,axis=0))[same].mean()),'pred',float(np.abs(np.diff(pred,axis=0))[same].mean()))
# noise floor estimate: medium segments add σ=0.15 noise -> irreducible ~ 0.45*0.0225
