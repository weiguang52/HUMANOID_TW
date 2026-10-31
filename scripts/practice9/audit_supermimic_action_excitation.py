"""Fit-only linear association screen; not nonlinear identifiability or causality."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir()
plan=dict(ridge_lambda=.001,state_std_floor=1e-8,svd_rank_relative_tolerance=1e-8,fit='All original fit traces, no validation regression fitting or hyperparameter selection',scope='Linear association and residual excitation only; not proof of nonlinear identifiability, physical action sensitivity or failure causality')
(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
rows=json.loads((a.collection/'results.json').read_text())['rows'];mapping=json.loads((a.collection/'control_mapping.json').read_text());records=[]
for row in rows:
 path=(a.collection/row['trace']).resolve();assert path.is_relative_to(a.collection.resolve())
 with np.load(path,allow_pickle=False) as z:s=z['state'].astype(np.float64);u=z['action'].astype(np.float64)
 assert s.shape==(len(u),73) and u.shape[1]==30 and np.isfinite(s).all() and np.isfinite(u).all()
 records.append(dict(row=row,state=s,action=u,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
fit=[x for x in records if x['row']['split']=='fit'];validation=[x for x in records if x['row']['split']=='validation'];assert len(fit)==len(validation)==8
assert {x['row']['trace'] for x in fit}.isdisjoint(x['row']['trace'] for x in validation)
s=np.concatenate([x['state'] for x in fit]);u=np.concatenate([x['action'] for x in fit]);sm=s.mean(0);ss=s.std(0);keep=ss>plan['state_std_floor'];um=u.mean(0);us=u.std(0);assert (us>0).all()
x=(s[:,keep]-sm[keep])/ss[keep];y=(u-um)/us
beta=np.linalg.solve(x.T@x/len(x)+plan['ridge_lambda']*np.eye(x.shape[1]),x.T@y/len(x))
assert np.isfinite(beta).all()
def summarize(parts):
 states=np.concatenate([r['state'] for r in parts]);actions=np.concatenate([r['action'] for r in parts]);pred=((states[:,keep]-sm[keep])/ss[keep])@beta*us+um;res=actions-pred
 baseline=np.mean((actions-um)**2,axis=0);mse=np.mean(res**2,axis=0)
 return dict(frames=len(states),action_rms_per_joint=np.sqrt(np.mean(actions**2,axis=0)).tolist(),residual_rms_per_joint=np.sqrt(mse).tolist(),residual_rms_over_fit_action_std=(np.sqrt(mse)/us).tolist(),r2_against_fit_mean=np.where(baseline>1e-18,1-mse/np.maximum(baseline,1e-18),np.nan).tolist(),action_abs_max=float(np.abs(actions).max()))
summary={split:summarize(parts) for split,parts in [('fit',fit),('validation',validation)]}
for split,parts in [('fit',fit),('validation',validation)]:
 nonzero=[x for x in parts if np.max(np.abs(x['action']))>1e-10];summary[split+'_nonzero_forcing']=summarize(nonzero)
sv=np.linalg.svd(x,compute_uv=False);residual=y-x@beta;rsv=np.linalg.svd(residual,compute_uv=False)
per_file=[dict(file=x['row']['trace'],split=x['row']['split'],sha256=x['sha256'],metrics=summarize([x])) for x in records]
report=dict(plan=plan,action_joint_names=mapping['action_joint_names'],fit_state_columns_kept=np.flatnonzero(keep).tolist(),state_design_singular_values=sv.tolist(),state_design_rank=int((sv>sv[0]*plan['svd_rank_relative_tolerance']).sum()),normalized_action_residual_singular_values=rsv.tolist(),summary=summary,per_file=per_file)
# JSON null denotes a zero-baseline R2; no invented success for zero-action files.
def clean(x):
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
(a.output/'audit.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False))
for split,d in summary.items():print(split,'frames',d['frames'],'median_R2',float(np.nanmedian(d['r2_against_fit_mean'])),'median_residual_over_fitstd',float(np.median(d['residual_rms_over_fit_action_std'])))
print('fit state rank',report['state_design_rank'],'of',len(report['fit_state_columns_kept']))
