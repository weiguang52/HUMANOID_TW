"""Contact evidence and instrumentation equivalence; no automatic policy promotion."""
from pathlib import Path
import json
import numpy as np
R=Path('/root/gpufree-data/datasets/practice9/contact_points_20261025')
OLD=R.parent/'perturbation_20261025'
results={}
for model in ['original','baseline']:
 for view in ['side','front']:
  for mid in [2,3]:
   p=R/model/view/f'motion_{mid}.npz'
   if not p.exists():continue
   a=np.load(p);b=np.load(OLD/model/'batch_42'/f'motion_{mid}_v00.npz')
   t=a['control_step']*float(a['control_dt']);force=a['contact_point_normal_load'];speed=a['contact_point_xy_rms_velocity'];count=a['contact_point_count']
   windows=[]
   for lo,hi in [(1,5),(5,10),(10,15),(15,20)]:
    mask=(t>=lo)&(t<hi);d=np.diff(a['foot_position'][:,:,:2],axis=0)*1000
    loaded=(force[1:]>1e-6)&(force[:-1]>1e-6);ix=mask[1:] & mask[:-1]
    windows.append(dict(start=lo,end=hi,point_speed_mm_s=np.nanmean(speed[mask],axis=0).tolist(),
     positive_load_fraction=(force[mask]>1e-6).mean(0).tolist(),above_1n_fraction=(force[mask]>1.).mean(0).tolist(),
     foot_origin_delta_loaded_mm=np.where(loaded[:,:,None],d,0)[ix].sum(0).tolist(),
     foot_origin_delta_unloaded_or_transition_mm=np.where(~loaded[:,:,None],d,0)[ix].sum(0).tolist()))
   results[f'{model}/{view}/{mid}']=dict(control_steps=len(a['terminal']),terminations=int(a['terminal'].sum()),
    max_anchor_difference_from_old_m=float(abs(a['anchor_actual']-b['anchor_actual']).max()),
    max_joint_difference_from_old_rad=float(abs(a['q']-b['q']).max()),
    max_force_disagreement_n=float(abs(force-a['force'][:,:,2]).max()),max_points=count.max(0).tolist(),
    net_drift_after_1s_mm=float(np.linalg.norm(a['anchor_actual'][-1,:2]-a['anchor_actual'][50,:2])*1000),windows=windows)
(R/'contact_summary.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
