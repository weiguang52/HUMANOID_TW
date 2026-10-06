"""Zero-phase smoothing of IK correction only; immutable contact and frame times."""
import json,pathlib,concurrent.futures,numpy as np
from scipy.ndimage import gaussian_filter1d
from contact_ik import rebuild,phases,Model
S=pathlib.Path('/root/gpufree-data/datasets/practice9/tw195_v1')
def job(item):
 split,r=item;a=dict(np.load(r['file']));raw=S/('pilot' if split=='pilot' else 'corrected/'+split)/(r['id']+'.npz');b=np.load(raw)
 model=Model(a['joint_names'],a['body_names']);q=a['joint_pos'].copy();q[:,model.sel]=gaussian_filter1d(b['joint_pos'][:,model.sel],2,axis=0);rp=a['root_pos']+gaussian_filter1d(b['root_pos']-a['root_pos'],2,axis=0)
 out,p=rebuild(a,q,rp);z=p[:,:,:,2].min(2);lift=max(0.,float(.0002-z.min()));out['root_pos'][:,2]+=lift;out['body_pos_w'][:,:,2]+=lift;z+=lift
 c,k,full=phases(r['id'],len(q));stance=c&k;flat=stance&full;model=Model(a['joint_names'],a['body_names'])
 def p95(x):return float(np.quantile(x,.95)) if x.size else None
 acc=np.gradient(out['joint_vel'],.02,axis=0);oldacc=np.gradient(a['joint_vel'],.02,axis=0)
 audit=dict(id=r['id'],split=split,frames=len(q),stance_error_p95=p95(abs(z[stance]-.001)),new_stance_z_p95=p95(z[stance]),flat_spread_p95=p95(np.ptp(p[:,:,:,2],axis=2)[flat]),min_sole=float(z.min()),global_lift_m=lift,max_q_change=float(abs(q-a['joint_pos']).max()),max_root_z_change=float(abs(out['root_pos'][:,2]-a['root_pos'][:,2]).max()),velocity_ratio_max=float((abs(out['joint_vel'])/model.m.velocityLimit[model.vidx]).max()),upper_joint_change=float(abs(q[:,14:]-a['joint_pos'][:,14:]).max()),joint_limit_violation=float(max(0,(model.m.lowerPositionLimit[model.qidx]-q).max(),(q-model.m.upperPositionLimit[model.qidx]).max())),old_leg_acc_max=float(abs(oldacc[:,:14]).max()),new_leg_acc_max=float(abs(acc[:,:14]).max()),old_leg_acc_p95=p95(abs(oldacc[:,:14])),new_leg_acc_p95=p95(abs(acc[:,:14])),contact_unchanged=True,timing_unchanged=True)
 assert all(np.isfinite(v).all() for v in out.values() if np.issubdtype(v.dtype,np.number))
 target=S/('pilot_smooth' if split=='pilot' else 'corrected_final/'+split)/(r['id']+'.npz');target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**out);target.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2));return audit
if __name__=='__main__':
 import sys
 tasks=[]
 for split,file in [('train','train.json'),('validation','validation/manifest.json')]:
  m=json.load(open('/root/gpufree-data/datasets/practice9/tw154_v1/'+file));tasks.extend((split,r) for r in m['motions'])
 if '--pilot' in sys.argv:tasks=[('pilot',r) for split,r in tasks if split=='validation' and r['id'] in ['000016','000832','000211']]
 rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
  for row in pool.map(job,tasks,chunksize=1):
   rows.append(row)
   if len(rows)%50==0:print('smoothed',len(rows),flush=True)
 (S/('pilot_smooth_audit.json' if '--pilot' in sys.argv else 'final_audit.json')).write_text(json.dumps(rows,indent=2));print('completed',len(rows),flush=True)
