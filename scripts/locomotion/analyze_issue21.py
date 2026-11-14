"""Analyze all available diagnostic pairs without changing frozen rules."""
import json,numpy as np,csv
from pathlib import Path
from issue21_common import ROOT,OLD,sha,save
from issue21_analysis import Mirror,crossings,slope,growth,GROUPS

def analyze():
 folder=ROOT/'evaluation';summaries=[];cross=[];components=[]
 for path in sorted(folder.glob('*.json')):
  meta=json.loads(path.read_text())
  if meta['mode']=='perturb':continue
  mirror=Mirror(meta['joint_names'],meta['contact_body_names'])
  for record in meta['records']:
   if record['phase']!=0:continue
   p=Path(record['path']);other=p.with_name(p.name.replace('_k0.npz','_k10.npz'));assert sha(p)==record['sha256'];a=np.load(p);b=np.load(other);errors=mirror.residuals(a,b);tag=p.stem.removesuffix('_k0');case={'case':tag,'mode':meta['mode'],'kind':record['kind'],'training_seed':meta['training_seed'],'num_envs':meta['num_envs'],'command':record['command'],'initial_mirror_max':{},'metrics':{}}
   expected=mirror.root(a['initial_root'].astype(float));actual=b['initial_root'].astype(float);actual[:,3:7]*=np.where((actual[:,3:7]*expected[:,3:7]).sum(-1,keepdims=True)<0,-1,1);case['initial_mirror_max']['root']=float(abs(actual-expected).max())
   for n in ['joint_pos','joint_vel']:case['initial_mirror_max'][n]=float(abs(b['initial_'+n]-mirror.joint(a['initial_'+n])).max())
   saved={}
   for name,e in errors.items():
    offset=0 if name in GROUPS or name=='action' else 1;flat=e.reshape(e.shape[0],e.shape[1],-1);linf=abs(flat).max(-1);l2=np.sqrt((flat*flat).sum(-1));saved[name+'_signed']=e;saved[name+'_linf']=linf;saved[name+'_l2']=l2
    if name in ['joint_pos','joint_vel','applied_torque','action','joint_pos_obs','joint_vel_obs','last_action','estimated_pd_torque','explicit_actuation_force','joint_target']:names=meta['joint_names']
    elif name=='contact':names=[n+'.'+axis for n in meta['contact_body_names'] for axis in ['Fx','Fy','Fz']]
    else:names={'ang_vel':['wx','wy','wz'],'gravity':['gx','gy','gz'],'commands':['vx','vy','wz'],'gait_phase':['sin','cos'],'root_position':['x','y','z'],'root_quaternion':['qw','qx','qy','qz'],'root_linear_velocity':['vx','vy','vz'],'root_angular_velocity':['wx','wy','wz']}[name]
    for ci,component in enumerate(names):
     for threshold in [1e-6,1e-4,1e-2]:
      hits=np.argwhere(abs(flat[:,:,ci])>threshold);entry={'case':tag,'quantity':name,'component':component,'threshold':threshold,'step':None,'env':None,'signed_residual':None}
      if len(hits):
       ti,ei=hits[0];entry.update(step=int(ti)+offset,env=int(ei),signed_residual=float(flat[ti,ei,ci]))
      components.append(entry)
    first=crossings(e,names,offset);case['metrics'][name]={'max_linf':float(linf.max()),'step0_or1_linf':float(linf[0].max()),'max_l2':float(l2.max()),'first_crossings':first}
    for hit in first:
     if hit['first']:cross.append({'case':tag,'quantity':name,'threshold':hit['threshold'],**hit['first']})
   force=np.maximum(np.linalg.norm(a['contact'],axis=-1).max(-1),np.linalg.norm(b['contact'],axis=-1).max(-1));fc=[int(v[0]) if len(v:=np.flatnonzero(force[:,e]>1.)) else None for e in range(force.shape[1])];case['first_contact_post_step']=[None if v is None else v+1 for v in fc];case['initial_contact_max_force']=max(float(np.linalg.norm(a['initial_contact'],axis=-1).max()),float(np.linalg.norm(b['initial_contact'],axis=-1).max()));case['max_contact_force']=float(force.max());case['growth_root_angular_velocity']=growth(saved['root_angular_velocity_linf'],fc);case['terminated_initial_trial']={'k0':a['terminated'].any(0).tolist(),'k10':b['terminated'].any(0).tolist()}
   case['counterfactual_action_error_max']=max(float(abs(a['counterfactual_action_error']).max()),float(abs(b['counterfactual_action_error']).max()));dest=ROOT/'analysis'/(tag+'_residuals.npz');dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,**saved);case['residual_file']=str(dest);case['residual_sha256']=sha(dest);summaries.append(case)
 save('pair_analysis.json',summaries)
 if components:
  with (ROOT/'first_component_crossings.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(components[0]));w.writeheader();w.writerows(components)
 if cross:
  with (ROOT/'first_crossings.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(cross[0]));w.writeheader();w.writerows(cross)
 perturb=[]
 for p in sorted(folder.glob('perturb_*.json')):
  meta=json.loads(p.read_text());ts=meta['training_seed'];control=json.loads((ROOT/'controls'/('s%d.json'%ts)).read_text());old=Path(control['records'][0]['path']);ref=np.load(old);baseline,baseline_valid=slope(ref['yaw'],ref['terminated']);baseline_record={'path':str(old),'sha256':sha(old),'fresh_independent_zero_epsilon_control':True,'censored':int((~baseline_valid).sum()),'slope_mean':float(np.nanmean(baseline)) if baseline_valid.any() else None,'positive':int((baseline>0).sum()),'negative':int((baseline<0).sum())}
  for rec in meta['records']:
   z=np.load(rec['path']);v,valid=slope(z['yaw'],z['terminated']);comparison=valid & baseline_valid;flips=comparison & (v*baseline<0);perturb.append({'training_seed':ts,'epsilon':rec['epsilon'],'baseline':baseline_record,'comparison_valid':int(comparison.sum()),'flips':int(flips.sum()),'flip_fraction_valid':float(flips.sum()/max(comparison.sum(),1)),'any_flip':bool(flips.any()),'majority_flip':bool(comparison.sum()>0 and flips.sum()>comparison.sum()/2),'positive':int((v>0).sum()),'negative':int((v<0).sum()),'zero':int((v==0).sum()),'censored':int((~valid).sum()),'slope_mean':float(np.nanmean(v)) if valid.any() else None,'slopes':[float(y) if np.isfinite(y) else None for y in v],'raw':rec['path']})
 save('perturbation_analysis.json',perturb);print('ISSUE21_ANALYZED_PAIRS',len(summaries),'PERTURB',len(perturb),flush=True)
if __name__=='__main__':analyze()
