"""All recorded GPU contact samples aligned to their own transition predictions."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from supermimic.algo.networks import WorldModel,WorldModelCfg
from supermimic_pose_integration_diagnostic import apply_offline_prediction_contract
p=argparse.ArgumentParser(description=__doc__)
for key in ['root','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir();contact_root=a.root/'contact_probe_v2';samples=json.loads((contact_root/'contact_samples.json').read_text());assert len(samples)==780
verified=json.loads((contact_root/'integrity.json').read_text());assert verified['finite'] and verified['contiguous'] and verified['samples']==780 and verified['episodes']==16
meta=json.loads((contact_root/'results.json').read_text())['rows'];mapping=json.loads((contact_root/'control_mapping.json').read_text());cache={};provenance=[]
for row in meta:
 path=contact_root/row['trace']
 with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in ['state','action','context','next_state']}
 assert all(np.isfinite(x).all() for x in d.values()) and np.array_equal(d['next_state'][:-1],d['state'][1:]);cache[row['trace']]=(d,row);provenance.append(dict(file=row['trace'],sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
selected=[];keys=[]
for sample in samples:
 key=f"seed{sample['seed']}_amp{sample['amplitude']:.2f}.npz";d,row=cache[key];i=sample['step'];assert 0<=i<len(d['state']);keys.append((key,i))
 counts={s:sum('terrain' in c['geoms'] and any(s+'_foot_collision' in n for n in c['geoms']) and c['normal_force_n']>1e-4 for c in sample['contacts']) for s in ['left','right']}
 group='both_3plus' if min(counts.values())>=3 else ('no_loaded_points' if max(counts.values())==0 else 'some_2orless')
 selected.append(dict(seed=sample['seed'],amplitude=sample['amplitude'],step=i+1,source_step_index=i,split=row['split'],group=group,counts=counts,state=d['state'][i],next_state=d['next_state'][i],action=d['action'][i],context=d['context'][i]))
for row in selected:
 row['constant_velocity_errors']={}
 for name,sl in [('root_angular_velocity',slice(10,13)),('root_linear_velocity',slice(7,10)),('joint_velocity',slice(43,73))]:row['constant_velocity_errors'][name]=float(np.sqrt(np.mean((row['state'][sl]-row['next_state'][sl])**2)))
assert len(set(keys))==len(keys)
data={k:torch.tensor(np.stack([r[k] for r in selected]),device='cuda:0') for k in ['state','action','context','next_state']};models={'independent':a.root/'standing_branch_pair_v1/candidate/fit/world_model_calibrated.pt','integrated':a.root/'pose_velocity_fit_v1/fit/world_model_calibrated.pt'};summaries={}
for name,path in models.items():
 c=torch.load(path,map_location='cpu',weights_only=False);w=c['world_model'];cfg=c['cfg'];assert cfg['extra']['tw_state_joint_names']==mapping['state_joint_names'] and cfg['extra']['tw_action_joint_names']==mapping['action_joint_names']
 base=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda();base.load_state_dict(w);base.eval();model=apply_offline_prediction_contract(base,c)
 with torch.no_grad():pred=model(data['state'],data['action'],data['context']);assert torch.isfinite(pred).all()
 errors={}
 for group,sl in [('root_angular_velocity',slice(10,13)),('root_linear_velocity',slice(7,10)),('joint_velocity',slice(43,73))]:errors[group]=(pred[:,sl]-data['next_state'][:,sl]).square().mean(-1).sqrt().cpu().numpy()
 for j,row in enumerate(selected):row.setdefault('model_errors',{})[name]={k:float(v[j]) for k,v in errors.items()}
 out=[]
 for split in ['fit','validation']:
  for group in ['both_3plus','some_2orless','no_loaded_points']:
   ids=[i for i,r in enumerate(selected) if r['split']==split and r['group']==group]
   baseline={k:float(np.sqrt(np.mean([selected[i]['constant_velocity_errors'][k]**2 for i in ids]))) for k in errors} if ids else {}
  out.append(dict(split=split,contact_group=group,samples=len(ids),constant_velocity_rmse=baseline,metrics={k:dict(rmse=float(np.sqrt(np.mean(v[ids]**2))),median=float(np.median(v[ids])),p95=float(np.percentile(v[ids],95))) for k,v in errors.items()} if ids else {}))
 summaries[name]=dict(model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),groups=out)
rows=[{k:v for k,v in r.items() if k not in ['state','next_state','action','context']} for r in selected]
forcing=[]
for split in ['fit','validation']:
 for amp in [0,.02,.05,.1]:
  for group in ['both_3plus','some_2orless','no_loaded_points']:
   rr=[r for r in rows if r['split']==split and r['amplitude']==amp and r['group']==group]
   if not rr:continue
   baseline=float(np.sqrt(np.mean([r['constant_velocity_errors']['root_angular_velocity']**2 for r in rr])))
   forcing.append(dict(split=split,amplitude=amp,group=group,samples=len(rr),root_angular_velocity_constant_rmse=baseline,models={name:dict(rmse=float(np.sqrt(np.mean([r['model_errors'][name]['root_angular_velocity']**2 for r in rr]))),ratio_to_constant=float(np.sqrt(np.mean([r['model_errors'][name]['root_angular_velocity']**2 for r in rr])))/baseline if baseline>0 else None) for name in models}))
report=dict(scope='All780native GPU samples, own post-step next_state alignment; point counts only, no claim of full face contact or causality. Original forcing split, already inspected synthetic development data, not fresh independent-motion validation.',force_threshold_n=1e-4,source_contact_sha256=hashlib.sha256((contact_root/'contact_samples.json').read_bytes()).hexdigest(),trace_sources=provenance,forcing_strata=forcing,models=summaries,rows=rows,promoted=False)
(a.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(summaries))
