"""Fit non-orientation goal statistics from training references; never a policy checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--parent',type=Path,required=True)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--isolation-contract',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert not a.output.exists()
a.output.mkdir(parents=True)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
contract=json.loads(a.isolation_contract.read_text())
assert contract['passed_source_isolation']
assert contract['training_manifest_sha256']==sha(a.manifest)
manifest=json.loads(a.manifest.read_text())
assert len(manifest['motions'])==332
files=[x['file'] for x in manifest['motions']]
assert not set(files)&{x['file'] for x in contract['motions']}
parent=torch.load(a.parent,map_location='cpu',weights_only=False)
assert parent.get('kind') is None
norm=parent['actor_critic']
cfg=tw_supermimic_env_cfg(files,num_envs=1,play=True)
cfg.events={}
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
 command=env.command_manager.get_term('motion')
 library=command.library
 feature=command.goal_features
 assert feature.frame_dim==75 and feature.horizon==10
 source_mean=norm['goal_norm.mean'].double()
 source_var=norm['goal_norm.var'].double()
 frozen=norm['goal_norm.frozen_mask']
 mask=torch.tensor([i%75 not in range(63,69) for i in range(feature.dim)])&~frozen
 assert int(mask.sum())==690
 count=0
 mean=torch.zeros(feature.dim,dtype=torch.float64,device='cuda:0')
 m2=torch.zeros_like(mean)
 inputs=[]
 for clip,row in enumerate(manifest['motions']):
  n=int(library.clip_lengths[clip])
  assert n==row['frames']
  inputs.append({'id':row['id'],'file':row['file'],'sha256':sha(row['file']),'frames':n})
  for start in range(0,n,512):
   times=torch.arange(start,min(start+512,n),device='cuda:0')
   clips=torch.full_like(times,clip)
   ref=library.gather(clips,times,feature.horizon+1)
   goal=feature.from_reference(ref,ref['body_quat'][:,0,0]).double()
   assert torch.isfinite(goal).all()
   k=len(goal)
   batch_mean=goal.mean(0)
   delta=batch_mean-mean
   m2+=((goal-batch_mean)**2).sum(0)+delta.square()*count*k/(count+k)
   mean+=delta*k/(count+k)
   count+=k
  print(json.dumps({'clip':clip,'id':row['id'],'total_frames':count}),flush=True)
 mean=mean.cpu();var=(m2/count).cpu()
 assert bool((var>=0).all()) and torch.isfinite(mean).all() and torch.isfinite(var).all()
 out_mean=source_mean.clone();out_var=source_var.clone()
 out_mean[mask]=mean[mask];out_var[mask]=var[mask]
 assert torch.equal(out_mean[~mask],source_mean[~mask]) and torch.equal(out_var[~mask],source_var[~mask])
 artifact={'kind':'offline_training_goal_normalizer_calibration_not_policy_checkpoint','parent_sha256':sha(a.parent),'manifest_sha256':sha(a.manifest),'isolation_contract_sha256':sha(a.isolation_contract),'contract':'all_training_frames_reference_root_goal_nonorientation_v1','fit_frames':count,'goal_mean':out_mean.float(),'goal_var':out_var.float(),'replacement_mask':mask,'frozen_mask':frozen,'epsilon':1e-5,'orientation_statistics_preserved':True,'inputs':inputs,'scope':'Reference-only statistics. No simulator steps, policy training, test-array loading or strategy promotion. Critic goal slice must use the same statistics if tested; all other statistics remain baseline. Does not calibrate observed off-reference root orientation.'}
 torch.save(artifact,a.output/'goal_normalizer.pt')
 report={k:v for k,v in artifact.items() if not torch.is_tensor(v)}
 report.update(artifact_sha256=sha(a.output/'goal_normalizer.pt'),replaced_dimensions=int(mask.sum()),source_nearzero_variance_dimensions=int((source_var[mask]<1e-5).sum()),fit_nearzero_variance_dimensions=int((out_var[mask]<1e-5).sum()),fit_variance_min=float(out_var[mask].min()),fit_variance_median=float(out_var[mask].median()))
 (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
 (a.output/'status').write_text('calibrated_not_trained_or_accepted\n')
finally:
 env.close()
