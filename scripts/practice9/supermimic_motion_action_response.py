"""Fit-only antithetic action response at archived dynamic PD commands."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
class MotionActionResponseLoss:
 def __init__(self,root,integrity,cfg,device='cuda:0'):
  root=Path(root);integrity=Path(integrity);proof=json.loads(integrity.read_text());assert proof['passed'] and not proof['smoke'];assert hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()==proof['plan_sha256']
  mapping=json.loads((root/'control_mapping.json').read_text())
  for source,target in [('state_joint_names','tw_state_joint_names'),('action_joint_names','tw_action_joint_names')]:assert mapping[source]==cfg['extra'][target]
  hashes={x['file']:x['sha256'] for x in proof['files']};groups={};self.files=[]
  for row in json.loads((root/'results.json').read_text())['rows']:
   if row['split']!='fit' or row['level']==0:continue
   path=(root/row['file']).resolve();assert path.is_relative_to(root.resolve());digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==hashes[row['file']]
   with np.load(path) as z:d={k:z[k].copy() for k in ('state','action','context','next_state')}
   groups.setdefault((row['source_trace'],row['index'],row['level']),{})[row['assignment']]=d;self.files.append({'file':row['file'],'sha256':digest})
  assert len(groups)==96 and len(self.files)==192;arrays={k:[] for k in ('state','action','context','next_state')};den=[]
  for (_,_,eps),g in sorted(groups.items()):
   assert set(g)=={0,1};a,b=[g[i] for i in (0,1)]
   for key in arrays:arrays[key].append(np.stack([a[key][::2],a[key][1::2],b[key][::2],b[key][1::2]],axis=1))
   den.extend([4*eps]*30)
  self.data={k:torch.as_tensor(np.concatenate(v),device=device,dtype=torch.float32) for k,v in arrays.items()};self.den=torch.tensor(den,device=device,dtype=torch.float32)[:,None];assert len(self.den)==2880
  self.target=self.derivative(self.data['next_state']).detach();assert torch.isfinite(self.target).all();self.generator=torch.Generator(device=device).manual_seed(57);self.last_indices=None
  self.meta={'conditions':2880,'fit_files':self.files,'integrity_sha256':hashlib.sha256(integrity.read_bytes()).hexdigest(),'generator_seed':57,'rule':'Same as static response: half parent-normalized root angular plus joint velocity derivative MSE; batch16,64 endpoints. Fit-only dynamic PD action centers.'}
 def derivative(self,value,idx=None):
  denominator=self.den if idx is None else self.den[idx]
  return (value[:,1]-value[:,0]+value[:,2]-value[:,3])/denominator
 def evaluate(self,model,idx):
  d=self.data;batch=len(idx);pred=model(d['state'][idx].reshape(-1,73),d['action'][idx].reshape(-1,30),d['context'][idx].reshape(-1,34)).reshape(batch,4,73)
  response=self.derivative(pred,idx);std=(model.state_norm.var+model.state_norm.eps).sqrt();std=torch.where(model.state_norm.frozen_mask,torch.ones_like(std),std);error=(response-self.target[idx])/std
  value=.5*(error[:,10:13].square().mean()+error[:,43:73].square().mean());assert torch.isfinite(value);return value
 def __call__(self,model,batch=16):
  idx=torch.randint(len(self.den),(batch,),device=self.den.device,generator=self.generator);self.last_indices=idx.detach().cpu().tolist();return self.evaluate(model,idx)
