"""Fit-only absolute next-state velocity targets at archived dynamic action centers."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
class MotionActionCenterLoss:
 def __init__(self,root,integrity,cfg,device='cuda:0'):
  root=Path(root);integrity=Path(integrity);proof=json.loads(integrity.read_text());assert proof['passed'] and not proof['smoke'];assert hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()==proof['plan_sha256']
  mapping=json.loads((root/'control_mapping.json').read_text())
  for source,target in [('state_joint_names','tw_state_joint_names'),('action_joint_names','tw_action_joint_names')]:assert mapping[source]==cfg['extra'][target]
  hashes={x['file']:x['sha256'] for x in proof['files']};arrays={k:[] for k in ('state','action','context','next_state')};files=[];keys=set()
  for row in json.loads((root/'results.json').read_text())['rows']:
   if row['split']!='fit' or row['level']!=0:continue
   path=(root/row['file']).resolve();assert path.is_relative_to(root.resolve());digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==hashes[row['file']]
   key=(row['source_trace'],row['index']);assert key not in keys;keys.add(key)
   with np.load(path) as z:
    for k in arrays:
     value=z[k];assert np.array_equal(value,np.broadcast_to(value[:1],value.shape));arrays[k].append(value[:1].copy())
   files.append({'file':row['file'],'sha256':digest})
  assert len(keys)==48;self.data={k:torch.as_tensor(np.concatenate(v),device=device,dtype=torch.float32) for k,v in arrays.items()};self.generator=torch.Generator(device=device).manual_seed(61);self.last_indices=None
  self.meta={'conditions':48,'fit_files':files,'generator_seed':61,'rule':'Half parent-normalized root angular plus joint velocity absolute next-state MSE, batch16, exact unperturbed dynamic PD centers; fit-only.'}
 def evaluate(self,model,idx):
  d=self.data;pred=model(d['state'][idx],d['action'][idx],d['context'][idx]);std=(model.state_norm.var+model.state_norm.eps).sqrt();std=torch.where(model.state_norm.frozen_mask,torch.ones_like(std),std);error=(pred-d['next_state'][idx])/std
  value=.5*(error[:,10:13].square().mean()+error[:,43:73].square().mean());assert torch.isfinite(value);return value
 def __call__(self,model,batch=16):
  idx=torch.randint(48,(batch,),device=self.data['state'].device,generator=self.generator);self.last_indices=idx.detach().cpu().tolist();return self.evaluate(model,idx)
