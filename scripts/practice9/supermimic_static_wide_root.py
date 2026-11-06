"""Additional fit-only static8/16 root responses, original normalization."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from supermimic_root_state_response import RootStateResponseLoss
class AdditionalStaticWideRootLoss:
 def __init__(self,root,cfg,device='cuda:0'):
  root=Path(root);proof=json.loads((root/'integrity.json').read_text());plan=json.loads((root/'plan.json').read_text())
  assert proof['passed'] and len(proof['files'])==72 and proof['retained_original_files']==40
  assert plan['contract']=='wide_static_root_velocity_branches_v1' and plan['levels']==[1.,2.,8.,16.]
  assert set(plan['fit_warmups']).isdisjoint(plan['validation_warmups'])
  mapping=json.loads((root/'control_mapping.json').read_text())
  for k in ['state','action']:assert mapping[k+'_joint_names']==cfg['extra']['tw_'+k+'_joint_names']
  hashes={x['file']:x['sha256'] for x in proof['files']};loaded={};files=[]
  for row in json.loads((root/'results.json').read_text())['rows']:
   if row['split']!='fit' or row['kind']!='branch' or row['level'] not in [8.,16.]:continue
   p=(root/row['file']).resolve();assert p.is_relative_to(root.resolve());digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==hashes[row['file']]
   with np.load(p) as z:loaded[(row['warmup'],row['level'],row['assignment'])]={k:z[k].copy() for k in ['state','action','context','next_state']}
   files.append({'file':row['file'],'sha256':digest})
  assert len(loaded)==16
  arrays={k:[] for k in ['state','action','context','next_state']};den=[];axes=[]
  for warmup in plan['fit_warmups']:
   for level in [8.,16.]:
    a,b=[loaded[(warmup,level,i)] for i in [0,1]]
    for k in arrays:arrays[k].append(np.stack([a[k][::2],a[k][1::2],b[k][::2],b[k][1::2]],axis=1))
    den.extend(4*level*np.array(plan['base_delta']));axes.extend(range(6))
  self.data={k:torch.as_tensor(np.concatenate(v),device=device,dtype=torch.float32) for k,v in arrays.items()}
  self.den=torch.tensor(den,device=device,dtype=torch.float32)[:,None];self.axes=torch.tensor(axes,device=device)
  assert len(self.den)==48;self.target=RootStateResponseLoss.derivative(self,self.data['next_state']).detach();assert torch.isfinite(self.target).all()
  self.generator=torch.Generator(device=device).manual_seed(64);self.history=[]
  self.meta={'fit_files':files,'fit_conditions':48,'levels':[8.,16.],'seed':64,'validation_arrays_loaded':False,'integrity_sha256':hashlib.sha256((root/'integrity.json').read_bytes()).hexdigest(),'normalization':'Original root-state derivative, frozen input/output root velocity standard deviations; half weight matching original static half','loader_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 def __call__(self,model):
  idx=torch.randint(48,(8,),generator=self.generator,device=self.den.device)
  pred=model(*[self.data[k][idx].reshape(-1,w) for k,w in [('state',73),('action',30),('context',34)]]).reshape(8,4,73)
  response=(pred[:,1,7:13]-pred[:,0,7:13]+pred[:,2,7:13]-pred[:,3,7:13])/self.den[idx]
  std=(model.state_norm.var[7:13]+model.state_norm.eps).sqrt();std=torch.where(model.state_norm.frozen_mask[7:13],torch.ones_like(std),std)
  value=.5*((response-self.target[idx])*std[self.axes[idx],None]/std[None,:]).square().mean();assert torch.isfinite(value)
  self.history.append({'update':len(self.history)+1,'indices':idx.cpu().tolist(),'half_weighted_loss':float(value)})
  return value
