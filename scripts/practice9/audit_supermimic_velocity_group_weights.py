"""Fit-only direct loss coefficients and velocity-group gradient diagnostics."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from supermimic_geometry_model import load_geometry_artifact
from supermimic_motion_windows import load_motion_fit_windows
p=argparse.ArgumentParser(description=__doc__)
for name in ['collection','audit','plan','output']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--checkpoint',type=Path,action='append',required=True)
a=p.parse_args();assert not a.output.exists()
mapping=json.loads((a.collection/'control_mapping.json').read_text())
w,meta=load_motion_fit_windows(a.collection,a.audit,mapping,window=1,source_plan_path=a.plan)
data={k:v if k=='state' else v[:,0] for k,v in w.items()}
groups={'linear':(7,10),'angular':(10,13),'joint_velocity':(43,73)}
rows=[];models=[]
for path in a.checkpoint:
 sha=hashlib.sha256(path.read_bytes()).hexdigest();artifact=torch.load(path,map_location='cpu',weights_only=False)
 model,predict=load_geometry_artifact(artifact);params=list(model.net.parameters());generator=torch.Generator(device='cuda:0').manual_seed(20261103)
 for batch in range(8):
  ix=torch.randint(len(data['state']),(32,),device='cuda:0',generator=generator)
  prediction=predict(data['state'][ix],data['action'][ix],data['context'][ix])
  error=model.state_norm.normalize(prediction)-model.state_norm.normalize(data['next_state'][ix])
  losses={name:error[:,lo:hi].square().mean() for name,(lo,hi) in groups.items()}
  grads={}
  for index,(name,value) in enumerate(losses.items()):
   g=torch.autograd.grad(value,params,retain_graph=index<2)
   grads[name]=torch.cat([x.reshape(-1) for x in g]).double()
   assert torch.isfinite(grads[name]).all()
  cosine={}
  for left,right in [('linear','angular'),('linear','joint_velocity'),('angular','joint_velocity')]:
   x,y=grads[left],grads[right];cosine[left+'__'+right]=float((x*y).sum()/(x.norm()*y.norm()).clamp_min(1e-30))
  rows.append(dict(checkpoint=str(path),batch=batch,normalized_mse={k:float(v) for k,v in losses.items()},gradient_norm={k:float(v.norm()) for k,v in grads.items()},cosine=cosine))
 for key,value in model.state_dict().items():assert torch.equal(value.cpu(),artifact['world_model'][key])
 assert hashlib.sha256(path.read_bytes()).hexdigest()==sha;models.append(dict(path=str(path),sha256=sha))
# Motion is half of H1 batch; base H1 coefficient .5; velocity anchor is .5/3.
direct={name:.5*(.5*((hi-lo)/69)+.5/3)+(1 if name=='linear' else 0) for name,(lo,hi) in groups.items()}
summary={str(path):dict(median_mse={k:float(np.median([r['normalized_mse'][k] for r in rows if r['checkpoint']==str(path)])) for k in groups},median_gradient_norm={k:float(np.median([r['gradient_norm'][k] for r in rows if r['checkpoint']==str(path)])) for k in groups}) for path in a.checkpoint}
report=dict(models=models,fit_input=meta,validation_arrays_loaded=False,optimizer_steps=0,direct_motion_h1_coefficients=direct,coefficient_scope='Direct velocity MSE only; excludes pose integration, H8, derivative/response and clipping/Adam effects',summary=summary,rows=rows,scope='Fixed8 fit batches; no causal or optimization guarantee and no policy acceptance')
a.output.write_text(json.dumps(report,indent=2));print(json.dumps(dict(direct_coefficients=direct,summary=summary)))
