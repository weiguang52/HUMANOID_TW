"""Fit-only local gradient scope evidence; no optimizer or checkpoint mutation."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from supermimic.algo.networks import WorldModel
from supermimic_geometry_model import load_geometry_artifact
from supermimic_motion_windows import load_motion_fit_windows

p=argparse.ArgumentParser(description=__doc__)
for name in ['collection','audit','plan','output']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--checkpoint',type=Path,action='append',required=True)
a=p.parse_args();assert not a.output.exists()
assert 'lin_vel = state[:, 7:10] + non_quat[:, 3:6]' in inspect.getsource(WorldModel.forward)
mapping=json.loads((a.collection/'control_mapping.json').read_text())
windows,meta=load_motion_fit_windows(a.collection,a.audit,mapping,window=1,source_plan_path=a.plan)
data={k:v if k=='state' else v[:,0] for k,v in windows.items()}
rows=[];models=[]
for path in a.checkpoint:
 sha=hashlib.sha256(path.read_bytes()).hexdigest()
 artifact=torch.load(path,map_location='cpu',weights_only=False)
 model,predict=load_geometry_artifact(artifact)
 params=list(model.net.parameters());head=list(model.net[-1].parameters())
 assert model.out_dim==72 and model.net[-1].out_features==72 and len(head)==2
 head_ids={id(x) for x in head}
 generator=torch.Generator(device='cuda:0').manual_seed(20261103)
 for batch in range(8):
  indices=torch.randint(len(data['state']),(32,),device='cuda:0',generator=generator)
  prediction=predict(data['state'][indices],data['action'][indices],data['context'][indices])
  error=model.state_norm.normalize(prediction)-model.state_norm.normalize(data['next_state'][indices])
  linear=error[:,7:10].square().mean();joint=error[:,13:73].square().mean();omega=error[:,10:13].square().mean()
  extra=torch.autograd.grad(linear,params,retain_graph=True)
  joint_grad=torch.autograd.grad(joint,params,retain_graph=True)
  omega_grad=torch.autograd.grad(omega,params)
  total=sum(float(x.double().square().sum()) for x in extra)
  trunk=sum(float(x.double().square().sum()) for parameter,x in zip(params,extra) if id(parameter) not in head_ids)
  for parameter,x in zip(params,extra):
   if id(parameter) in head_ids:
    outside=x.clone();outside[3:6]=0
    assert torch.count_nonzero(outside)==0, 'Unexpected root-linear output dependency'
  result=dict(checkpoint=str(path),batch=batch,linear_loss=float(linear),joint_loss=float(joint),omega_loss=float(omega),trunk_gradient_squared_fraction=trunk/max(total,1e-30))
  for name,protected in [('joint',joint_grad),('omega',omega_grad)]:
   dot=sum(float((x.double()*y.double()).sum()) for x,y in zip(extra,protected))
   norm=sum(float(y.double().square().sum()) for y in protected)
   head_dot=sum(float((x.double()*y.double()).sum()) for parameter,x,y in zip(params,extra,protected) if id(parameter) in head_ids)
   result[name]=dict(full_dot=dot,full_cosine=dot/max((total*norm)**.5,1e-30),full_conflict=dot<0,head_only_dot=head_dot)
  rows.append(result)
 for key,value in model.state_dict().items():assert torch.equal(value.cpu(),artifact['world_model'][key])
 assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
 models.append(dict(path=str(path),sha256=sha))
report=dict(models=models,fit_input=meta,optimizer_steps=0,validation_arrays_loaded=False,root_linear_delta_rows=[3,4,5],rows=rows,scope='Fixed8 fit-only batches; local Euclidean gradient evidence only, no Adam or training nonregression guarantee, no policy acceptance')
a.output.write_text(json.dumps(report,indent=2));print(json.dumps({str(path):dict(joint_conflicts=sum(r['joint']['full_conflict'] for r in rows if r['checkpoint']==str(path)),omega_conflicts=sum(r['omega']['full_conflict'] for r in rows if r['checkpoint']==str(path)),median_trunk_fraction=float(np.median([r['trunk_gradient_squared_fraction'] for r in rows if r['checkpoint']==str(path)]))) for path in a.checkpoint}))
