"""Matched native teacher-forced and autoregressive windows; no fitting or policy claim."""
import argparse,ast,json,hashlib,sys,datetime
from pathlib import Path
import numpy as np
import torch
from supermimic_pose_integration_diagnostic import PoseVelocityModel
p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--checkpoint',type=Path,action='append',required=True);p.add_argument('--trace-manifest',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
loader=Path('scripts/practice9/audit_supermimic_calibration_model.py');tree=ast.parse(loader.read_text());cut=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='trainer' for t in n.targets));program=compile(ast.fix_missing_locations(ast.Module(body=tree.body[:cut],type_ignores=[])),str(loader),'exec')
manifest=json.loads(a.trace_manifest.read_text()) if a.trace_manifest else None
if manifest:assert manifest['collection']==str(a.collection) and manifest['results_sha256']==hashlib.sha256((a.collection/'results.json').read_bytes()).hexdigest()
out={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'metric_precision':'original API dtype for endpoint metrics; float64 for additional velocity accumulation','trace_manifest_sha256':hashlib.sha256(a.trace_manifest.read_bytes()).hexdigest() if a.trace_manifest else None,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'loader_sha256':hashlib.sha256(loader.read_bytes()).hexdigest(),'models':[],'scope':'All valid fixed H1/H8/H32 windows, matched recorded actions/context; teacher forcing uses real state only for model input, integrates predicted velocities independently; native endpoint velocity integration defect retained; no fitting or policy acceptance'}
def metric(pred,target):
 q=torch.nn.functional.normalize(pred[:,3:7],dim=-1);tq=torch.nn.functional.normalize(target[:,3:7],dim=-1)
 return {'root_position_p95_m':float(torch.quantile((pred[:,:3]-target[:,:3]).norm(dim=-1),.95)),'root_orientation_p95_rad':float(torch.quantile(2*torch.acos((q*tq).sum(-1).abs().clamp(0,1)),.95)),**{k:float((pred[:,sl]-target[:,sl]).square().mean().sqrt()) for k,sl in [('root_linear_rmse',slice(7,10)),('root_angular_rmse',slice(10,13)),('joint_position_rmse',slice(13,43)),('joint_velocity_rmse',slice(43,73))]}}
for checkpoint in a.checkpoint:
 sys.argv=[str(loader),'--collection',str(a.collection),'--checkpoint',str(checkpoint),'--output',str(a.output/'unused.json')];env={'__name__':'__main__','__file__':str(loader)};exec(program,env);model=env['m'];before={k:v.clone() for k,v in model.state_dict().items()};integration=next(x for x in model.modules() if isinstance(x,PoseVelocityModel));rows=[]
 for row in json.loads((a.collection/'results.json').read_text())['rows']:
  path=(a.collection/row['trace']).resolve();assert path.is_relative_to(a.collection.resolve());digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==(row['sha256'] if 'sha256' in row else manifest['traces'][row['trace']])
  with np.load(path) as z:data={k:torch.as_tensor(z[k].copy(),device='cuda') for k in ['state','action','context','next_state']}
  assert torch.equal(data['state'][1:],data['next_state'][:-1]);n=len(data['state']);results={}
  with torch.no_grad():
   for h in [1,8,32]:
    count=n-h+1
    if count<=0:results[str(h)]={'windows':0};continue
    auto=data['state'][:count];teacher=auto.clone();oracle=auto.clone();accum={k:torch.zeros_like(auto[:,sl],dtype=torch.float64) for k,sl in [('teacher',slice(7,10)),('auto',slice(7,10))]};velocity={k:torch.zeros(3,device='cuda',dtype=torch.float64) for k in ['teacher','auto']}
    for j in range(h):
     real=data['state'][j:j+count];target=data['next_state'][j:j+count];act=data['action'][j:j+count];ctx=data['context'][j:j+count]
     forced=model(real,act,ctx);auto=model(auto,act,ctx)
     for name,pred in [('teacher',forced),('auto',auto)]:
      accum[name]+=.02*(pred[:,7:10].double()-target[:,7:10].double())
      velocity[name]+=torch.stack([(pred[:,sl].double()-target[:,sl].double()).square().mean() for sl in [slice(7,10),slice(10,13),slice(43,73)]])/h
     def integrate(prev,endpoint):
      hybrid=torch.cat([prev[:,:7],real[:,7:13],prev[:,13:43],real[:,43:73]],dim=-1)
      return integration.integrate_prediction(hybrid.double(),endpoint.double()).to(real.dtype)
     teacher=integrate(teacher,forced);oracle=integrate(oracle,target)
    target=data['next_state'][h-1:h-1+count];entry={'windows':count,'autoregressive':metric(auto,target),'teacher_integrated':metric(teacher,target),'native_integration_defect':metric(oracle,target)}
    for name in ['teacher','auto']:entry[name+'_velocity_rmse_over_window']={k:float(v.sqrt()) for k,v in zip(['linear','angular','joint'],velocity[name])};entry[name+'_velocity_error_integral_p95_m']=float(torch.quantile(accum[name].norm(dim=-1),.95))
    results[str(h)]=entry
  rows.append({'trace':row['trace'],'split':row['split'],'source_sha256':digest,'horizons':results})
 assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items());out['models'].append({'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'weights_unchanged':True,'rows':rows});(a.output/'partial.json').write_text(json.dumps(out,indent=2));print('AUDITED',checkpoint,len(rows),flush=True)
 del model,env,before;torch.cuda.empty_cache()
(a.output/'audit.json').write_text(json.dumps(out,indent=2))
