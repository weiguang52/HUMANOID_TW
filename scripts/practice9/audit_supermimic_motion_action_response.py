"""Frozen dynamic action response comparison; no fitting or policy export."""
import argparse,hashlib,json,datetime,ast,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--integrity',type=Path,required=True);p.add_argument('--checkpoint',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
r=a.collection;proof=json.loads(a.integrity.read_text());assert proof['passed'] and not proof['smoke'];assert hashlib.sha256((r/'plan.json').read_bytes()).hexdigest()==proof['plan_sha256'];hashes={x['file']:x['sha256'] for x in proof['files']};groups={}
for row in json.loads((r/'results.json').read_text())['rows']:
 path=r/row['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[row['file']]
 if row['level']==0:continue
 with np.load(path) as z:d={k:z[k].copy() for k in ('state','action','context','next_state')}
 key=(row['source_trace'],row['index'],row['level']);groups.setdefault(key,{'split':row['split'],'assignments':{}})['assignments'][row['assignment']]=d
assert len(groups)==192
loader=Path('scripts/practice9/audit_supermimic_calibration_model.py');tree=ast.parse(loader.read_text());cut=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='trainer' for x in n.targets));program=compile(ast.fix_missing_locations(ast.Module(body=tree.body[:cut],type_ignores=[])),str(loader),'exec')
out={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'loader_sha256':hashlib.sha256(loader.read_bytes()).hexdigest(),'integrity_sha256':hashlib.sha256(a.integrity.read_bytes()).hexdigest(),'models':[],'policy_acceptance':False,'scope':'Fixed dynamic finite-action response; source splits retained; no fitting.'}
for checkpoint in a.checkpoint:
 sys.argv=[str(loader),'--collection',str(r),'--checkpoint',str(checkpoint),'--output',str(a.output/'unused_loader_report.json')];env={'__name__':'__main__','__file__':str(loader)};exec(program,env);model=env['m'];before={k:v.clone() for k,v in model.state_dict().items()};norm=next(x.state_norm for x in model.modules() if hasattr(x,'action_norm'));std=(norm.var+norm.eps).sqrt();std=torch.where(norm.frozen_mask,torch.ones_like(std),std);rows=[]
 with torch.no_grad():
  for (trace,index,eps),entry in groups.items():
   aa,bb=[entry['assignments'][i] for i in (0,1)];arrays={k:np.stack([aa[k][::2],aa[k][1::2],bb[k][::2],bb[k][1::2]],axis=1) for k in aa};v={k:torch.as_tensor(x,device='cuda') for k,x in arrays.items()};pred=model(v['state'].reshape(-1,73),v['action'].reshape(-1,30),v['context'].reshape(-1,34)).reshape(30,4,73)
   def derivative(x):return (x[:,1].double()-x[:,0].double()+x[:,2].double()-x[:,3].double())/(4*eps)
   actual=derivative(v['next_state']);estimated=derivative(pred);assert torch.isfinite(estimated).all();metrics={}
   for name,sl in [('root_linear',slice(7,10)),('root_angular',slice(10,13)),('joint_velocity',slice(43,73))]:
    error=(estimated[:,sl]-actual[:,sl])/std[sl];metrics[name]={'normalized_rmse':float(error.square().mean().sqrt()),'physical_rmse':float((estimated[:,sl]-actual[:,sl]).square().mean().sqrt())}
   rows.append({'trace':trace,'index':index,'epsilon':eps,'split':entry['split'],'metrics':metrics})
 assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items());report={'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'rows':rows,'weights_unchanged':True};out['models'].append(report);(a.output/'partial.json').write_text(json.dumps(out,indent=2));print('AUDITED',report['sha256'],len(rows),flush=True)
 del before,model,env;torch.cuda.empty_cache()
(a.output/'audit.json').write_text(json.dumps(out,indent=2))
