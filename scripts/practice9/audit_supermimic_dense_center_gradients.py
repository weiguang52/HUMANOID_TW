"""Fit-only frozen dense-center gradient scope; no optimizer or policy claim."""
import ast,json,sys,hashlib,datetime
from pathlib import Path
import torch
source=Path('scripts/practice9/audit_supermimic_dense_center_coverage.py')
tree=ast.parse(source.read_text())
cut=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='out' for t in n.targets))
E={'__name__':'__main__','__file__':str(source)}
exec(compile(ast.fix_missing_locations(ast.Module(body=tree.body[:cut],type_ignores=[])),str(source),'exec'),E)
a=E['a'];groups=E['groups'];loader=E['loader'];program=E['program']
entries=[v['data'] for v in groups.values() if v['split']=='fit'];assert len(entries)==895
batch={k:torch.cat([torch.as_tensor(x[k],device='cuda') for x in entries]) for k in entries[0]}
out={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'models':[],'fit_states':895,'validation_arrays_used_for_gradients':False,'optimizer_steps':0,'scope':'Fixed16 fit-only batches32, Euclidean local gradient evidence, no Adam/training or policy guarantee'}
for checkpoint in a.checkpoint:
 sys.argv=[str(loader),'--collection',str(a.collection),'--checkpoint',str(checkpoint),'--output',str(a.output/'unused.json')]
 env={'__name__':'__main__','__file__':str(loader)};exec(program,env);model=env['m'];before={k:v.clone() for k,v in model.state_dict().items()}
 raw=next(x for x in model.modules() if hasattr(x,'action_norm'));params=list(raw.net.parameters());assert len(params)==8
 for p in params:p.requires_grad_(True)
 norm=next(x.state_norm for x in model.modules() if hasattr(x,'action_norm'));std=(norm.var+norm.eps).sqrt();std=torch.where(norm.frozen_mask,torch.ones_like(std),std)
 generator=torch.Generator(device='cuda').manual_seed(20261106);rows=[]
 for i in range(16):
  idx=torch.randint(895,(32,),device='cuda',generator=generator);pred=model(batch['state'][idx],batch['action'][idx],batch['context'][idx]);error=(pred-batch['next_state'][idx])/std
  loss={k:.5*error[:,sl].square().mean() for k,sl in [('linear',slice(7,10)),('angular',slice(10,13)),('joint_velocity',slice(43,73))]}
  grads={k:torch.autograd.grad(v,params,retain_graph=True) for k,v in loss.items()};row={'batch':i,'indices':idx.cpu().tolist(),'loss':{k:float(v) for k,v in loss.items()},'pairs':{}}
  for k in ['angular','joint_velocity']:
   x=grads['linear'];y=grads[k];dot=sum(float((u*v).sum()) for u,v in zip(x,y));nx=sum(float(u.square().sum()) for u in x);ny=sum(float(v.square().sum()) for v in y)
   row['pairs'][k]={'dot':dot,'cosine':dot/max((nx*ny)**.5,1e-30),'linear_norm':nx**.5,'other_norm':ny**.5}
  rows.append(row)
 assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
 report={'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'weights_unchanged':True,'rows':rows};out['models'].append(report)
 (a.output/'partial.json').write_text(json.dumps(out,indent=2));print('AUDITED',report['sha256'],flush=True)
 del model,env,before,grads;torch.cuda.empty_cache()
(a.output/'audit.json').write_text(json.dumps(out,indent=2))
