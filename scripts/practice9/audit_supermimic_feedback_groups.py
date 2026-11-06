"""Frozen state-feedback oracle ablations; no optimization or deployable controller."""
import ast
from pathlib import Path
source=Path('scripts/practice9/audit_supermimic_rollout_exposure.py')
tree=ast.parse(source.read_text());cut=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='checkpoint')
exec(compile(ast.fix_missing_locations(ast.Module(body=tree.body[:cut],type_ignores=[])),str(source),'exec'))
out['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();out['scope']='H32 fixed native windows; replace only selected model-input state groups with recorded truth, integrate predicted velocities on each branch own state. Oracle diagnostic, not deployable, no fitting or acceptance.'
groups={'none':[], 'root_pose':[(0,7)], 'root_velocity':[(7,13)], 'joint_position':[(13,43)], 'joint_velocity':[(43,73)], 'all':[(0,73)]}
for checkpoint in a.checkpoint:
 sys.argv=[str(loader),'--collection',str(a.collection),'--checkpoint',str(checkpoint),'--output',str(a.output/'unused.json')];env={'__name__':'__main__','__file__':str(loader)};exec(program,env);model=env['m'];before={k:v.clone() for k,v in model.state_dict().items()};integration=next(x for x in model.modules() if isinstance(x,PoseVelocityModel));rows=[]
 for row in json.loads((a.collection/'results.json').read_text())['rows']:
  path=(a.collection/row['trace']).resolve();assert path.is_relative_to(a.collection.resolve());digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==(row['sha256'] if 'sha256' in row else manifest['traces'][row['trace']])
  with np.load(path) as z:data={k:torch.as_tensor(z[k].copy(),device='cuda') for k in ['state','action','context','next_state']}
  n=len(data['state']);count=n-31;results={}
  if count>0:
   with torch.no_grad():
    for name,slices in groups.items():
     state=data['state'][:count].clone()
     for j in range(32):
      real=data['state'][j:j+count];hybrid=state.clone()
      for lo,hi in slices:hybrid[:,lo:hi]=real[:,lo:hi]
      act=data['action'][j:j+count];ctx=data['context'][j:j+count]
      endpoint=model(hybrid.double(),act.double(),ctx.double())
      following=integration.integrate_prediction(state.double(),endpoint).to(state.dtype)
      if name=='none':assert torch.equal(following,model(state,act,ctx)),('baseline mismatch',row['trace'],j)
      if j==0:assert torch.equal(following,model(real,act,ctx)),('initial group mismatch',name,row['trace'])
      state=following
     results[name]=metric(state,data['next_state'][31:31+count])
  rows.append({'trace':row['trace'],'split':row['split'],'source_sha256':digest,'windows':max(0,count),'groups':results})
 assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items());out['models'].append({'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'weights_unchanged':True,'baseline_every_step_bitexact':True,'rows':rows});(a.output/'partial.json').write_text(json.dumps(out,indent=2));print('AUDITED',checkpoint,len(rows),flush=True)
 del model,env,before;torch.cuda.empty_cache()
(a.output/'audit.json').write_text(json.dumps(out,indent=2))
