"""Exact frozen root-position error ledger under recorded commands; no fitting."""
import ast
from pathlib import Path
source=Path('scripts/practice9/audit_supermimic_rollout_exposure.py')
tree=ast.parse(source.read_text());cut=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='checkpoint')
exec(compile(ast.fix_missing_locations(ast.Module(body=tree.body[:cut],type_ignores=[])),str(source),'exec'))
out['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();out['scope']='All valid fixed H32 native windows; algebraic decomposition of world-position drift into native integration defect, real-state one-step bias, predicted-state input response, and old-velocity blend feedback. No optimization or physical stability claim.'
for checkpoint in a.checkpoint:
 sys.argv=[str(loader),'--collection',str(a.collection),'--checkpoint',str(checkpoint),'--output',str(a.output/'unused.json')];env={'__name__':'__main__','__file__':str(loader)};exec(program,env);model=env['m'];before={k:v.clone() for k,v in model.state_dict().items()};integration=next(x for x in model.modules() if isinstance(x,PoseVelocityModel));alpha=integration.alpha['translation'];dt=integration.dt;rows=[]
 for row in json.loads((a.collection/'results.json').read_text())['rows']:
  path=(a.collection/row['trace']).resolve();assert path.is_relative_to(a.collection.resolve());digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==row['sha256']
  with np.load(path) as z:data={k:torch.as_tensor(z[k].copy(),device='cuda') for k in ['state','action','context','next_state']}
  count=len(data['state'])-31;entry={}
  if count>0:
   state=data['state'][:count].clone();terms={k:torch.zeros(count,3,device='cuda',dtype=torch.float64) for k in ['native_integration','one_step_bias','input_feedback','old_velocity_feedback','rounding']}
   with torch.no_grad():
    for j in range(32):
     real=data['state'][j:j+count];target=data['next_state'][j:j+count];act=data['action'][j:j+count];ctx=data['context'][j:j+count];forced=model(real,act,ctx);following=model(state,act,ctx)
     terms['native_integration']+=real[:,:3].double()+dt*((1-alpha)*real[:,7:10].double()+alpha*target[:,7:10].double())-target[:,:3].double()
     terms['one_step_bias']+=dt*alpha*(forced[:,7:10].double()-target[:,7:10].double())
     terms['input_feedback']+=dt*alpha*(following[:,7:10].double()-forced[:,7:10].double())
     terms['old_velocity_feedback']+=dt*(1-alpha)*(state[:,7:10].double()-real[:,7:10].double())
     terms['rounding']+=following[:,:3].double()-state[:,:3].double()-dt*((1-alpha)*state[:,7:10].double()+alpha*following[:,7:10].double())
     state=following
    error=state[:,:3].double()-data['next_state'][31:31+count,:3].double();reconstructed=sum(terms.values());residual=float((error-reconstructed).abs().max());assert residual<1e-12
    entry={'metric':metric(state,data['next_state'][31:31+count]),'ledger_residual_max_m':residual,'components':{k:{'p95_m':float(torch.quantile(v.norm(dim=-1),.95)),'rms_m':float(v.square().mean().sqrt()),'mean_signed_xyz_m':v.mean(0).cpu().tolist(),'projection_on_final_error_mean':float(((v*error).sum(-1)/(error.square().sum(-1)+1e-30)).mean())} for k,v in terms.items()}}
  rows.append({'trace':row['trace'],'split':row['split'],'source_sha256':digest,'windows':max(0,count),**entry})
 assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items());out['models'].append({'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'alpha_translation':alpha,'weights_unchanged':True,'rows':rows});(a.output/'partial.json').write_text(json.dumps(out,indent=2));print('AUDITED',checkpoint,len(rows),flush=True)
 del model,env,before;torch.cuda.empty_cache()
(a.output/'audit.json').write_text(json.dumps(out,indent=2))
