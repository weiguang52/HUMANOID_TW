"""Frozen endpoint and finite-response errors at every acquired velocity level."""
import ast
from pathlib import Path
from supermimic_wide_root_response import load_wide_root_response,LEVELS
source=Path('scripts/practice9/audit_supermimic_rollout_exposure.py');tree=ast.parse(source.read_text());cut=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='checkpoint')
exec(compile(ast.fix_missing_locations(ast.Module(body=tree.body[:cut],type_ignores=[])),str(source),'exec'))
out['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();out['scope']='All verified wide native root branches, source-isolated fit/validation and all levels; no terminal filtering, optimizer or policy acceptance. Large contact finite responses are not infinitesimal Jacobians.'
def derivative(v,den,sl):return (v[:,1,sl]-v[:,0,sl]+v[:,2,sl]-v[:,3,sl])/den
old_path=a.collection.parent.parent/'hidden_moment_native_root_feedback_v1/audit/audit.json';old=json.loads(old_path.read_text());references={m['sha256']:m for m in old['models']}
for checkpoint in a.checkpoint:
 sys.argv=[str(loader),'--collection',str(a.collection),'--checkpoint',str(checkpoint),'--output',str(a.output/'unused.json')];env={'__name__':'__main__','__file__':str(loader)};exec(program,env);model=env['m'];before={k:v.clone() for k,v in model.state_dict().items()};rows=[];sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 for split in ['fit','validation']:
  data,den,axes,levels,meta=load_wide_root_response(a.collection,env['cfg'],split);preds=[]
  with torch.no_grad():
   for start in range(0,len(den),32):preds.append(model(*[data[k][start:start+32].reshape(-1,w).double() for k,w in [('state',73),('action',30),('context',34)]]).reshape(-1,4,73))
   pred=torch.cat(preds);target=data['next_state'].double();den=den.double();raw=next(x for x in model.modules() if hasattr(x,'state_norm'));std=(raw.state_norm.var+raw.state_norm.eps).sqrt().double();std=torch.where(raw.state_norm.frozen_mask,torch.ones_like(std),std)
   response={k:(derivative(pred,den,sl)-derivative(target,den,sl))*std[axes+7,None]/std[None,sl] for k,sl in [('root',slice(7,13)),('joint',slice(43,73))]}
   err=pred-target
   def stats(mask):
    return {'conditions':int(mask.sum()),'normalized_response_mse':{k:float(v[mask].square().mean()) for k,v in response.items()},'absolute_velocity_rmse':{k:float(err[mask,:,sl].square().mean().sqrt()) for k,sl in [('linear',slice(7,10)),('angular',slice(10,13)),('joint',slice(43,73))]},'terminated_endpoint_fraction':float(data['terminated'][mask].double().mean()),'truncated_endpoint_fraction':float(data['truncated'][mask].double().mean())}
   base=stats(levels<=2);ref=next(x for x in references[sha]['rows'] if x['split']==split);diff=abs(base['normalized_response_mse']['root']-ref['normalized_response_mse']);assert diff<1e-10,(split,diff)
   for k,v in base['absolute_velocity_rmse'].items():assert abs(v-ref['absolute_velocity_rmse'][k])<1e-10
   rows.append({'split':split,'input_evidence':meta,'original_levels_regression_max_response_difference':diff,'levels':{str(level):stats(levels==level) for level in LEVELS},'all':stats(levels>0)})
 assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items());assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==sha;out['models'].append({'checkpoint':str(checkpoint),'sha256':sha,'weights_unchanged':True,'rows':rows});(a.output/'partial.json').write_text(json.dumps(out,indent=2));print('AUDITED',checkpoint,flush=True)
 del model,env,before;torch.cuda.empty_cache()
(a.output/'audit.json').write_text(json.dumps(out,indent=2))
