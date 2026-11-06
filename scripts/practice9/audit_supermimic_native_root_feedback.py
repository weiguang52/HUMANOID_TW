"""Frozen root-velocity finite response versus replay-verified native branches."""
import ast
from pathlib import Path
from supermimic_motion_root_response import load_motion_root_response
source=Path('scripts/practice9/audit_supermimic_rollout_exposure.py')
tree=ast.parse(source.read_text());cut=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='checkpoint')
exec(compile(ast.fix_missing_locations(ast.Module(body=tree.body[:cut],type_ignores=[])),str(source),'exec'))
out['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();out['scope']='All 48 fit and 48 validation native anchors, two fixed perturbation magnitudes, six velocity axes, swapped repeats; finite contact responses, not global Jacobians or physical stability proof. No fitting.'
def response(pred,den):return (pred[:,1,7:13]-pred[:,0,7:13]+pred[:,2,7:13]-pred[:,3,7:13])/den
for checkpoint in a.checkpoint:
 sys.argv=[str(loader),'--collection',str(a.collection),'--checkpoint',str(checkpoint),'--output',str(a.output/'unused.json')];env={'__name__':'__main__','__file__':str(loader)};exec(program,env);model=env['m'];before={k:v.clone() for k,v in model.state_dict().items()};rows=[]
 for split in ['fit','validation']:
  data,den,axes,meta=load_motion_root_response(a.collection,env['cfg'],split);preds=[]
  with torch.no_grad():
   for start in range(0,len(den),32):preds.append(model(*[data[k][start:start+32].reshape(-1,w).double() for k,w in [('state',73),('action',30),('context',34)]]).reshape(-1,4,73))
   pred=torch.cat(preds);target=data['next_state'].double();den=den.double();rp=response(pred,den);rt=response(target,den)
   raw=next(x for x in model.modules() if hasattr(x,'state_norm'));std=(raw.state_norm.var[7:13]+raw.state_norm.eps).sqrt().double();std=torch.where(raw.state_norm.frozen_mask[7:13],torch.ones_like(std),std)
   norm=(rp-rt)*std[axes,None]/std[None,:]
   gains={}
   for name,r in [('model',rp),('native',rt)]:
    matrix=r.reshape(48,2,6,6).transpose(-1,-2);normalized=matrix*std[None,None,None,:]/std[None,None,:,None]
    radius=torch.linalg.eigvals(matrix).abs().amax(-1);singular=torch.linalg.svdvals(normalized)[...,0]
    gains[name]={'spectral_radius':radius.cpu().tolist(),'normalized_max_singular':singular.cpu().tolist(),'median_spectral_radius':float(radius.median()),'median_max_singular':float(singular.median())}
   err=pred-target;rows.append({'split':split,'conditions':len(den),'input_evidence':meta,'normalized_response_mse':float(norm.square().mean()),'absolute_velocity_rmse':{k:float(err[:,:,sl].square().mean().sqrt()) for k,sl in [('linear',slice(7,10)),('angular',slice(10,13)),('joint',slice(43,73))]},'response_gains':gains})
 assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items());out['models'].append({'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'weights_unchanged':True,'rows':rows});(a.output/'partial.json').write_text(json.dumps(out,indent=2));print('AUDITED',checkpoint,flush=True)
 del model,env,before;torch.cuda.empty_cache()
(a.output/'audit.json').write_text(json.dumps(out,indent=2))
