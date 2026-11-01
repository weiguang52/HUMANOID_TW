"""Fit-only teacher-forced versus recurrent velocity errors; no optimization."""
import argparse,hashlib,json
from pathlib import Path
import torch
from supermimic_geometry_model import load_geometry_artifact
from supermimic_motion_windows import load_motion_fit_windows
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--pair',default='mixed_horizon_fit_v1')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir()
r=a.root;collection=r/'motion_pd_collection_cache_v2/full'
mapping=json.loads((collection/'control_mapping.json').read_text())
w,meta=load_motion_fit_windows(collection,collection.parent/'full_audit.json',mapping,window=32)
assert len(w['state'])==337
reports=[]
for arm in ['control','candidate']:
 path=r/a.pair/arm/'fit/world_model_calibrated.pt'
 c=torch.load(path,map_location='cpu',weights_only=False)
 m,predict=load_geometry_artifact(c);m.eval()
 for param in m.parameters():param.requires_grad_(False)
 for param in m.net.parameters():param.requires_grad_(True)
 params=list(m.net.named_parameters())
 sums={mode:torch.zeros(32,3,device='cuda:0',dtype=torch.float64) for mode in ['teacher_forced','recurrent']}
 count=0
 with torch.no_grad():
  for start in range(0,337,16):
   ids=slice(start,min(start+16,337));pred=w['state'][ids];n=len(pred);count+=n
   for j in range(32):
    true=w['state'][ids] if j==0 else w['next_state'][ids,j-1]
    teacher=predict(true,w['action'][ids,j],w['context'][ids,j])
    pred=predict(pred,w['action'][ids,j],w['context'][ids,j])
    for mode,result in [('teacher_forced',teacher),('recurrent',pred)]:
     e=result-w['next_state'][ids,j]
     for k,sl in enumerate([slice(7,10),slice(10,13),slice(43,73)]):sums[mode][j,k]+=e[:,sl].double().square().mean(1).sum()
 gradients=[]
 for batch in range(4):
  gen=torch.Generator(device='cuda:0').manual_seed(1042+100*batch)
  ids=torch.randint(337,(4,),device='cuda:0',generator=gen)
  pred=w['state'][ids];losses={};loss=0
  for j in range(32):
   pred=predict(pred,w['action'][ids,j],w['context'][ids,j])
   e=m.state_norm.normalize(pred)-m.state_norm.normalize(w['next_state'][ids,j])
   loss=loss+sum(e[:,sl].square().mean() for sl in [slice(7,10),slice(10,13),slice(43,73)])/3
   if j in [7,31]:losses[str(j+1)]=loss/(j+1)
  gs={};norms={}
  for label,value in losses.items():
   raw=torch.autograd.grad(value,[v for _,v in params],retain_graph=True)
   flat=[]
   for (name,_),g in zip(params,raw):
    g=g.detach().clone()
    if name=='0.weight':g[:,:2]=0
    flat.append(g.flatten())
   gs[label]=torch.cat(flat);assert torch.isfinite(gs[label]).all()
   norms[label]=float(gs[label].norm())
  cosine=float(torch.dot(gs['8'],gs['32'])/(gs['8'].norm()*gs['32'].norm()))
  gradients.append(dict(batch=batch,indices=ids.tolist(),velocity_losses={k:float(v) for k,v in losses.items()},gradient_norms=norms,cosine_h8_h32=cosine))
  del gs,raw,flat,losses,loss,pred
 for k,v in m.state_dict().items():assert torch.equal(v.cpu(),c['world_model'][k]),k
 row=dict(arm=arm,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),windows=count,velocity_rmse_by_step={k:(v/count).sqrt().cpu().tolist() for k,v in sums.items()},gradient_batches=gradients)
 reports.append(row);print(json.dumps(row),flush=True)
 del m,predict,c
out=dict(reports=reports,fit_input=meta,weights_unchanged=True,validation_arrays_loaded=False,scope='All337 fit H32 windows only; velocity units m/s,rad/s,rad/s. Teacher forcing resets to true state each step; recurrent keeps predictions. Four fixed matched batches compare normalized velocity-loss H8/H32 gradients only, not full training objective. No optimizer, coefficient selection, policy or acceptance claim.')
(a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
