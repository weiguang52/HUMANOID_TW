"""Frozen fit-only one-step input transplantation along unmodified rollouts."""
import argparse,hashlib,json
from pathlib import Path
import torch
from supermimic_geometry_model import load_geometry_artifact
from supermimic_motion_windows import load_motion_fit_windows
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir()
r=a.root;collection=r/'motion_pd_collection_cache_v2/full'
mapping=json.loads((collection/'control_mapping.json').read_text())
w,meta=load_motion_fit_windows(collection,collection.parent/'full_audit.json',mapping,window=32)
groups={'root_pose':slice(0,7),'root_velocity':slice(7,13),'joint_position':slice(13,43),'joint_velocity':slice(43,73)}
labels=['recurrent','teacher_forced',*groups];metrics=[slice(7,10),slice(10,13),slice(43,73)]
reports=[]
for arm in ['control','candidate']:
 path=r/'motion_h1_anchor_fit_v1'/arm/'fit/world_model_calibrated.pt'
 c=torch.load(path,map_location='cpu',weights_only=False);m,predict=load_geometry_artifact(c);m.eval()
 for v in m.parameters():v.requires_grad_(False)
 squared=torch.zeros(6,32,3,device='cuda:0',dtype=torch.float64);signed=torch.zeros_like(squared)
 first_max=0.;count=0
 with torch.no_grad():
  for start in range(0,337,16):
   ids=slice(start,min(start+16,337));pred=w['state'][ids];n=len(pred);count+=n
   for j in range(32):
    true=w['state'][ids] if j==0 else w['next_state'][ids,j-1]
    inputs=[pred,true]
    for sl in groups.values():
     mixed=pred.clone();mixed[:,sl]=true[:,sl];inputs.append(mixed)
    assert all(torch.equal(x,true) for x in inputs) if j==0 else True
    actual=predict(torch.cat(inputs),w['action'][ids,j].repeat(6,1),w['context'][ids,j].repeat(6,1))
    assert torch.isfinite(actual).all()
    parts=actual.split(n);target=w['next_state'][ids,j]
    if j==0:first_max=max(first_max,max(float((x-parts[0]).abs().max()) for x in parts))
    for k,result in enumerate(parts):
     e=(result-target).double()
     for z,sl in enumerate(metrics):
      squared[k,j,z]+=e[:,sl].square().mean(1).sum();signed[k,j,z]+=e[:,sl].mean(1).sum()
    pred=parts[0]
 assert count==337 and first_max<=1e-5
 for key,v in m.state_dict().items():assert torch.equal(v.cpu(),c['world_model'][key]),key
 row=dict(arm=arm,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),windows=count,steps=32,input_modes=labels,first_step_max_difference=first_max,velocity_rmse={label:(squared[i]/count).sqrt().cpu().tolist() for i,label in enumerate(labels)},velocity_signed_mean={label:(signed[i]/count).cpu().tolist() for i,label in enumerate(labels)})
 reports.append(row);print(json.dumps(dict(arm=arm,step32={label:row['velocity_rmse'][label][31] for label in labels})),flush=True)
 del m,predict,c
out=dict(reports=reports,fit_input=meta,weights_unchanged=True,validation_arrays_loaded=False,scope='All337 fit H32 windows. Velocity groups linear/angular/joint units m/s,rad/s,rad/s. One-group true-input substitutions are evaluated on the SAME unmodified free trajectory and never fed back. Hybrid inputs can be physically inconsistent; sensitivity only, no physical causal, stability, policy or acceptance claim.')
(a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
