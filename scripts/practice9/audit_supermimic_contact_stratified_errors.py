"""Fit-only frozen rollout errors aligned to recorded post-step native contacts."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from supermimic_geometry_model import load_geometry_artifact
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--collection',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,action='append',required=True)
p.add_argument('--reference-report',type=Path,action='append',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();assert len(a.checkpoint)==len(a.reference_report)
a.output.mkdir(parents=True)
manifest=json.loads((a.collection/'results.json').read_text());traces=[];sources=[]
for row in manifest['rows']:
 if row['split']!='fit':continue
 f=a.collection/row['trace'];assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256']
 with np.load(f,allow_pickle=False) as z:d={k:z[k].copy() for k in ['state','action','context','next_state']}
 cfile=a.collection/row['native_contact_file'];contact=json.loads(cfile.read_text());assert contact['state_read_unchanged']
 samples=contact['samples'];assert len(samples)==len(d['state'])==row['steps']
 assert [s['step'] for s in samples]==list(range(1,len(samples)+1))
 assert np.array_equal(d['next_state'][:-1],d['state'][1:])
 masks=[];points=[]
 for s in samples:
  feet=[s['feet'][side] for side in ['left','right']]
  pt=[int(x['loaded_ground_points']) for x in feet]
  assert all(x>=0 for x in pt)
  masks.append(int(pt[0]>0)+2*int(pt[1]>0));points.append(pt)
 traces.append((row,d,np.array(masks),np.array(points)))
 sources.append(dict(trace=row['trace'],sha256=row['sha256'],contact_sha256=hashlib.sha256(cfile.read_bytes()).hexdigest()))
assert len(traces)==16 and len({r['source_family'] for r,_,_,_ in traces})==4
rows=[];models=[]
for model_i,(path,reference) in enumerate(zip(a.checkpoint,a.reference_report)):
 sha=hashlib.sha256(path.read_bytes()).hexdigest();c=torch.load(path,map_location='cpu',weights_only=False)
 m,predict=load_geometry_artifact(c);m.eval()
 ref=json.loads(reference.read_text());assert ref['checkpoint_sha256']==sha
 refrows={x['trace']:x for x in ref['rows'] if x['split']=='fit'}
 for row,d,masks,points in traces:
  tensors={k:torch.tensor(v,device='cuda:0') for k,v in d.items()}
  for h in [1,8,32]:
   n=len(masks)-h+1
   if n<=0:continue
   with torch.no_grad():
    pred=tensors['state'][:n]
    for offset in range(h):pred=predict(pred,tensors['action'][offset:offset+n],tensors['context'][offset:offset+n])
    target=tensors['next_state'][h-1:];assert torch.isfinite(pred).all()
    root=torch.linalg.vector_norm(pred[:,:3]-target[:,:3],dim=-1)
    omega=(pred[:,10:13]-target[:,10:13]).square().mean(-1)
    linear=(pred[:,7:10]-target[:,7:10]).square().mean(-1)
    jointv=(pred[:,43:73]-target[:,43:73]).square().mean(-1)
    root_np=root.cpu().numpy();om=omega.cpu().numpy();lv=linear.cpu().numpy();jv=jointv.cpu().numpy()
   expected=refrows[row['trace']]['world_model'][str(h)]
   measured=[float(torch.quantile(root,.95)),float(omega.mean().sqrt())]
   old=[expected['root_position_error_p95_m'],expected['root_angular_velocity_rmse_rad_s']]
   assert np.allclose(measured,old,rtol=1e-5,atol=1e-6),(row['trace'],h,measured,old)
   labels=[];changes=[]
   for start in range(n):
    # Post-step mask[i] corresponds next_state[i]. Start0 lacks state contact.
    seq=masks[max(0,start-1):start+h];change=int(np.count_nonzero(seq[1:]!=seq[:-1]))
    changes.append(change)
    if start==0:label='initial_contact_unknown'
    elif change:label='support_mask_transition'
    else:label=['stable_no_loaded_feet','stable_left_only','stable_right_only','stable_both_feet'][int(seq[0])]
    labels.append(label)
   np.savez_compressed(a.output/f'model{model_i}_{row["trace"][:-4]}_h{h}.npz',start=np.arange(n),root_error_m=root_np,omega_mse=om,linear_mse=lv,joint_velocity_mse=jv,support_mask_endpoint=masks[h-1:],loaded_points_endpoint=points[h-1:],mask_changes=np.array(changes),group=np.array(labels))
   for label in sorted(set(labels)):
    idx=np.array(labels)==label
    rows.append(dict(model=model_i,trace=row['trace'],id=row['id'],source_family=row['source_family'],horizon=h,group=label,windows=int(idx.sum()),root_mean_m=float(root_np[idx].mean()),root_p95_m=float(np.quantile(root_np[idx],.95)),omega_rmse=float(np.sqrt(om[idx].mean())),linear_rmse=float(np.sqrt(lv[idx].mean())),joint_velocity_rmse=float(np.sqrt(jv[idx].mean()))))
  print(json.dumps(dict(model=model_i,trace=row['trace'],completed=True)),flush=True)
 for k,v in m.state_dict().items():assert torch.equal(v.cpu(),c['world_model'][k])
 assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
 models.append(dict(path=str(path),sha256=sha,reference_report=str(reference)))
 del m,predict,c,tensors
report=dict(models=models,sources=sources,rows=rows,fit_traces=16,fit_families=4,validation_arrays_loaded=False,optimizer_updates=0,weights_unchanged=True,reference_metrics_reproduced=True,scope='All fit windows, no reset crossing/padding. Native contacts recorded after each step label next_state, first start unknown. Existing loaded force threshold1e-4. Overlapping windows correlated; mask transitions do not measure full contact manifold changes and do not establish causality. No policy/physics acceptance.')
(a.output/'audit.json').write_text(json.dumps(report,indent=2))
print('COMPLETE',flush=True)
