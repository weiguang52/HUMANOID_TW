"""Frozen declared-precision predictions versus fixed fit native pose probes."""
import argparse,hashlib,json,datetime
from pathlib import Path
import numpy as np,torch
from supermimic_geometry_model import load_geometry_artifact
from supermimic_precision_contract import validate
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--queries',type=Path,required=True);p.add_argument('--collection',type=Path,required=True);p.add_argument('--checkpoint',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();plan=json.loads((a.queries/'query_plan.json').read_text());assert plan['contract'] in ['fit_dynamic_root_pose_probe_v1','fit_dynamic_root_pose_full_v1'] and plan['queries']==(336 if plan['contract']=='fit_dynamic_root_pose_probe_v1' else 1200) and not plan['validation_arrays_loaded']
audit=json.loads((a.collection/'independent_audit.json').read_text());assert audit['passed'] and not audit['smoke'] and audit['records']==plan['queries']*4 and audit['fit_reference_initialization_verified']
capture=json.loads((a.collection/'plan.json').read_text());assert capture['query_plan_sha256']==hashlib.sha256((a.queries/'query_plan.json').read_bytes()).hexdigest()
query_file=a.queries/'queries.npz';assert capture['queries_sha256']==plan['query_sha256']==hashlib.sha256(query_file.read_bytes()).hexdigest()
with np.load(query_file,allow_pickle=False) as z:queries={k:z[k].copy() for k in z.files}
pose_write_roundoff=[];height_roundoff=[];rotation_roundoff=[]
for row in plan['rows']:
 i=row['query'];q=queries['query_state'][i];base=queries['source_state'][i];axis=row['perturb_axis'];sign=row['sign'];level=row.get('level',0 if axis==-1 else 1)
 assert np.array_equal(q[7:],base[7:]) and row['is_native_anchor']==(axis==-1)
 if axis==-1:assert sign==0 and np.array_equal(q,base)
 elif axis in [0,1,2]:
  assert row['delta']==.0005*level and sign in [-1,1] and np.array_equal(np.delete(q[:3],axis),np.delete(base[:3],axis)) and np.array_equal(q[3:7],base[3:7])
  error=abs(float(q[axis])-float(base[axis])-sign*.0005*level);assert error<=4*np.finfo(np.float32).eps;pose_write_roundoff.append(error)
  if axis==2:height_roundoff.append(error)
 else:
  assert axis in [3,4,5] and row['delta']==.001*level and sign in [-1,1] and np.array_equal(q[:3],base[:3])
  aa=q[3:7].astype(float);bb=base[3:7].astype(float);aa/=np.linalg.norm(aa);bb/=np.linalg.norm(bb)
  w=aa[0]*bb[0]+np.dot(aa[1:],bb[1:]);v=bb[0]*aa[1:]-aa[0]*bb[1:]-np.cross(aa[1:],bb[1:])
  if w<0:w=-w;v=-v
  length=np.linalg.norm(v);rotvec=v*(2*np.arctan2(length,w)/length) if length else v
  expected=np.zeros(3);expected[axis-3]=sign*.001*level
  error=float(np.max(np.abs(rotvec-expected)));assert error<=4*np.finfo(np.float32).eps;rotation_roundoff.append(error)

f=a.collection/'transitions.npz';assert hashlib.sha256(f.read_bytes()).hexdigest()==audit['transitions_sha256']
with np.load(f,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
lookup={(int(q),bool(w),int(rep)):i for i,(q,w,rep) in enumerate(zip(d['query'],d['warm_source'],d['repeat']))};assert len(lookup)==plan['queries']*4
query_map={(x['anchor_query'],x['perturb_axis'],x.get('level',0 if x['is_native_anchor'] else 1),x['sign']):x['query'] for x in plan['rows']};assert len(query_map)==plan['queries']
anchors=sorted({x['anchor_query'] for x in plan['rows']});assert len(anchors)==48
rows=[];conditions=[];models=[];reference_std=None
for model_i,path in enumerate(a.checkpoint):
 sha=hashlib.sha256(path.read_bytes()).hexdigest();c=torch.load(path,map_location='cpu',weights_only=False);assert validate(c)
 raw,predict=load_geometry_artifact(c);raw.eval()
 std=(raw.state_norm.var[7:13]+raw.state_norm.eps).sqrt();std=torch.where(raw.state_norm.frozen_mask[7:13],torch.ones_like(std),std).cpu().numpy().astype(float)
 if reference_std is None:reference_std=std
 else:assert np.array_equal(reference_std,std)
 with torch.no_grad():prediction=predict(*[torch.tensor(d[k],device='cuda:0') for k in ['state','action','context']]).cpu().numpy().astype(float)
 assert np.isfinite(prediction).all()
 for warm in [True,False]:
  for axis,level in [(axis,level) for axis in plan.get('axes',[2,3,4]) for level in plan.get('levels',[1])]:
   native_values=[];pred_values=[];scaled_errors=[];warm_changes=[]
   for anchor in anchors:
    plus=query_map[(anchor,axis,level,1)];minus=query_map[(anchor,axis,level,-1)];delta=plan['rows'][plus]['delta'];assert delta==plan['rows'][minus]['delta']
    ip,im=lookup[(plus,warm,0)],lookup[(minus,warm,0)]
    native=(d['next_state'][ip,7:13].astype(float)-d['next_state'][im,7:13].astype(float))/(2*delta)
    pred=(prediction[ip,7:13]-prediction[im,7:13])/(2*delta)
    jp,jm=lookup[(plus,not warm,0)],lookup[(minus,not warm,0)]
    other=(d['next_state'][jp,7:13].astype(float)-d['next_state'][jm,7:13].astype(float))/(2*delta)
    scaled=(pred-native)*(delta/level)/std;warm_scaled=(native-other)*(delta/level)/std
    native_values.append(native);pred_values.append(pred);scaled_errors.append(scaled);warm_changes.append(warm_scaled)
    conditions.append(dict(model=model_i,warm_source=warm,anchor_query=anchor,axis=axis,level=level,delta=delta,base_delta=delta/level,native_response=native.tolist(),predicted_response=pred.tolist(),scaled_error_rms=float(np.sqrt(np.mean(scaled**2))),warm_sensitivity_rms=float(np.sqrt(np.mean(warm_scaled**2))),terminated=bool(d['terminated'][ip] or d['terminated'][im])))
   native_values=np.array(native_values);pred_values=np.array(pred_values);scaled_errors=np.array(scaled_errors);warm_changes=np.array(warm_changes)
   n=native_values*(delta/level)/std;q=pred_values*(delta/level)/std;den=np.linalg.norm(n)*np.linalg.norm(q)
   rows.append(dict(model=model_i,warm_source=warm,axis=axis,level=level,conditions=48,native_response_rms_per_output=np.sqrt(np.mean(native_values**2,0)).tolist(),predicted_response_rms_per_output=np.sqrt(np.mean(pred_values**2,0)).tolist(),response_error_rmse_per_output=np.sqrt(np.mean((pred_values-native_values)**2,0)).tolist(),scaled_error_rms=float(np.sqrt(np.mean(scaled_errors**2))),scaled_native_response_rms=float(np.sqrt(np.mean(n**2))),scaled_predicted_response_rms=float(np.sqrt(np.mean(q**2))),normalized_cosine=float(np.sum(n*q)/den) if den>1e-15 else None,warm_sensitivity_rms=float(np.sqrt(np.mean(warm_changes**2)))))
 for k,v in raw.state_dict().items():assert torch.equal(v.cpu(),c['world_model'][k])
 assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
 models.append(dict(checkpoint=str(path),sha256=sha,precision_contract=c['precision_contract']));del raw,predict,c,prediction
out=dict(requested_world_pose_semantics_verified=True,max_requested_height_roundoff=max(height_roundoff),max_requested_translation_roundoff=max(pose_write_roundoff),max_requested_rotation_roundoff=max(rotation_roundoff),query_plan_sha256=capture['query_plan_sha256'],timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=models,rows=rows,conditions=conditions,queries=plan['queries'],native_records=plan['queries']*4,transitions_sha256=audit['transitions_sha256'],validation_arrays_loaded=False,weights_unchanged=True,optimizer_updates=0,training_approved=False,promotion_approved=False,scope='All48 fixed fit anchors, declared axes/levels centered finite differences, repeat0 both warm conditions; repeat1 independently audited. Velocity outputs world-frame, normalize with unchanged parent std and base input delta. Conditional finite responses at contact, not smooth Jacobian certificate or independent heldout/policy acceptance; no sample filtering.')
a.output.write_text(json.dumps(out,indent=2))
for row in rows:print(json.dumps(row),flush=True)
