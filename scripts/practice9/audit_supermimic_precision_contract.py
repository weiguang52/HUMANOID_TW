"""Audit actual declared arithmetic, default regression, rejection and gradients."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from supermimic_geometry_model import load_geometry_artifact
from supermimic_pose_integration_diagnostic import PoseVelocityModel
from supermimic_yaw_canonical import YawCanonicalModel,apply_coordinate_contract
from supermimic_precision_contract import validate,state_digest,DeclaredFloat64Predictor
from supermimic_precision_diagnostic import Float64PredictionDiagnostic
from supermimic_yaw_transform import rotate_world_state,state_metrics
from supermimic.algo.networks import WorldModel
p=argparse.ArgumentParser(description=__doc__)
for key in ['source','artifact','queries','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();source_sha=hashlib.sha256(a.source.read_bytes()).hexdigest();artifact_sha=hashlib.sha256(a.artifact.read_bytes()).hexdigest()
s=torch.load(a.source,map_location='cpu',weights_only=False);c=torch.load(a.artifact,map_location='cpu',weights_only=False);assert validate(c) and not validate(s)
assert c['precision_contract']['source_checkpoint_sha256']==source_sha and state_digest(s['world_model'])==state_digest(c['world_model'])
with np.load(a.queries,allow_pickle=False) as z:d={k:torch.tensor(z[k],device='cuda:0') for k in ['state','action','context']}
assert len(d['state'])==144
old,old_predict=load_geometry_artifact(s);raw,predict=load_geometry_artifact(c);assert isinstance(predict,DeclaredFloat64Predictor)
legacy=PoseVelocityModel(YawCanonicalModel(old,True),s['prediction_contract']);diagnostic=Float64PredictionDiagnostic(old_predict)
runtime=[m for m in predict.modules() if isinstance(m,WorldModel)];assert len(runtime)==1
assert all(v.dtype==torch.float64 for v in runtime[0].parameters())
with torch.no_grad():
 for _ in range(3):old_predict(*d.values());legacy(*d.values());predict(*d.values());diagnostic(*d.values())
 original=old_predict(*d.values());expected=legacy(*d.values());declared=predict(*d.values());diagnosed=diagnostic(*d.values())
 assert torch.equal(original,expected)
 diagnostic_error=float((declared-diagnosed).abs().max());assert diagnostic_error<=1e-7
 assert declared.dtype==d['state'].dtype==torch.float32 and torch.isfinite(declared).all()
 extra=[]
 value=predict(rotate_world_state(d['state'],3),d['action'],d['context']);metric=state_metrics(value,rotate_world_state(declared,3));extra.append(dict(check='quarter_turn_3',metrics=metric,passed=metric['max_state_component_error']<=1e-5))
 shifted=d['state'].clone();shifted[:,:2]+=shifted.new_tensor([1.,-1.5]);value=predict(shifted,d['action'],d['context']);target=declared.clone();target[:,:2]+=target.new_tensor([1.,-1.5]);metric=state_metrics(value,target);extra.append(dict(check='xy_translation',metrics=metric,passed=metric['max_state_component_error']<=1e-5))
assert all(x['passed'] for x in extra)
inputs=[v[:8].detach().clone().requires_grad_(True) for v in d.values()];value=predict(*inputs);grad=torch.autograd.grad(value.square().mean(),inputs);assert all(torch.isfinite(v).all() for v in grad)
rejected=[]
try:apply_coordinate_contract(raw,c)
except ValueError:rejected.append('raw_float32_coordinate_bypass')
else:raise AssertionError('Raw float32 helper ignored declared precision')
def must_reject(name,bad,loader=False):
 try:
  if loader:load_geometry_artifact(bad)
  else:validate(bad)
 except (AssertionError,ValueError,KeyError,TypeError):rejected.append(name)
 else:raise AssertionError('Invalid precision declaration accepted: '+name)
bad=dict(c);bad.pop('precision_contract');must_reject('float64_coordinate_without_precision',bad,True)
bad=dict(c,kind='policy_checkpoint');must_reject('policy_kind',bad)
bad=dict(c,precision_contract=dict(c['precision_contract'],contract='unknown'));must_reject('unknown_contract',bad)
bad=dict(c,resumable_training=True);must_reject('resumable_training',bad)
bad=dict(c,optimizer={});must_reject('optimizer_present',bad)
bad=dict(c,precision_contract=dict(c['precision_contract'],api_dtype='float16'));must_reject('unsupported_api_dtype',bad)
bad=dict(c,world_model=dict(c['world_model']));key=next(k for k in bad['world_model'] if k.endswith('.weight'));bad['world_model'][key]=bad['world_model'][key].clone();bad['world_model'][key].view(-1)[0]+=1.;must_reject('stored_weight_digest_mismatch',bad)
timing={}
with torch.no_grad():
 for name,fn in [('original',old_predict),('declared_float64',predict)]:
  for _ in range(3):fn(*d.values())
  torch.cuda.synchronize();start=time.perf_counter()
  for _ in range(10):fn(*d.values())
  torch.cuda.synchronize();timing[name+'_seconds_per144']=((time.perf_counter()-start)/10)
for model,state in [(old,s['world_model']),(raw,c['world_model'])]:
 for k,v in model.state_dict().items():assert torch.equal(v.cpu(),state[k]),k
for k,v in runtime[0].state_dict().items():assert torch.equal(v.cpu().to(c['world_model'][k].dtype),c['world_model'][k]),k
assert hashlib.sha256(a.source.read_bytes()).hexdigest()==source_sha and hashlib.sha256(a.artifact.read_bytes()).hexdigest()==artifact_sha
out=dict(passed=True,source_sha256=source_sha,artifact_sha256=artifact_sha,default_source_bitexact_legacy=True,declared_vs_diagnostic_max_error=diagnostic_error,stored_weights_unchanged=True,runtime_parameters_float64=True,api_dtype='float32',extra_coordinate_checks=extra,finite_state_action_context_gradients=True,rejected_cases=rejected,timing=timing,optimizer_updates=0,promotion_approved=False,scope='Declared actual loader, fixed144 coordinate inputs +8 gradient probes; coordinate1e-5 unchanged. Timing local10 repeats, no physical-control acceptance. New artifact is distinct from failed original source.')
a.output.write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
