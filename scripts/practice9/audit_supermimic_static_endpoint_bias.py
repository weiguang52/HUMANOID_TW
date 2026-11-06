"""Frozen native endpoint error decomposition; no optimizer or policy claim."""
import ast,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
collection=Path(sys.argv[sys.argv.index('--collection')+1]);output=Path(sys.argv[sys.argv.index('--output')+1])
proof=json.loads((collection/'integrity.json').read_text());plan=json.loads((collection/'plan.json').read_text())
assert proof['passed'] and len(proof['files'])==72 and proof['retained_original_files']==40
assert plan['contract']=='wide_static_root_velocity_branches_v1' and plan['levels']==[1.,2.,8.,16.]
def endpoint_metrics(pred,ds,base):
 errors=[p-d['next_state'].astype(float) for p,d in zip(pred,ds)]
 error=np.stack([errors[0][::2],errors[0][1::2],errors[1][::2],errors[1][1::2]],axis=1)
 assert error.shape==(6,4,73) and np.isfinite(error).all()
 norm=base.state_norm;std=(norm.var+norm.eps).sqrt();std=torch.where(norm.frozen_mask,torch.ones_like(std),std).detach().cpu().double().numpy()
 sign=np.array([-1.,1.,1.,-1.])[None,:,None];out={}
 for name,sl in [('root_linear',slice(7,10)),('root_angular',slice(10,13)),('joint_velocity',slice(43,73))]:
  e=error[:,:,sl]/std[sl];common=e.mean(axis=1,keepdims=True);odd=(e*sign).mean(axis=1,keepdims=True);residual=e-common-sign*odd
  full=float(np.mean(e**2));parts={'common_mse':float(np.mean(common**2)),'odd_mse':float(np.mean(odd**2)),'repeat_residual_mse':float(np.mean(residual**2))};defect=abs(full-sum(parts.values()));assert defect<1e-12
  out[name]={'normalized_endpoint_mse':full,**parts,'identity_residual':defect,'physical_endpoint_rmse':float(np.sqrt(np.mean(error[:,:,sl]**2))),'common_axis_mse':np.mean(common**2,axis=(1,2)).tolist(),'odd_axis_mse':np.mean(odd**2,axis=(1,2)).tolist()}
 return out
source=Path(__file__).with_name('audit_supermimic_root_state_response.py');tree=ast.parse(source.read_text());contracts=0;insertions=0
for n in ast.walk(tree):
 if isinstance(n,ast.Constant) and n.value=='root_velocity_state_branches_v1':n.value='wide_static_root_velocity_branches_v1';contracts+=1
 if isinstance(n,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='normalized_input_root_linear_response' for k in n.keys):
  n.keys.append(ast.Constant('endpoint_error_decomposition'));n.values.append(ast.parse('_endpoint_metrics(pred,ds,base)',mode='eval').body);insertions+=1
assert contracts==insertions==1
env={'__name__':'__main__','__file__':str(source),'_endpoint_metrics':endpoint_metrics}
exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),env)
d=json.loads((output/'audit.json').read_text());assert len(d['models'])==3
for m in d['models']:assert m['weights_and_norms_unchanged'] and len(m['conditions'])==32
reference=collection.parent.parent/'static_wide_response_pair_v1/static_wide_native_response/audit.json';ref=json.loads(reference.read_text())
assert [m['sha256'] for m in d['models']]==[m['sha256'] for m in ref['models']]
for new,old in zip(d['models'],ref['models']):
 stripped=[{k:v for k,v in row.items() if k!='endpoint_error_decomposition'} for row in new['conditions']];assert stripped==old['conditions']
record={'passed':True,'base_auditor_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'integrity_sha256':hashlib.sha256((collection/'integrity.json').read_bytes()).hexdigest(),'reference_audit_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),'original_response_metrics_bitexact':True,'endpoint_count_per_model':768,'optimizer_updates':0,'scope':'All four levels and partitions, all768 perturbed endpoints,6input axes; output-normalized error common/odd/repeat decomposition. Not physical causality or control acceptance.'}
(output/'integrity.json').write_text(json.dumps(record,indent=2));print('ENDPOINT_BIAS_AUDITED',record)
