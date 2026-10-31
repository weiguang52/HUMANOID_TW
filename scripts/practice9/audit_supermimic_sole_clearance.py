"""Verify differentiable sole heights against archived native geometry."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from supermimic_sole_clearance import SoleClearance
p=argparse.ArgumentParser(description=__doc__)
for k in ['geometry','collection','asset','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();d=json.loads(a.geometry.read_text())
assert d['asset_sha256']==hashlib.sha256(a.asset.read_bytes()).hexdigest()
mapping=json.loads((a.collection/'control_mapping.json').read_text())
f=SoleClearance(a.asset,mapping['state_joint_names'],d['explicit_environment_ground_height_m']).cuda()
rows=[];maximum=0.;count=0
for row in d['rows']:
 path=(a.collection/row['file']).resolve();assert path.is_relative_to(a.collection.resolve())
 assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
 with np.load(path,allow_pickle=False) as z:s=torch.tensor(z['state'],device='cuda:0')
 with torch.no_grad():
  height=f(s);native=torch.tensor(row['corner_height_m'],device='cuda:0')
  error=float((height-native).abs().max());maximum=max(maximum,error)
  shifted=s.clone();shifted[:,:2]+=torch.tensor([.02,-.03],device='cuda:0')
  xy_error=float((f(shifted)-height).abs().max());assert xy_error<=1e-6
  flipped=s.clone();flipped[:,3:7]*=-1
  sign_error=float((f(flipped)-height).abs().max());assert sign_error<=1e-6
 s=s.detach().requires_grad_();value=f(s).sum();value.backward()
 assert torch.isfinite(s.grad).all()
 z_error=float((s.grad[:,2]-16).abs().max());assert z_error<=1e-5
 rows.append(dict(file=row['file'],frames=len(s),max_height_error_m=error,xy_invariance_error_m=xy_error,quaternion_sign_error_m=sign_error,z_sum_gradient_error=z_error))
 count+=len(s)
result=dict(passed=maximum<=1e-6,frames=count,max_native_height_error_m=maximum,atol_m=1e-6,
 geometry_sha256=hashlib.sha256(a.geometry.read_bytes()).hexdigest(),rows=rows,
 scope='All archived input poses; verifies geometry and finite gradients, not loaded contacts or model/control accuracy')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}));assert result['passed']
