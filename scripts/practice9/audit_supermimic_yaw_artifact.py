"""Audit the actual serialized yaw prediction function without fitting or promotion."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from supermimic_geometry_model import load_geometry_artifact
from supermimic_yaw_canonical import KIND
from supermimic_yaw_transform import rotate_world_state,state_metrics
p=argparse.ArgumentParser(description=__doc__)
for k in ['checkpoint','queries','output']:p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--internal-float64-diagnostic',action='store_true')
a=p.parse_args();assert not a.output.exists();c=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert c['kind']==KIND
m,predict=load_geometry_artifact(c)
if a.internal_float64_diagnostic:
 from supermimic_precision_diagnostic import Float64PredictionDiagnostic
 predict=Float64PredictionDiagnostic(predict)
with np.load(a.queries,allow_pickle=False) as z:data={k:torch.tensor(z[k],device='cuda:0') for k in ['state','action','context']}
assert len(data['state'])==144;rows=[]
with torch.no_grad():
 original=predict(data['state'],data['action'],data['context']);assert torch.isfinite(original).all()
 for turns in [0,1,2]:
  value=predict(rotate_world_state(data['state'],turns),data['action'],data['context']);assert torch.isfinite(value).all()
  metrics=state_metrics(value,rotate_world_state(original,turns));rows.append(dict(check='quarter_turn_'+str(turns),metrics=metrics,passed=metrics['max_state_component_error']<=1e-5))
 flipped=data['state'].clone();flipped[:,3:7]*=-1;value=predict(flipped,data['action'],data['context']);assert torch.isfinite(value).all()
 metrics=state_metrics(value,original);rows.append(dict(check='quaternion_antipode',metrics=metrics,passed=metrics['max_state_component_error']<=1e-5))
 for k,v in m.state_dict().items():assert torch.equal(v.cpu(),c['world_model'][k]),k
out=dict(checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),query_sha256=hashlib.sha256(a.queries.read_bytes()).hexdigest(),coordinate_contract=c['coordinate_contract'],states=144,rows=rows,all_within_structure_atol=all(x['passed'] for x in rows),weights_unchanged=True,scope='Actual loader function at fixed144 float32 queries only; disabled control need not be equivariant. Failures retained at1e-5, no threshold change or physical/policy acceptance.')
out['arithmetic_diagnostic']='float64_internal_float32_api' if a.internal_float64_diagnostic else 'original_artifact_contract'
if c.get('precision_contract') is not None:out['precision_contract']=c['precision_contract']
a.output.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)
