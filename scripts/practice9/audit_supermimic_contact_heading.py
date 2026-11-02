"""Fit-only heading identifiability and proposed feature integrity audit."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from supermimic_yaw_canonical import canonicalize_state,rotate_vector
from supermimic_contact_heading import contact_heading_features,CONTRACT
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queries',type=Path,required=True);p.add_argument('--collection',type=Path,required=True);a=p.parse_args()
plan=json.loads((a.queries/'query_plan.json').read_text());audit=json.loads((a.collection/'independent_audit.json').read_text());assert audit['passed'] and audit.get('diagnostic_cone','pyramidal')=='pyramidal'
assert hashlib.sha256((a.collection/'transitions.npz').read_bytes()).hexdigest()==audit['transitions_sha256']
with np.load(a.collection/'transitions.npz') as z:d={k:z[k].copy() for k in z.files}
s=torch.tensor(d['state'],dtype=torch.float64);n=torch.tensor(d['next_state'],dtype=torch.float64)
local,frames=canonicalize_state(s);features=contact_heading_features(s)
lookup={(int(q),bool(w),int(rep)):i for i,(q,w,rep) in enumerate(zip(d['query'],d['warm_source'],d['repeat']))}
records=[]
for row in plan['rows']:
 zero=next(x['query'] for x in plan['rows'] if x['anchor_query']==row['anchor_query'] and x['yaw_degrees']==0)
 for warm in [True,False]:
  i,j=lookup[row['query'],warm,0],lookup[zero,warm,0]
  assert np.array_equal(d['action'][i],d['action'][j]) and np.array_equal(d['context'][i],d['context'][j])
  # Compare actual root/joint velocity targets in each INPUT's canonical frame.
  targets=[]
  for k in [i,j]:
   c,ss=frames['c'][k:k+1],frames['s'][k:k+1]
   targets.append(torch.cat([rotate_vector(n[k:k+1,7:10],c,-ss),rotate_vector(n[k:k+1,10:13],c,-ss),n[k:k+1,43:73]],dim=-1))
  gap=targets[0]-targets[1]
  records.append(dict(angle=row['yaw_degrees'],query=row['query'],warm_source=warm,canonical_input_max_difference=float((local[i]-local[j]).abs().max()),feature_distance=float((features[i]-features[j]).norm()),velocity_target_gap_l2=float(gap.norm()),shared_prediction_pair_rmse_lower_bound=float(gap.square().mean().sqrt()/2)))
probe=s[:24].clone().requires_grad_(True);g=torch.autograd.grad(contact_heading_features(probe).square().sum()+contact_heading_features(probe)[:,0].sum(),probe)[0];assert torch.isfinite(g).all()
summaries=[]
for angle in plan['angles_degrees']:
 rs=[r for r in records if r['angle']==angle]
 summaries.append(dict(angle=angle,n=len(rs),canonical_input_difference_max=max(r['canonical_input_max_difference'] for r in rs),feature_distance_min=min(r['feature_distance'] for r in rs),feature_distance_max=max(r['feature_distance'] for r in rs),velocity_target_gap_l2_max=max(r['velocity_target_gap_l2'] for r in rs),pair_rmse_lower_bound_max=max(r['shared_prediction_pair_rmse_lower_bound'] for r in rs)))
for r in records:
 if r['angle'] in [0,90,180]:assert r['feature_distance']<2e-6
 else:assert r['feature_distance']>1
out=dict(contract=CONTRACT,summary=summaries,records=records,finite_gradient=True,validation_arrays_loaded=False,optimizer_updates=0,training_approved=False,promotion_approved=False,scope='Near-identical canonical inputs have different native labels. Pair lower bound assumes identical prediction, not a bound for a model exploiting floating-point input noise. Features alone do not prove learning or control.')
(a.collection/'contact_heading_identifiability.json').write_text(json.dumps(out,indent=2));print(json.dumps(summaries))
