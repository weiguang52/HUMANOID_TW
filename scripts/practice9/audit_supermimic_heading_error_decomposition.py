"""Frozen H1 oracle decomposition; not a deployable predictor or acceptance."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
from supermimic_geometry_model import load_geometry_artifact
from supermimic_yaw_canonical import canonicalize_state,rotate_vector
p=argparse.ArgumentParser(description=__doc__)
for k in ['queries','collection','checkpoint','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();plan=json.loads((a.queries/'query_plan.json').read_text());audit=json.loads((a.collection/'independent_audit.json').read_text());assert audit['passed']
f=a.collection/'transitions.npz';assert hashlib.sha256(f.read_bytes()).hexdigest()==audit['transitions_sha256']
with np.load(f) as z:d={k:z[k].copy() for k in z.files}
ids=np.flatnonzero(d['repeat']==0);lookup={(int(q),bool(w)):i for i,(q,w) in enumerate(zip(d['query'][ids],d['warm_source'][ids]))};zero=[]
for i in ids:
 row=plan['rows'][int(d['query'][i])];base=next(x['query'] for x in plan['rows'] if x['anchor_query']==row['anchor_query'] and x['yaw_degrees']==0);zero.append(lookup[base,bool(d['warm_source'][i])])
sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest();c=torch.load(a.checkpoint,map_location='cpu',weights_only=False);raw,predict=load_geometry_artifact(c);predict.eval()
with torch.no_grad():
 inputs={k:torch.tensor(d[k][ids],device='cuda:0') for k in ['state','action','context']};pred=torch.cat([predict(*[inputs[k][i:i+64] for k in ['state','action','context']]) for i in range(0,len(ids),64)])
 _,frame=canonicalize_state(inputs['state'].double())
 def local_velocity(s):return torch.cat([rotate_vector(s[:,7:10].double(),frame['c'],-frame['s']),rotate_vector(s[:,10:13].double(),frame['c'],-frame['s']),s[:,43:73].double()],dim=-1)
 target=local_velocity(torch.tensor(d['next_state'][ids],device='cuda:0'));base=local_velocity(pred);z=torch.tensor(zero,device='cuda:0');response_error=(base-base[z])-(target-target[z]);oracle=base-response_error
 cols=list(range(7,13))+list(range(43,73));std=(raw.state_norm.var[cols]+raw.state_norm.eps).sqrt().double();std=torch.where(raw.state_norm.frozen_mask[cols],torch.ones_like(std),std)
 def score(error):return dict(scaled_velocity_mse=float((error/std).square().mean()),root_linear_rmse=float(error[:,:3].square().mean().sqrt()),root_angular_rmse=float(error[:,3:6].square().mean().sqrt()),joint_velocity_rmse=float(error[:,6:].square().mean().sqrt()))
 rows=[]
 labels=[str(plan['rows'][int(d['query'][i])]['source_family']) for i in ids];angles=[plan['rows'][int(d['query'][i])]['yaw_degrees'] for i in ids]
 for family in ['ALL']+sorted(set(labels)):
  keep=torch.tensor([i for i,label in enumerate(labels) if family=='ALL' or label==family],device='cuda:0')
  rows.append(dict(source_family=family,records=len(keep),absolute=score((base-target)[keep]),response=score(response_error[keep]),oracle_remove_response=score((oracle-target)[keep])))
 for angle in [0,30,45,90,180]:
  keep=torch.tensor([i for i,x in enumerate(angles) if x==angle],device='cuda:0');rows.append(dict(angle=angle,records=len(keep),absolute=score((base-target)[keep]),response=score(response_error[keep]),oracle_remove_response=score((oracle-target)[keep])))
assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==sha
for k,v in raw.state_dict().items():assert torch.equal(v.cpu(),c['world_model'][k])
out=dict(checkpoint_sha256=sha,transitions_sha256=audit['transitions_sha256'],rows=rows,optimizer_updates=0,policy_acceptance=False,scope='Uses native zero-angle truth to remove only paired heading response error; oracle is not deployable. H1 variance comparison does not bound H8/H32 causal impact or establish independent generalization. All repeat0 source/zero conditions, no filtering.')
a.output.write_text(json.dumps(out,indent=2));print(json.dumps(rows[0]))
