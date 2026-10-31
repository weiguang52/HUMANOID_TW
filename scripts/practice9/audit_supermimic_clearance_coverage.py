"""Read-only geometric coverage in sole corner heights, not full-state identifiability."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from supermimic_sole_clearance import SoleClearance
p=argparse.ArgumentParser(description=__doc__)
for k in ['root','asset','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();r=a.root
mapping=json.loads((r/'root_pose_branches_v1/control_mapping.json').read_text())
f=SoleClearance(a.asset,mapping['state_joint_names'],0.).cuda()
fit=json.loads((r/'root_pose_response_fit_v1/control/fit/fit_input.json').read_text())
groups={};sources=[]
def add_files(label,root,files):
 values=[]
 for row in files:
  path=(root/row['file']).resolve();assert path.is_relative_to(root.resolve())
  assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
  with np.load(path,allow_pickle=False) as z:state=torch.tensor(z['state'],device='cuda:0')
  with torch.no_grad():height=f(state).reshape(len(state),16)
  values.append(height);sources.append(dict(group=label,file=str(path),sha256=row['sha256'],states=len(state)))
 groups[label]=torch.cat(values)
add_files('trajectory_fit',r/'static_perturb_v1',fit['trajectory_files'])
add_files('action_branch_fit',r/'orthogonal_branches_v1',fit['one_step_files'])
for label,dirname in [('velocity_response_fit','root_velocity_branches_v1'),('pose_response_fit','root_pose_branches_v1')]:
 root=r/dirname;integrity=json.loads((root/'integrity.json').read_text());assert integrity['passed']
 hashes={x['file']:x['sha256'] for x in integrity['files']}
 rows=json.loads((root/'results.json').read_text())['rows']
 add_files(label,root,[dict(file=x['file'],sha256=hashes[x['file']]) for x in rows if x['split']=='fit'])
geometry=json.loads((r/'root_pose_response_fit_v1/contact_geometry.json').read_text())
queries=[];labels=[]
for row in geometry['rows']:
 if row['split']!='validation':continue
 h=torch.tensor(row['corner_height_m'],device='cuda:0',dtype=torch.float32).reshape(-1,16)
 queries.append(h);labels.extend([dict(file=row['file'],state_index=i) for i in range(len(h))])
q=torch.cat(queries);summaries={};distances={}
for label,values in groups.items():
 dist=[]
 for start in range(0,len(q),64):
  e=q[start:start+64,None]-values[None]
  dist.append(e.square().mean(-1).min(-1).values.sqrt())
 dist=torch.cat(dist).cpu().numpy();distances[label]=dist
 footmin=values.reshape(-1,2,8).min(-1).values.cpu().numpy()*1000
 summaries[label]=dict(states=len(values),minimum_corner_height_quantiles_mm=np.quantile(footmin,[0,.05,.5,.95,1],axis=0).tolist(),nearest_validation_rms_height_quantiles_mm=np.quantile(dist*1000,[0,.5,.95,1]).tolist())
out=dict(sources=sources,groups=summaries,validation_states=len(q),queries=[dict(**v,nearest_rms_height_mm={k:float(x[i]*1000) for k,x in distances.items()}) for i,v in enumerate(labels)],
 scope='Original fit sources only; validation used only as queries. Corner-height distances do not establish full state, velocity, action, contact-force or independent-motion coverage. No acceptance threshold or parameter fitting.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(groups=summaries,validation_states=len(q))))
