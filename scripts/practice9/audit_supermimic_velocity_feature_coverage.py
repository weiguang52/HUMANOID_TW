"""Read-only fit coverage in fixed physical height/velocity feature coordinates."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from supermimic_sole_corner_velocity import SoleCornerVelocity
p=argparse.ArgumentParser(description=__doc__)
for k in ['root','asset','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();r=a.root
names=json.loads((r/'root_pose_branches_v1/control_mapping.json').read_text())['state_joint_names']
f=SoleCornerVelocity(a.asset,names).cuda();groups={};queries={};sources=[]
def load_feature(root,row,filekey,group):
 path=(root/row[filekey]).resolve();assert path.is_relative_to(root.resolve())
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 if 'sha256' in row:assert digest==row['sha256']
 with np.load(path,allow_pickle=False) as z:state=torch.tensor(z['state'],device='cuda:0')
 assert state.ndim==2 and state.shape[1]==73 and torch.isfinite(state).all()
 with torch.no_grad():h,v=f.height_and_velocity(state)
 h=h[...,[0,2,4,6]].reshape(-1,8);v=v[...,[0,2,4,6]].reshape(-1,8)
 sources.append(dict(group=group,file=str(path),sha256=digest,states=len(state)))
 return torch.cat([h/.001,v/.05],-1),v
fit=json.loads((r/'sole_geometry_fit_v1/control/fit/fit_input.json').read_text())
for label,dirname,key in [('trajectory_fit','static_perturb_v1','trajectory_files'),('action_fit','orthogonal_branches_v1','one_step_files')]:
 values=[load_feature(r/dirname,row,'file',label) for row in fit[key]];groups[label]=(torch.cat([x[0] for x in values]),torch.cat([x[1] for x in values]))
for name,dirname in [('pose','root_pose_branches_v1'),('velocity','root_velocity_branches_v1')]:
 root=r/dirname;integ=json.loads((root/'integrity.json').read_text());assert integ['passed']
 hashes={x['file']:x['sha256'] for x in integ['files']}
 for split in ['fit','validation']:
  label=name+'_'+split
  values=[load_feature(root,dict(row,sha256=hashes[row['file']]),'file',label) for row in json.loads((root/'results.json').read_text())['rows'] if row['split']==split]
  (groups if split=='fit' else queries)[label]=(torch.cat([x[0] for x in values]),torch.cat([x[1] for x in values]))
root=r/'static_perturb_v1';values=[load_feature(root,row,'trace','trajectory_validation') for row in json.loads((root/'results.json').read_text())['rows'] if row['split']=='validation']
queries['trajectory_validation']=(torch.cat([x[0] for x in values]),torch.cat([x[1] for x in values]))
summary={}
for label,(feature,velocity) in groups.items():
 row=dict(fit_states=len(feature),corner_velocity_quantiles_m_s=np.quantile(velocity.cpu().numpy(),[0,.05,.5,.95,1]).tolist(),queries={})
 for query_label,(q,_) in queries.items():
  distances=[]
  for start in range(0,len(q),64):distances.append((q[start:start+64,None]-feature[None]).square().mean(-1).min(-1).values.sqrt())
  row['queries'][query_label]=dict(states=len(q),nearest_joint_feature_rms_quantiles=np.quantile(torch.cat(distances).cpu().numpy(),[0,.5,.95,1]).tolist())
 summary[label]=row
out=dict(sources=sources,groups=summary,height_scale_m=.001,velocity_scale_m_s=.05,
 scope='Descriptive fixed-coordinate nearest distances only. Validation is queried, never fitted or normalized. Does not establish complete state/action/contact-force coverage or control success; no acceptance threshold introduced.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary))
