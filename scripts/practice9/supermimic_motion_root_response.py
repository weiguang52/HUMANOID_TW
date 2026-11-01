"""Source-bound motion root-velocity response pools; default fit-only loading."""
import hashlib,json
from pathlib import Path
import numpy as np,torch

def load_motion_root_response(root,cfg,split='fit',device='cuda:0'):
 root=Path(root).resolve();assert split in ['fit','validation']
 audit_path=root.parent/'serial_integrity.json';audit=json.loads(audit_path.read_text());assert audit['passed'] and not audit['smoke'] and audit['endpoints']==5760
 plan=json.loads((root/'plan.json').read_text());assert plan['serial'] and plan['num_envs']==1 and plan['contract']=='motion_root_velocity_branches_v1'
 assert plan['asset_sha256']==cfg['extra']['tw_mjcf_sha256'] and plan['physics']==cfg['extra']['tw_physics_contract']
 mapping=json.loads((root/'control_mapping.json').read_text())
 for a,b in [('state_joint_names','tw_state_joint_names'),('action_joint_names','tw_action_joint_names')]:assert mapping[a]==cfg['extra'][b]
 hashes={x['file']:x['sha256'] for x in audit['files']};loaded={};files=[]
 for row in json.loads((root/'results.json').read_text())['rows']:
  if row['split']!=split or row['level']==0:continue
  path=(root/row['file']).resolve();assert path.is_relative_to(root)
  sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==hashes[row['file']]==row['sha256']
  with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in ['state','action','context','next_state']}
  loaded[(row['source_trace'],row['index'],row['level'],row['assignment'])]=d;files.append(dict(file=row['file'],sha256=sha))
 assert len(files)==192
 arrays={k:[] for k in ['state','action','context','next_state']};den=[];axes=[]
 anchors=sorted({(key[0],key[1]) for key in loaded});assert len(anchors)==48
 for trace,index in anchors:
  for level in plan['levels']:
   a,b=[loaded[(trace,index,level,j)] for j in [0,1]]
   for k in arrays:arrays[k].append(np.stack([a[k][::2],a[k][1::2],b[k][::2],b[k][1::2]],1))
   den.extend(4*level*np.array(plan['base_delta']));axes.extend(range(6))
 data={k:torch.as_tensor(np.concatenate(v),device=device,dtype=torch.float32) for k,v in arrays.items()}
 assert len(data['state'])==576
 meta=dict(split=split,files=files,conditions=576,anchors=48,audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),plan_sha256=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(),validation_arrays_loaded=split=='validation')
 return data,torch.tensor(den,device=device,dtype=torch.float32)[:,None],torch.tensor(axes,device=device),meta

def derivative(values,den):return (values[:,1,7:13]-values[:,0,7:13]+values[:,2,7:13]-values[:,3,7:13])/den
