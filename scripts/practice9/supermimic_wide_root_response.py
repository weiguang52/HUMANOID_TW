"""Load only fully verified wide native root-velocity branches; fit is default."""
import hashlib,json
from pathlib import Path
import numpy as np,torch
LEVELS=[1.,2.,8.,32.,64.]
def load_wide_root_response(root,cfg,split='fit',device='cuda:0'):
 root=Path(root).resolve();assert split in ['fit','validation'];proof_path=root.parent/'full_integrity.json';proof=json.loads(proof_path.read_text());plan=json.loads((root/'plan.json').read_text())
 assert proof['passed'] and not proof['smoke'] and proof['batches']==1056 and proof['endpoints']==12672 and not any(proof['counts'].values())
 assert len(proof['original_levels_regression'])==480 and proof['wide_levels']==LEVELS
 assert plan['contract']=='wide_motion_root_velocity_branches_v1' and plan['levels']==LEVELS and plan['serial'] and plan['num_envs']==1 and plan['restore_native_state']
 assert plan['asset_sha256']==cfg['extra']['tw_mjcf_sha256'] and plan['physics']==cfg['extra']['tw_physics_contract']
 variant=json.loads((root/'wide_velocity_diagnostic.json').read_text());assert variant['runtime_dense_verified'] and not variant['training_approved'] and variant['levels']==LEVELS
 mapping=json.loads((root/'control_mapping.json').read_text())
 for k,v in [('state_joint_names','tw_state_joint_names'),('action_joint_names','tw_action_joint_names')]:assert mapping[k]==cfg['extra'][v]
 hashes={x['file']:x['sha256'] for x in proof['files']};loaded={};files=[]
 for row in json.loads((root/'results.json').read_text())['rows']:
  if row['split']!=split or row['level']==0:continue
  path=(root/row['file']).resolve();assert path.is_relative_to(root);sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==hashes[row['file']]==row['sha256']
  with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in ['state','action','context','next_state','terminated','truncated']}
  loaded[(row['source_trace'],row['index'],row['level'],row['assignment'])]=d;files.append({'file':row['file'],'sha256':sha})
 assert len(files)==480
 anchors=sorted({(k[0],k[1]) for k in loaded});assert len(anchors)==48
 arrays={k:[] for k in ['state','action','context','next_state','terminated','truncated']};den=[];axes=[];levels=[]
 for trace,index in anchors:
  for level in LEVELS:
   a,b=[loaded[(trace,index,level,i)] for i in [0,1]]
   for k in arrays:arrays[k].append(np.stack([a[k][::2],a[k][1::2],b[k][::2],b[k][1::2]],1))
   den.extend(4*level*np.array(plan['base_delta']));axes.extend(range(6));levels.extend([level]*6)
 data={k:torch.as_tensor(np.concatenate(v),device=device) for k,v in arrays.items()};assert len(data['state'])==1440
 meta={'split':split,'files':files,'anchors':48,'conditions':1440,'levels':LEVELS,'proof_sha256':hashlib.sha256(proof_path.read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(),'validation_arrays_loaded':split=='validation','scope':'Verified diagnostic physics only; loading is not approval for policy training'}
 return data,torch.tensor(den,device=device,dtype=torch.float32)[:,None],torch.tensor(axes,device=device),torch.tensor(levels,device=device),meta

from supermimic_motion_root_response import MotionRootResponseLoss,derivative
class WideMotionRootResponseLoss(MotionRootResponseLoss):
 def __init__(self,original_root,wide_root,cfg,enabled,device='cuda:0'):
  super().__init__(original_root,cfg,device);assert enabled in [0,1];self.enabled=bool(enabled)
  wide,den,axes,levels,meta=load_wide_root_response(wide_root,cfg,'fit',device)
  idx=torch.arange(576,device=device);overlap=(idx//12)*30+((idx%12)//6)*6+idx%6
  for k in ['state','action','context','next_state']:assert torch.equal(self.data[k],wide[k][overlap]),('original overlap',k)
  assert torch.equal(self.den,den[overlap]) and torch.equal(self.axes,axes[overlap])
  self.wide_meta=meta;self.wide_generator=torch.Generator(device=device).manual_seed(62);self.history=[]
  if enabled:self.data=wide;self.den=den;self.axes=axes;self.target=derivative(self.data['next_state'],self.den).detach()
 def sample(self):
  old=torch.randint(576,(8,),device=self.den.device,generator=self.generator);level=torch.randint(5,(8,),device=self.den.device,generator=self.wide_generator)
  new=(old//12)*30+level*6+old%6;used=new if self.enabled else old
  self.history.append({'update':len(self.history)+1,'original_indices':old.cpu().tolist(),'wide_level_indices':level.cpu().tolist(),'used_indices':used.cpu().tolist(),'anchor_indices':(old//12).cpu().tolist(),'axis_indices':(old%6).cpu().tolist()})
  return used
