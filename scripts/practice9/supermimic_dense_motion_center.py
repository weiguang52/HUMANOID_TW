"""Dense native fit-only counterpart to the exact sparse center objective."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from supermimic_motion_action_center import MotionActionCenterLoss
class DenseMotionCenterLoss(MotionActionCenterLoss):
 def __init__(self,root,cfg,device='cuda:0'):
  root=Path(root);mapping=json.loads((root/'control_mapping.json').read_text())
  for source,target in [('state_joint_names','tw_state_joint_names'),('action_joint_names','tw_action_joint_names')]:assert mapping[source]==cfg['extra'][target]
  arrays={k:[] for k in ('state','action','context','next_state')};files=[];families=set()
  for row in json.loads((root/'results.json').read_text())['rows']:
   if row['split']!='fit':continue
   path=(root/row['trace']).resolve();assert path.is_relative_to(root.resolve());digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==row['sha256']
   with np.load(path) as z:
    assert len(z['state'])==row['steps']
    for k in arrays:arrays[k].append(z[k].copy())
   families.add(row['source_family']);files.append({'file':row['trace'],'sha256':digest})
  assert len(files)==16 and len(families)==4
  self.data={k:torch.as_tensor(np.concatenate(v),device=device,dtype=torch.float32) for k,v in arrays.items()};assert len(self.data['state'])==895
  self.generator=torch.Generator(device=device).manual_seed(61);self.last_indices=None
  self.meta={'conditions':895,'fit_files':files,'generator_seed':61,'rule':'Identical sparse center loss, normalization, batch16; uniform complete895 fit native states, no validation load.'}
 def __call__(self,model,batch=16):
  idx=torch.randint(895,(batch,),device=self.data['state'].device,generator=self.generator);self.last_indices=idx.detach().cpu().tolist();return self.evaluate(model,idx)
