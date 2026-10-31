"""Load only audited motion fit episodes into contiguous H8 windows."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch

def load_motion_fit_windows(collection,audit_path,mapping,device='cuda:0'):
 collection=Path(collection).resolve();audit_path=Path(audit_path).resolve()
 audit=json.loads(audit_path.read_text());assert audit['passed'] and not audit['smoke']
 assert audit_path.parent==collection.parent
 info=json.loads((collection/'input.json').read_text());assert info['refresh_command_cache'] and not info['smoke']
 source_plan=collection.parent/'plan.json';plan=json.loads(source_plan.read_text())
 assert plan['command_cache_contract']=='current_state_current_frame_v1'
 assert hashlib.sha256(source_plan.read_bytes()).hexdigest()==info['plan_sha256']
 actual=json.loads((collection/'control_mapping.json').read_text())
 for key in ['state_joint_names','action_joint_names']:assert actual[key]==mapping[key]
 for key in ['offset','scale']:np.testing.assert_allclose(actual[key],mapping[key],atol=1e-7,rtol=0)
 rows=json.loads((collection/'results.json').read_text())['rows'];assert len(rows)==32
 fit_families={x['source_family'] for x in rows if x['split']=='fit'}
 val_families={x['source_family'] for x in rows if x['split']=='validation'}
 assert len(fit_families)==len(val_families)==4 and not fit_families&val_families
 audited={x['trace']:x for x in audit['rows']};assert len(audited)==32
 keys=['state','action','context','next_state'];windows={k:[] for k in keys};files=[];states=0
 for row in rows:
  if row['split']!='fit':continue
  path=(collection/row['trace']).resolve();assert path.is_relative_to(collection)
  sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==row['sha256']
  assert audited[row['trace']]['command_cache_z_checked']
  with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in keys};done=(z['terminated']|z['truncated']).reshape(-1)
  n=len(data['state']);assert n==row['steps']==audited[row['trace']]['steps'] and not done[:-1].any()
  for k,width in [('state',73),('action',30),('context',34),('next_state',73)]:assert data[k].shape==(n,width) and np.isfinite(data[k]).all()
  assert np.array_equal(data['next_state'][:-1],data['state'][1:])
  files.append(dict(file=row['trace'],sha256=sha,states=n,source_family=row['source_family']));states+=n
  for start in range(n-7):
   windows['state'].append(data['state'][start])
   for k in keys[1:]:windows[k].append(data[k][start:start+8])
 assert len(files)==16 and states==828 and len(windows['state'])==716
 meta=dict(files=files,states=states,windows=716,fit_families=sorted(fit_families),validation_families_metadata_only=sorted(val_families),audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),source_plan_sha256=info['plan_sha256'],results_sha256=hashlib.sha256((collection/'results.json').read_bytes()).hexdigest(),mapping_sha256=hashlib.sha256((collection/'control_mapping.json').read_bytes()).hexdigest(),validation_arrays_loaded=False)
 return {k:torch.tensor(np.stack(v),device=device) for k,v in windows.items()},meta
