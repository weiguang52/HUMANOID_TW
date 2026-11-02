"""Load only audited motion fit episodes into contiguous H8 windows."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch

def load_motion_fit_windows(collection,audit_path,mapping,device='cuda:0',window=8,*,source_plan_path=None):
 assert window in [1,8,32]
 collection=Path(collection).resolve();audit_path=Path(audit_path).resolve()
 audit=json.loads(audit_path.read_text());assert audit['passed'] and not audit['smoke']
 if source_plan_path is None:assert audit_path.parent==collection.parent
 else:assert audit_path.parent==collection
 info=json.loads((collection/'input.json').read_text());assert info['refresh_command_cache'] and not info['smoke']
 source_plan=collection.parent/'plan.json' if source_plan_path is None else Path(source_plan_path).resolve();plan=json.loads(source_plan.read_text())
 if source_plan_path is not None:
  assert plan['diagnostic_only'] and not plan['training_export_approved']
  exported=source_plan.parent/'exported_reference_audit.json';checked=json.loads(exported.read_text())
  assert checked['passed'] and checked['plan_sha256']==hashlib.sha256(source_plan.read_bytes()).hexdigest()
  assert [x['id'] for x in checked['rows']]==[x['id'] for x in plan['motions']]
  assert checked['asset_sha256']==info['asset_sha256']==plan['asset_sha256']
 assert plan['command_cache_contract']=='current_state_current_frame_v1'
 assert hashlib.sha256(source_plan.read_bytes()).hexdigest()==info['plan_sha256']
 actual=json.loads((collection/'control_mapping.json').read_text())
 for key in ['state_joint_names','action_joint_names']:assert actual[key]==mapping[key]
 for key in ['offset','scale']:np.testing.assert_allclose(actual[key],mapping[key],atol=1e-7,rtol=0)
 rows=json.loads((collection/'results.json').read_text())['rows'];assert len(rows)==32
 expected={(m['id'],int((m['frames']-1)*f),m['split'],m['source_family']) for m in plan['motions'] for f in plan['phase_fractions']}
 assert len(expected)==32 and {(r['id'],r['frame'],r['split'],r['source_family']) for r in rows}==expected
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
  for start in range(n-window+1):
   windows['state'].append(data['state'][start])
   for k in keys[1:]:windows[k].append(data[k][start:start+window])
 assert len(files)==16
 if source_plan_path is None:assert states==828 and len(windows['state'])=={1:828,8:716,32:337}[window]
 else:
  assert states==sum(r['steps'] for r in rows if r['split']=='fit')
  assert len(windows['state'])==sum(max(0,r['steps']-window+1) for r in rows if r['split']=='fit')
 meta=dict(files=files,states=states,windows=len(windows['state']),fit_families=sorted(fit_families),validation_families_metadata_only=sorted(val_families),audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),source_plan_sha256=info['plan_sha256'],results_sha256=hashlib.sha256((collection/'results.json').read_bytes()).hexdigest(),mapping_sha256=hashlib.sha256((collection/'control_mapping.json').read_bytes()).hexdigest(),validation_arrays_loaded=False)
 if window==32:meta.update(window=32,eligible_episodes=sum(x['states']>=32 for x in files),short_episodes_retained_in_h8=[x['file'] for x in files if x['states']<32])
 return {k:torch.tensor(np.stack(v),device=device) for k,v in windows.items()},meta
