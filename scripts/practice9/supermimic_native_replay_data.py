"""Audited real native transitions for offline dynamics only, never reference export."""
import hashlib,json
from pathlib import Path
import numpy as np,torch

def load_native_replay(root,cfg,device='cuda:0'):
 root=Path(root).resolve();collection=root/'native_full';planfile=root/'collection_plan.json';plan=json.loads(planfile.read_text());assert plan['contract']=='expanded_native_contact_sources_v1' and not plan['training_export_approved']
 auditfile=root/'full_independent_audit.json';audit=json.loads(auditfile.read_text());assert audit['passed'] and not audit['smoke'] and len(audit['rows'])==96
 info=json.loads((collection/'input.json').read_text());assert info['plan_sha256']==hashlib.sha256(planfile.read_bytes()).hexdigest() and info['native_solver_state'] and info['refresh_command_cache']
 assert info['asset_sha256']==cfg['extra']['tw_mjcf_sha256'] and info['physics']==cfg['extra']['tw_physics_contract']
 yawroot=root.parent/'expanded_contact_yaw_v1';yaw=json.loads((yawroot/'full/independent_audit.json').read_text());assert yaw['passed'] and not yaw['smoke'] and yaw['max_archived_anchor_error']<=1e-5
 assert Path(json.loads((yawroot/'query_plan.json').read_text())['manifest']).resolve()==planfile
 mapping=json.loads((collection/'control_mapping.json').read_text());assert mapping['state_joint_names']==cfg['extra']['tw_state_joint_names'] and mapping['action_joint_names']==cfg['extra']['tw_action_joint_names']
 blocked=set()
 for key in ['historical_validation','development8','final7']:
  spec=plan['inputs'][key];p=Path(spec['file']);assert hashlib.sha256(p.read_bytes()).hexdigest()==spec['sha256'];blocked.update(x['source_family'] for x in json.loads(p.read_text())['motions'])
 rows=json.loads((collection/'results.json').read_text())['rows'];assert len(rows)==96 and len({r['source_family'] for r in rows})==24 and all(r['split']=='fit' and r['source_family'] not in blocked for r in rows)
 expected={(m['id'],int((m['frames']-1)*f)) for m in plan['motions'] for f in plan['phase_fractions']};assert {(r['id'],r['frame']) for r in rows}==expected
 arrays={k:[] for k in ['state','action','context','next_state']};files=[]
 for row in rows:
  p=(collection/row['trace']).resolve();assert p.is_relative_to(collection) and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
  with np.load(p,allow_pickle=False) as z:
   n=len(z['state']);assert n==row['steps'] and np.array_equal(z['next_state'][:-1],z['state'][1:])
   for k,width in [('state',73),('action',30),('context',34),('next_state',73)]:assert z[k].shape==(n,width) and np.isfinite(z[k]).all();arrays[k].append(z[k].copy())
  files.append(dict(file=str(p),sha256=row['sha256'],states=n,source_family=row['source_family'],termination_reasons=row['termination_reasons']))
 data={k:torch.tensor(np.concatenate(v),device=device) for k,v in arrays.items()};assert len(data['state'])==4515
 meta=dict(states=4515,files=files,fit_families=sorted({r['source_family'] for r in rows}),validation_arrays_loaded=False,reference_export_approved=False,offline_transition_fit_only=True,collection_audit_sha256=hashlib.sha256(auditfile.read_bytes()).hexdigest(),query_repeat_audit_sha256=hashlib.sha256((yawroot/'full/independent_audit.json').read_bytes()).hexdigest(),source_plan_sha256=info['plan_sha256'],scope='All real native transitions including original terminal failures, no reference feasibility or policy acceptance claim.')
 return data,meta
