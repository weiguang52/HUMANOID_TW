"""Audited fit-only native labels at fixed predicted states, for offline calibration."""
import hashlib,json
from pathlib import Path
import numpy as np,torch

def load_neighborhood_fit(root,cfg,device='cuda:0'):
 root=Path(root).resolve();plan=json.loads((root/'query_plan.json').read_text())
 audit_file=root/'full/independent_audit.json';audit=json.loads(audit_file.read_text());capture=json.loads((root/'full/plan.json').read_text())
 assert audit['passed'] and not audit['smoke'] and audit['queries']==192 and audit['records']==768
 assert not plan['validation_arrays_loaded'] and plan['queries']==192 and plan['optimizer_updates']==0 and plan['weights_unchanged']
 assert capture['runtime_dense'] and capture['physics']==cfg['extra']['tw_physics_contract'] and capture['asset_sha256']==cfg['extra']['tw_mjcf_sha256']
 assert capture['query_plan_sha256']==hashlib.sha256((root/'query_plan.json').read_bytes()).hexdigest()
 assert capture['queries_sha256']==plan['query_sha256']==hashlib.sha256((root/'queries.npz').read_bytes()).hexdigest()
 collection=Path(plan['collection']);mapping=json.loads((collection/'control_mapping.json').read_text())
 for key,target in [('state_joint_names','tw_state_joint_names'),('action_joint_names','tw_action_joint_names')]:assert mapping[key]==cfg['extra'][target]
 manifest=json.loads(Path(plan['manifest']).read_text());fit={x['source_family'] for x in manifest['motions'] if x['split']=='fit'};val={x['source_family'] for x in manifest['motions'] if x['split']=='validation'}
 assert len(fit)==len(val)==4 and not fit&val
 assert len(plan['sources'])==16 and {x['source_family'] for x in plan['sources']}==fit
 for source in plan['sources']:
  f=(collection/source['trace']).resolve();assert f.is_relative_to(collection.resolve()) and hashlib.sha256(f.read_bytes()).hexdigest()==source['sha256']
 assert len(plan['rows'])==192 and all(x['split']=='fit' and x['source_family'] in fit for x in plan['rows'])
 file=root/'full/transitions.npz';sha=hashlib.sha256(file.read_bytes()).hexdigest();assert sha==audit['transitions_sha256']
 with np.load(file,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 query=d['query'];depth=np.array([plan['rows'][int(i)]['depth'] for i in query]);mask=(d['repeat']==0)&(depth>0)
 assert int(mask.sum())==288 and set(depth[mask])=={1,4,8}
 assert len(set(query[mask].tolist()))==144
 assert all(int(((depth==v)&mask).sum())==96 for v in [1,4,8])
 assert int((d['warm_source'][mask]==True).sum())==144 and int((d['warm_source'][mask]==False).sum())==144
 data={k:d[k][mask] for k in ['state','action','context','next_state']}
 for k,width in [('state',73),('action',30),('context',34),('next_state',73)]:assert data[k].shape==(288,width) and np.isfinite(data[k]).all()
 meta=dict(conditions=288,unique_queries=144,source_families=sorted(fit),validation_families_metadata_only=sorted(val),validation_arrays_loaded=False,transitions_sha256=sha,query_plan_sha256=capture['query_plan_sha256'],queries_sha256=plan['query_sha256'],native_audit_sha256=hashlib.sha256(audit_file.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(Path(plan['manifest']).read_bytes()).hexdigest(),generator_checkpoint_sha256=plan['checkpoint_sha256'],terminated_retained=int(d['terminated'][mask].sum()),joint_range_advisories_retained=int((d['input_joint_range_excess'][mask]>0).sum()),selection='Every predicted depth1/4/8, both warm conditions, repeat0 only; repeat1 independent replay audit. No termination/range filtering.',offline_calibration_approved=True,policy_promotion_approved=False)
 return {k:torch.tensor(v,device=device) for k,v in data.items()},meta
