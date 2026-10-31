"""Empirical action support and frozen-model accuracy on all real eval episodes."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from evaluate_supermimic_tw import audit_world_model
from supermimic.algo.networks import WorldModel,WorldModelCfg
p=argparse.ArgumentParser(description=__doc__)
for name in ['collection','evaluation','dynamics-checkpoint','output']:
 p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
mapping=json.loads((a.collection/'control_mapping.json').read_text())
source=json.loads((a.collection/'results.json').read_text())['rows']
fits=[row for row in source if row['split']=='fit'];assert fits
paths=[(a.collection/row['trace']).resolve() for row in fits]
assert len(set(paths))==len(paths)
assert set(paths).isdisjoint({(a.collection/row['trace']).resolve() for row in source if row['split']!='fit'})
parts=[];fit_provenance=[]
for path in paths:
 assert path.is_relative_to(a.collection.resolve())
 with np.load(path,allow_pickle=False) as z:actions=z['action'].copy()
 assert actions.ndim==2 and actions.shape[1]==30 and len(actions)>0 and np.isfinite(actions).all()
 parts.append(actions);fit_provenance.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
fit=np.concatenate(parts);low=fit.min(0);high=fit.max(0)
c=torch.load(a.dynamics_checkpoint,map_location='cpu',weights_only=False)
w=c['world_model'];cfg=c['cfg'];extra=cfg['extra']
assert mapping['state_joint_names']==extra['tw_state_joint_names']
assert mapping['action_joint_names']==extra['tw_action_joint_names']
model=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda()
model.load_state_dict(w,strict=True);model.eval()
t=SimpleNamespace(world_model=model,device='cuda:0',env=SimpleNamespace(step_dt=.02),spec=SimpleNamespace(joint_names=mapping['state_joint_names']))
evaluation=json.loads((a.evaluation/'results.json').read_text());rows=[]
policy_path=Path(evaluation['checkpoint'])
assert hashlib.sha256(policy_path.read_bytes()).hexdigest()==evaluation['checkpoint_sha256']
policy=torch.load(policy_path,map_location='cpu',weights_only=False)
for key in ['tw_state_joint_names','tw_action_joint_names','tw_joint_coordinate_contract','tw_physics_contract','tw_context_layout']:
 assert policy['cfg']['extra'][key]==extra[key],key
# Keep the provenance check separate from the counterfactual dynamics model.
del policy
keys=[(row['id'],row['agent']) for row in evaluation['rows']];assert len(set(keys))==len(keys)
for row in evaluation['rows']:
 path=(a.evaluation/row['transition_file']).resolve();assert path.is_relative_to(a.evaluation.resolve())
 with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in ['state','next_state','action','context']}
 count=len(data['state']);assert count==row['steps'] and count>0
 for key,width in [('state',73),('next_state',73),('action',30),('context',34)]:
  assert data[key].shape==(count,width) and np.isfinite(data[key]).all()
 assert np.array_equal(data['next_state'][:-1],data['state'][1:])
 assert row['joint_names']==mapping['state_joint_names']
 outside=(data['action']<low-1e-6)|(data['action']>high+1e-6)
 rows.append(dict(id=row['id'],agent=row['agent'],actual_steps=count,actual_seconds=count*.02,
  terminated=row['terminated'],termination_reasons=row['termination_reasons'],
  action_abs_max=float(np.abs(data['action']).max()),fraction_components_outside_fit=float(outside.mean()),
  steps_with_any_component_outside_fit=int(outside.any(1).sum()),trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
  world_model=audit_world_model(t,data)))
report=dict(matmul_precision=torch.get_float32_matmul_precision(),allow_tf32=torch.backends.cuda.matmul.allow_tf32,scope='Empirical per-component fit ranges only, not support-density or failure-causality proof. All actual eval episodes; no padding/reset-crossing. Model may differ from policy checkpoint: counterfactual frozen prediction only.',
 model_checkpoint_sha256=hashlib.sha256(a.dynamics_checkpoint.read_bytes()).hexdigest(),
 actual_policy_checkpoint_sha256=evaluation['checkpoint_sha256'],fit_records=len(fit),fit_files=fit_provenance,
 fit_action_abs_max=float(np.abs(fit).max()),fit_action_low=low.tolist(),fit_action_high=high.tolist(),rows=rows)
a.output.write_text(json.dumps(report,indent=2));print(json.dumps([{k:v for k,v in row.items() if k!='world_model'} for row in rows]))
