"""Evaluate a frozen world model on real development trajectories; no fitting."""
import argparse,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from supermimic.algo.networks import WorldModel,WorldModelCfg
from evaluate_supermimic_tw import audit_world_model
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pose-integration-fit-audit',type=Path);a=p.parse_args()
assert not a.output.exists()
c=torch.load(a.checkpoint,map_location='cpu',weights_only=False);w=c['world_model'];cfg=c['cfg'];extra=cfg['extra'];mapping=json.loads((a.collection/'control_mapping.json').read_text())
assert mapping['state_joint_names']==extra['tw_state_joint_names']
assert mapping['action_joint_names']==extra['tw_action_joint_names']
m=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda();m.load_state_dict(w,strict=True);m.eval()
from supermimic_pose_integration_diagnostic import apply_offline_prediction_contract
m=apply_offline_prediction_contract(m,c)
if a.pose_integration_fit_audit is not None:
 from supermimic_pose_integration_diagnostic import PoseIntegrationDiagnostic
 m=PoseIntegrationDiagnostic(m,a.pose_integration_fit_audit,hashlib.sha256(a.checkpoint.read_bytes()).hexdigest())
trainer=SimpleNamespace(world_model=m,device='cuda:0',env=SimpleNamespace(step_dt=.02),spec=SimpleNamespace(joint_names=mapping['state_joint_names']))
rows=[]
for r in json.loads((a.collection/'results.json').read_text())['rows']:
 with np.load(a.collection/r['trace']) as d:
  metrics=audit_world_model(trainer,{k:d[k] for k in d.files});s=d['state'];n=d['next_state'];baseline={}
  for h in (1,8,32):
   count=len(s)-h+1
   if count>0:
    err=np.linalg.norm(s[:count,:3]+h*.02*s[:count,7:10]-n[h-1:,:3],axis=-1);baseline[str(h)]=dict(windows=count,root_position_p95_m=float(np.quantile(err,.95)))
 rows.append(dict(r,world_model=metrics,constant_velocity=baseline))
report=dict(checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),rows=rows,scope='Frozen old model; development source families separated for future fitting. Old policy trained on both partitions, so not a fresh generalization test. No fitting, reset-crossing or padding.')
if a.pose_integration_fit_audit is not None:report['pose_integration_diagnostic']=m.metadata
if c.get('prediction_contract') is not None:report['prediction_contract']=c['prediction_contract']
a.output.write_text(json.dumps(report,indent=2)+'\n')
for split in ('fit','validation'):
 rr=[r for r in rows if r['split']==split];print(split,len(rr),'H32 root p95 median',float(np.median([r['world_model']['32']['root_position_error_p95_m'] for r in rr if r['world_model']['32']['windows']>0])))
