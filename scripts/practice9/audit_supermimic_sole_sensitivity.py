"""Read-only fixed-branch sole orientation sensitivity, not a new training reward."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch,mujoco
from mjlab.envs import ManagerBasedRlEnv
from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.trainer import SuperMimicTrainer
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import enable_tw_velocity_limits,disable_tw_velocity_limits
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists();a.output.mkdir()
cases=['standing_retention_zero_readout_v1','standing_policy_anchor_v1'];base=Path('/root/gpufree-data/twrl_runs/supermimic_tw_20261031')
models={k:torch.load(base/k/'calibrated_retained/checkpoint_final.pt',map_location='cuda:0',weights_only=False) for k in cases}
c=models[cases[-1]];cfg=SuperMimicCfg(**c['cfg']);cfg.compile=False;cfg.wm_anchor_collection=None;cfg.policy_anchor_collection=None
asset=Path(os.environ['TW_SUPERMIMIC_MJCF']);assert hashlib.sha256(asset.read_bytes()).hexdigest()==cfg.extra['tw_mjcf_sha256']
manifest=Path('/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/static_probe/manifest.json');m=json.loads(manifest.read_text());assert m['synthetic_diagnostic']
collections=['action_response_v2','action_response_replay_v2_warmup10','action_response_replay_v2_warmup50','action_response_replay_v2_warmup100','action_response_smallscale_warmup50']
plan=dict(collections=collections,models={k:hashlib.sha256((base/k/'calibrated_retained/checkpoint_final.pt').read_bytes()).hexdigest() for k in cases},scope='All existing fixed one-step branches; sole score=sum(1-world_z_cosine), diagnostic only. No model, policy or reward updates; not full control-gradient or causality proof.')
(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
e=tw_supermimic_env_cfg([m['motions'][0]['file']],num_envs=1,play=True);e.seed=42;e.events={};e.auto_reset=False;e.commands['motion'].pose_range={};e.commands['motion'].velocity_range={};e.commands['motion'].joint_position_range=(0.,0.)
env=ManagerBasedRlEnv(e,device='cuda:0');enable_tw_velocity_limits(env);t=SuperMimicTrainer(env,cfg,log_dir=a.output/'runtime');t.load(base/cases[-1]/'calibrated_retained/checkpoint_final.pt');t.initialize()
try:
 env.reset();t.command.reset_to_frame(torch.tensor([0],device='cuda:0'),0,0);env.sim.forward();default_context=t.extractor.context().detach()
 indices=[t.kinematics.body_names.index(s+'_foot') for s in ['left','right']]
 for side in ['left','right']:
  gid=mujoco.mj_name2id(env.sim.mj_model,mujoco.mjtObj.mjOBJ_GEOM,'robot/'+side+'_foot_collision');np.testing.assert_allclose(env.sim.mj_model.geom_quat[gid],[1,0,0,0],atol=1e-7,rtol=0)
 def cosines(s):
  _,q=t.kinematics(s[:,:3],s[:,3:7],s[:,13:43]);q=torch.nn.functional.normalize(q[:,indices],dim=-1);return 1-2*q[:,:,1:3].square().sum(-1)
 def compare(x,y):
  den=np.linalg.norm(x)*np.linalg.norm(y)
  return dict(actual_norm=float(np.linalg.norm(x)),model_norm=float(np.linalg.norm(y)),cosine=float(x@y/den) if den>1e-15 else None,rmse=float(np.sqrt(np.mean((x-y)**2))))
 early=[]
 for case in cases:
  root=a.root/'learned_policy_native_contacts_v1'/case
  report=json.loads((root/'results.json').read_text());r=next(x for x in report['rows'] if x['agent']=='trained')
  with np.load(root/r['transition_file']) as z:states=torch.tensor(z['next_state'],device='cuda:0');actions=z['action'].copy()
  raw=json.loads((root/r['native_contact_file']).read_text())['samples'];actual=np.array([[x['feet'][s]['rotation_w'][2][2] for s in ['left','right']] for x in raw])
  fk=cosines(states).detach().cpu().numpy();error=float(np.max(np.abs(fk-actual)));assert error<1e-5
  scale=np.array(json.loads((a.root/'static_perturb_v1/control_mapping.json').read_text())['scale']).reshape(-1)
  samples=[]
  for step in sorted({1,5,10,20,len(actions)}):
   samples.append(dict(step=step,seconds=step*.02,normalized_action=actions[step-1].tolist(),target_delta_deg=np.degrees(actions[step-1]*scale).tolist(),sole_z_cosine=fk[step-1].tolist()))
  early.append(dict(case=case,action_joint_names=cfg.extra['tw_action_joint_names'],fk_vs_gpu_sole_z_max_abs_error=error,fixed_samples=samples))
 (a.output/'early_commands.json').write_text(json.dumps(early,indent=2))
 rows=[]
 for collection in collections:
  root=a.root/collection;cp=json.loads((root/'plan.json').read_text());path=root/'paired_responses.npz'
  with np.load(path) as z:data={k:z[k].copy() for k in z.files}
  states=torch.tensor(data['state'],device='cuda:0');actual=torch.tensor(data['next_state'],device='cuda:0')
  action=torch.zeros((len(states),30),device='cuda:0');action[torch.arange(len(states)),torch.tensor(data['joint'],device='cuda:0')]=torch.tensor(data['epsilon']*data['sign'],device='cuda:0',dtype=torch.float32)
  if 'action' in data:torch.testing.assert_close(action,torch.tensor(data['action'],device='cuda:0'),rtol=0,atol=0)
  context=torch.tensor(data['context'],device='cuda:0') if 'context' in data else default_context.expand(len(states),-1)
  actual_score=(1-cosines(actual)).sum(-1).detach().cpu().numpy()
  predicted={}
  for name,model in models.items():
   for key in ['tw_state_joint_names','tw_action_joint_names','tw_physics_contract','tw_joint_coordinate_contract']:assert model['cfg']['extra'][key]==cfg.extra[key]
   t.world_model.load_state_dict(model['world_model'],strict=True);t.world_model.eval()
   with torch.no_grad():predicted[name]=(1-cosines(t.world_model(states,action,context))).sum(-1).cpu().numpy()
  for eps in cp['epsilons']:
   ids=np.flatnonzero(data['epsilon']==eps);assert len(ids)==60
   assert data['joint'][ids].tolist()==np.repeat(np.arange(30),2).tolist() and data['sign'][ids].tolist()==[-1,1]*30
   real=(actual_score[ids][1::2]-actual_score[ids][::2])/(2*eps)
   row=dict(collection=collection,trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),epsilon=eps,actual_gradient=real.tolist(),initial_pair_state_max_abs_difference=float(np.max(np.abs(data['state'][ids][1::2]-data['state'][ids][::2]))),context_provenance='recorded' if 'context' in data else 'reconstructed deterministic event-free static context; limitation',models={})
   for name,scores in predicted.items():
    pred=(scores[ids][1::2]-scores[ids][::2])/(2*eps);assert np.isfinite(real).all() and np.isfinite(pred).all();row['models'][name]=dict(metrics=compare(real,pred),gradient=pred.tolist())
   rows.append(row)
  (a.output/'sensitivity.json').write_text(json.dumps(dict(plan=plan,rows=rows),indent=2))
 print(json.dumps([dict(collection=r['collection'],epsilon=r['epsilon'],models={k:v['metrics'] for k,v in r['models'].items()}) for r in rows]))
finally:t.close();env.close();disable_tw_velocity_limits(env)
