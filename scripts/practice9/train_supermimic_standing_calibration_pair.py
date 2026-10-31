"""Explicit isolated continuation pilots; final checkpoints only, no promotion."""
import argparse,hashlib,json,os
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends
from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.trainer import SuperMimicTrainer
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import enable_tw_velocity_limits,disable_tw_velocity_limits,PHYSICS_CONTRACT,LIMITS
p=argparse.ArgumentParser(description=__doc__)
for name in ['checkpoint','dynamics-checkpoint','manifest','output','run-root']:
 p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--branch',choices=['control','calibrated','calibrated_retained'],required=True)
p.add_argument('--anchor-collection',type=Path)
p.add_argument('--policy-anchor-collection',type=Path)
p.add_argument('--zero-readout',action='store_true')
p.add_argument('--pilot-updates',type=int,choices=[4,100],default=100)
p.add_argument('--num-envs',type=int,choices=[8,256],default=256)
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
configure_torch_backends()
c=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
d=torch.load(a.dynamics_checkpoint,map_location='cpu',weights_only=False)
assert c['iteration']==1000 and d['calibration_updates']==1000
assert d['kind']=='offline_dynamics_calibration_not_policy_checkpoint'
assert d['calibration_plan']['source_checkpoint_sha256']==hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
for k,v in d['world_model'].items():
 if not k.startswith('net.'):assert torch.equal(v,c['world_model'][k]),k
cfg=SuperMimicCfg(**c['cfg']);extra=cfg.extra
assert extra['tw_physics_contract']==PHYSICS_CONTRACT and extra['tw_joint_speed_limits']==LIMITS
assert extra['tw_mjcf_sha256']==hashlib.sha256(Path(os.environ['TW_SUPERMIMIC_MJCF']).read_bytes()).hexdigest()
m=json.loads(a.manifest.read_text());assert m['synthetic_diagnostic']
files=tuple(row['file'] for row in m['motions'])
assert extra['tw_motion_input_sha256']==hashlib.sha256(json.dumps(files).encode()).hexdigest()
cfg.num_iterations=1000+a.pilot_updates;cfg.save_interval=a.pilot_updates;cfg.experiment_name=a.branch;cfg.log_dir=str(a.run_root)
assert not (a.run_root/a.branch).exists()
if a.branch=='calibrated_retained':
 assert a.anchor_collection is not None
 cfg.wm_anchor_collection=str(a.anchor_collection.resolve());cfg.wm_anchor_fraction=.5
else:assert a.anchor_collection is None
if a.policy_anchor_collection is not None:
 assert a.branch=='calibrated_retained' and a.zero_readout and len(files)==1
 import numpy as np
 with np.load(files[0],allow_pickle=False) as ref:
  for key in ['joint_pos','body_pos_w','body_quat_w']:
   assert np.allclose(ref[key],ref[key][0],rtol=0,atol=1e-7)
  for key in ['joint_vel','body_lin_vel_w','body_ang_vel_w']:assert np.abs(ref[key]).max()<1e-7
 cfg.policy_anchor_collection=str(a.policy_anchor_collection.resolve());cfg.policy_anchor_fraction=.5
 cfg.extra['tw_policy_anchor_reference_sha256']=hashlib.sha256(Path(files[0]).read_bytes()).hexdigest()
plan=dict(branch=a.branch,updates=a.pilot_updates,initial_iteration=1000,final_iteration=cfg.num_iterations,seed=42,num_envs=a.num_envs,
 source_checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
 calibrated_sha256=hashlib.sha256(a.dynamics_checkpoint.read_bytes()).hexdigest(),
 differing_variable='Initial dynamics net only; both reset WM optimizer state; actor/critic optimizer and RNG restored from identical source',
 learning_rate='Preserve source cosine schedule and final ratio, no restart of schedule',
 selection='Final1100 only; evaluate original20s window, stop at first original termination; no automatic promotion',
 limitation='Single training seed continuation, not3 independent training seeds')
if a.branch=='calibrated_retained':
 plan['differing_variable']='Relative to prior calibrated arm, half same-size dynamics batch uses fit-only anchor windows; all other mechanisms unchanged'
 plan['anchor_collection']=cfg.wm_anchor_collection;plan['anchor_fraction']=.5
if a.policy_anchor_collection is not None:
 plan['differing_variable']='Relative to zero-readout retained arm, half same-size policy imagination-start batch uses fit-only past-history windows; all other mechanisms unchanged'
 plan['policy_anchor_collection']=cfg.policy_anchor_collection;plan['policy_anchor_fraction']=.5
(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
e=tw_supermimic_env_cfg(files,num_envs=a.num_envs,play=False);e.seed=42;e.episode_length_s=30.
env=ManagerBasedRlEnv(e,device=cfg.device);enable_tw_velocity_limits(env)
try:
 assert list(env.scene['robot'].joint_names)==extra['tw_state_joint_names']
 assert list(env.action_manager.get_term('joint_pos').target_names)==extra['tw_action_joint_names']
 t=SuperMimicTrainer(env,cfg,log_dir=a.run_root);t.load(a.checkpoint)
 for k,v in t.actor_critic.state_dict().items():assert torch.equal(v.cpu(),c['actor_critic'][k]),k
 if a.branch.startswith('calibrated'):t.world_model.load_state_dict(d['world_model'],strict=True)
 if t.wm_anchor is not None:
  plan['anchor_source']=t.wm_anchor.metadata
  plan['dynamics_batch']=dict(total=cfg.wm_batch_size,fit=t.wm_anchor_batch,online=cfg.wm_batch_size-t.wm_anchor_batch)
  (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
 if a.zero_readout:
  from supermimic_actor_initialization import zero_readout
  initial=zero_readout(t.actor_critic,t.actor_optimizer)
  with torch.no_grad():
   for value in [0.,1.]:
    action=t.actor_critic.act(torch.full((1,t.obs_dim),value,device=t.device),torch.full((1,t.goal_dim),value,device=t.device))
    assert torch.isfinite(action).all() and bool((action==0).all())
  initial['nominal_output_probe_passed']=True
  (a.output/'initialization_check.json').write_text(json.dumps(initial,indent=2))
  plan['actor_initialization_check']=initial
  (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
 if t.policy_anchor is not None:
  plan['policy_anchor_source']=t.policy_anchor.metadata
  count=int(cfg.policy_batch_size*cfg.policy_anchor_fraction)
  plan['policy_start_batch']=dict(total=cfg.policy_batch_size,fit=count,online=cfg.policy_batch_size-count)
  (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
 t.wm_optimizer.state.clear()
 cfg.extra['tw_standing_calibration_pilot']=plan
 (a.output/'status').write_text('training\n')
 try:
  t.train();assert t.iteration==cfg.num_iterations
  if t.policy_anchor is not None:
   assert t._last_policy_anchor_count==int(cfg.policy_batch_size*cfg.policy_anchor_fraction)
   (a.output/'policy_anchor_check.json').write_text(json.dumps(dict(actual_fit_start_count=t._last_policy_anchor_count,total=cfg.policy_batch_size,history_contract='past_only_shared_bias_reset_padding_v2',source=t.policy_anchor.metadata),indent=2))
  t.save('checkpoint_final.pt')
  (a.output/'status').write_text('trained_not_evaluated\n')
 except BaseException:
  t.save('checkpoint_interrupt.pt');(a.output/'status').write_text('failed\n');raise
 finally:t.close()
finally:env.close();disable_tw_velocity_limits(env)
