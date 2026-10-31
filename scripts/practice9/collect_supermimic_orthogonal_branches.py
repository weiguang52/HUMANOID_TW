"""Fixed static PD orthogonal action branches, explicitly not trajectory windows."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import enable_tw_velocity_limits,disable_tw_velocity_limits,PHYSICS_CONTRACT,LIMITS
from supermimic.algo.state import SuperMimicStateSpec,StateExtractor
p=argparse.ArgumentParser(description=__doc__)
for key in ['manifest','source-checkpoint','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
m=json.loads(a.manifest.read_text());assert m['synthetic_diagnostic'] and len(m['motions'])==1
c=torch.load(a.source_checkpoint,map_location='cpu',weights_only=False);extra=c['cfg']['extra']
assert extra['tw_physics_contract']==PHYSICS_CONTRACT and extra['tw_mjcf_sha256']==hashlib.sha256(Path(os.environ['TW_SUPERMIMIC_MJCF']).read_bytes()).hexdigest()
plan=dict(contract='branch_transitions_not_episodes_v1',fit_warmups=[5,25,75,150],validation_warmups=[15,35,125,175],epsilons=[.02,.05],num_envs=60,seed=42,replay_atol=1e-5,sign_assignments=2,steps_per_branch=1,source_checkpoint_sha256=hashlib.sha256(a.source_checkpoint.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),scope='Static temporal split with convergent/nearby states; not independent motion-source or hardware generalization. Never interpret batches as contiguous episodes.')
(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
e=tw_supermimic_env_cfg([m['motions'][0]['file']],num_envs=60,play=True);e.seed=42;e.events={};e.auto_reset=False;e.commands['motion'].pose_range={};e.commands['motion'].velocity_range={};e.commands['motion'].joint_position_range=(0.,0.)
env=ManagerBasedRlEnv(e,device='cuda:0');enable_tw_velocity_limits(env);robot=env.scene['robot'];cmd=env.command_manager.get_term('motion');term=env.action_manager.get_term('joint_pos')
spec=SuperMimicStateSpec.from_env(env,cmd.cfg.body_names);extract=StateExtractor(env,spec,torso_body_name='chest',left_foot_geom_pattern=r'^left_foot_collision(_\d+)?$',right_foot_geom_pattern=r'^right_foot_collision(_\d+)?$')
assert list(robot.joint_names)==extra['tw_state_joint_names'] and list(term.target_names)==extra['tw_action_joint_names']
(a.output/'control_mapping.json').write_text(json.dumps(dict(state_joint_names=list(robot.joint_names),action_joint_names=list(term.target_names),offset=term._offset.cpu().tolist(),scale=term._scale.cpu().tolist()),indent=2))
zero=torch.zeros((60,30),device='cuda:0');ids=torch.arange(60,device='cuda:0');caps=torch.tensor([LIMITS[n] for n in robot.joint_names],device='cuda:0');reports=[]
def reset(warmup):
 env.reset();cmd.reset_to_frame(ids,0,0);env.sim.forward()
 for _ in range(warmup):
  out=env.step(zero);assert not bool((out[2]|out[3]).any()), 'Warmup failure; preserve logs, no retry'
 return extract.state().clone(),extract.context().clone()
def step(action):
 out=env.step(action);n=extract.state().clone();assert bool((robot.data.joint_vel.abs()<=caps+1e-5).all());return n,(out[2]|out[3]).clone()
def save(name,**arrays):
 values={k:v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else np.asarray(v) for k,v in arrays.items()};assert all(np.isfinite(v).all() for v in values.values());np.savez_compressed(a.output/name,**values)
try:
 assert abs(env.step_dt-.02)<1e-9
 for split,warmups in [('fit',plan['fit_warmups']),('validation',plan['validation_warmups'])]:
  for warmup in warmups:
   s0,ctx0=reset(warmup);n0,done0=step(zero);assert not bool(done0.any()), 'Nominal PD zero response terminated'
   basefile=f'{split}_warmup{warmup}_zero.npz';save(basefile,state=s0,context=ctx0,action=zero,next_state=n0,terminated=done0)
   reports.append(dict(split=split,warmup=warmup,kind='zero',file=basefile,count=60))
   for eps in plan['epsilons']:
    for assignment in range(2):
     s,ctx=reset(warmup);error=float((s-s0).abs().max());torch.testing.assert_close(s,s0,rtol=0,atol=plan['replay_atol']);torch.testing.assert_close(ctx,ctx0,rtol=0,atol=1e-7)
     joint=np.repeat(np.arange(30),2);sign=np.tile([-1,1],30)*(1 if assignment==0 else -1)
     u=zero.clone();u[ids,torch.tensor(joint,device='cuda:0')]=torch.tensor(sign*eps,device='cuda:0',dtype=torch.float32)
     n,done=step(u);name=f'{split}_warmup{warmup}_eps{eps:.2f}_assignment{assignment}.npz'
     save(name,state=s,context=ctx,action=u,next_state=n,joint=joint,sign=sign,terminated=done,zero_state=s0,zero_next_state=n0)
     r=dict(split=split,warmup=warmup,kind='branch',epsilon=eps,assignment=assignment,file=name,count=60,max_replay_state_error=error,terminated_count=int(done.sum()))
     reports.append(r);print(json.dumps(r),flush=True);(a.output/'partial.json').write_text(json.dumps(reports,indent=2))
 (a.output/'results.json').write_text(json.dumps(dict(contract=plan['contract'],rows=reports),indent=2));(a.output/'status').write_text('collected_pending_audit\n')
except BaseException:
 (a.output/'status').write_text('failed\n');raise
finally:env.close();disable_tw_velocity_limits(env)
