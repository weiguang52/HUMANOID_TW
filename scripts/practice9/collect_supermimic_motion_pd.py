"""Deterministic native reference-target PD snippets; never learned walking."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import enable_tw_velocity_limits,disable_tw_velocity_limits,LIMITS,PHYSICS_CONTRACT
from supermimic.algo.state import SuperMimicStateSpec,StateExtractor
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
plan=json.loads(a.plan.read_text());motions=plan['motions'];assert len(motions)==8
assert len({x['source_family'] for x in motions})==8
for x in motions:assert hashlib.sha256(Path(x['file']).read_bytes()).hexdigest()==x['sha256']
(a.output/'input.json').write_text(json.dumps(dict(plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),smoke=a.smoke,physics=PHYSICS_CONTRACT,asset_sha256=hashlib.sha256(Path(os.environ['TW_SUPERMIMIC_MJCF']).read_bytes()).hexdigest()),indent=2))
cfg=tw_supermimic_env_cfg([x['file'] for x in motions],num_envs=1,play=True)
cfg.events={};cfg.seed=42;cfg.auto_reset=False
cfg.commands['motion'].pose_range={};cfg.commands['motion'].velocity_range={};cfg.commands['motion'].joint_position_range=(0.,0.)
env=ManagerBasedRlEnv(cfg,device='cuda:0');enable_tw_velocity_limits(env)
robot=env.scene['robot'];cmd=env.command_manager.get_term('motion');term=env.action_manager.get_term('joint_pos')
order=torch.arange(30,device='cuda:0')[term._target_ids];assert len(order)==30 and len(torch.unique(order))==30
spec=SuperMimicStateSpec.from_env(env,cmd.cfg.body_names)
extract=StateExtractor(env,spec,torso_body_name='chest',left_foot_geom_pattern=r'^left_foot_collision(_\d+)?$',right_foot_geom_pattern=r'^right_foot_collision(_\d+)?$')
caps=torch.tensor([LIMITS[n] for n in robot.joint_names],device='cuda:0')
assert term.cfg.clip is None,'Need explicit clipped-target audit before collection'
assert torch.all(torch.as_tensor(term._scale)!=0)
(a.output/'control_mapping.json').write_text(json.dumps(dict(state_joint_names=list(robot.joint_names),action_joint_names=[robot.joint_names[i] for i in order.cpu().tolist()],offset=term._offset.cpu().tolist(),scale=term._scale.cpu().tolist(),clip=repr(term.cfg.clip),step_dt=env.step_dt),indent=2))
reports=[]
try:
 for clip_id,motion in enumerate(motions[:2] if a.smoke else motions):
  phases=[0.] if a.smoke else plan['phase_fractions']
  for fraction in phases:
   frame=int((motion['frames']-1)*fraction);steps=min(3 if a.smoke else plan['max_steps'],motion['frames']-frame-1)
   env.reset();cmd.reset_to_frame(torch.tensor([0],device='cuda:0'),clip_id,frame);env.sim.forward()
   records={k:[] for k in ['state','action','context','next_state','desired_joint_pos','terminated','truncated']};reasons=[];max_error=0.
   for step in range(steps):
    desired=cmd.joint_pos.clone();target=desired[:,order]
    action=(target-term._offset)/term._scale
    error=float((action*term._scale+term._offset-target).abs().max());assert error<=1e-6;max_error=max(max_error,error)
    state=extract.state().clone();context=extract.context().clone();result=env.step(action);nxt=extract.state().clone()
    assert torch.all(robot.data.joint_vel.abs()<=caps+1e-5)
    for k,v in [('state',state),('action',action),('context',context),('next_state',nxt),('desired_joint_pos',desired),('terminated',result[2][:,None]),('truncated',result[3][:,None])]:
     assert torch.isfinite(v).all();records[k].append(v[0].detach().cpu().numpy().copy())
    if bool((result[2]|result[3])[0]):
     reasons=[n for n in env.termination_manager.active_terms if bool(env.termination_manager.get_term(n)[0])];break
   name=motion['id']+'_frame'+str(frame)+'.npz'
   arrays={k:np.stack(v) for k,v in records.items()};assert np.array_equal(arrays['next_state'][:-1],arrays['state'][1:])
   np.savez_compressed(a.output/name,**arrays)
   row=dict(id=motion['id'],split=motion['split'],source_family=motion['source_family'],reference_sha256=motion['sha256'],frame=frame,phase_fraction=fraction,planned_steps=steps,steps=len(arrays['state']),termination_reasons=reasons,trace=name,target_inverse_max_error=max_error,sha256=hashlib.sha256((a.output/name).read_bytes()).hexdigest(),learned_policy=False)
   reports.append(row);(a.output/'partial.json').write_text(json.dumps(dict(rows=reports),indent=2));print(json.dumps(row),flush=True)
 (a.output/'results.json').write_text(json.dumps(dict(rows=reports,scope='Native deterministic reference-target PD snippets; no learned walking, no policy acceptance; all failures retained'),indent=2))
 (a.output/'status').write_text('collected_pending_independent_audit\n')
except BaseException:
 (a.output/'status').write_text('failed\n')
 if 'records' in locals() and records['state']:np.savez_compressed(a.output/'interrupted_trace.npz',**{k:np.stack(v) for k,v in records.items() if v})
 raise
finally:env.close();disable_tw_velocity_limits(env)
