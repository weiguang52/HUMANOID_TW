"""Local real-physics lower-body target response; no gain tuning or policy training."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import LIMITS,enable_tw_velocity_limits,disable_tw_velocity_limits
from supermimic.algo.state import SuperMimicStateSpec,StateExtractor
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);plan=json.loads(a.plan.read_text());motions=plan['motions']
for r in motions:assert hashlib.sha256(Path(r['file']).read_bytes()).hexdigest()==r['sha256']
cfg=tw_supermimic_env_cfg([r['file'] for r in motions],num_envs=30,play=True)
if plan.get('origin_layout')=='coincident_independent_worlds':cfg.scene.env_spacing=0.
cfg.events={};cfg.auto_reset=False;cfg.seed=42;cfg.commands['motion'].pose_range={};cfg.commands['motion'].velocity_range={};cfg.commands['motion'].joint_position_range=(0.,0.)
env=ManagerBasedRlEnv(cfg,device='cuda:0');enable_tw_velocity_limits(env);robot=env.scene['robot']
if plan.get('origin_layout')=='coincident_independent_worlds':assert float(abs(env.scene.env_origins).max())==0.
cmd=env.command_manager.get_term('motion');term=env.action_manager.get_term('joint_pos');order=torch.arange(30,device='cuda:0')[term._target_ids];names=list(robot.joint_names);action_names=[names[i] for i in order.tolist()];lower=[n for n in names if n.startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot'))];assert len(lower)==14 and term.cfg.clip is None
spec=SuperMimicStateSpec.from_env(env,cmd.cfg.body_names);extract=StateExtractor(env,spec,torso_body_name='chest',left_foot_geom_pattern=r'^left_foot_collision(_\d+)?$',right_foot_geom_pattern=r'^right_foot_collision(_\d+)?$');caps=torch.tensor([LIMITS[n] for n in names],device='cuda:0');offset=torch.zeros((30,30),device='cuda:0');variants=[dict(kind='zero'),dict(kind='zero_repeat')]
for j,n in enumerate(lower):
 for k,sign in enumerate([1,-1]):offset[2+j*2+k,action_names.index(n)]=sign*.02;variants.append(dict(joint=n,sign=sign,target_delta_rad=sign*.02))
(a.output/'input.json').write_text(json.dumps(dict(plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),state_joint_names=names,action_joint_names=action_names,variants=variants,smoke=a.smoke,scope='Real local target authority only; simultaneous worlds may differ numerically; duplicate zero and initial states must pass original1e-5 pairing before causal use.'),indent=2)+'\n');rows=[]
try:
 for clip,r in enumerate(motions[:1] if a.smoke else motions):
  for fraction in ([0.] if a.smoke else plan['phase_fractions']):
   frame=int((r['frames']-1)*fraction);env.reset();cmd.reset_to_frame(torch.arange(30,device='cuda:0'),clip,frame);env.sim.forward();cmd.update_relative_body_poses();initial=extract.state().clone();ctx=extract.context().clone();initial_error=float(abs(initial-initial[:1]).max());data={k:[] for k in ['state','action','context','next_state','done']}
   steps=min(2 if a.smoke else 8,r['frames']-frame-1)
   for step in range(steps):
    s=extract.state().clone();target=cmd.joint_pos[:,order]+offset;action=(target-term._offset)/term._scale;assert float(abs(action*term._scale+term._offset-target).max())<=1e-6;result=env.step(action);nxt=extract.state().clone();assert bool((robot.data.joint_vel.abs()<=caps+1e-5).all())
    for k,v in [('state',s),('action',action),('context',ctx),('next_state',nxt),('done',(result[2]|result[3]))]:data[k].append(v.cpu().numpy().copy())
    if bool((result[2]|result[3]).any()):break
   arrays={k:np.stack(v) for k,v in data.items()};duplicate=float(abs(arrays['next_state'][:,0]-arrays['next_state'][:,1]).max());name=r['id']+'_frame'+str(frame)+'.npz';np.savez_compressed(a.output/name,**arrays);row=dict(id=r['id'],split=r['split'],frame=frame,steps=len(arrays['state']),initial_pairing_max_error=initial_error,context_pairing_max_error=float(abs(ctx-ctx[:1]).max()),duplicate_zero_max_error=duplicate,pairing_passed=bool(initial_error<=1e-5 and float(abs(ctx-ctx[:1]).max())<=1e-5 and duplicate<=1e-5),trace=name,sha256=hashlib.sha256((a.output/name).read_bytes()).hexdigest(),any_termination=bool(arrays['done'].any()));rows.append(row);(a.output/'partial.json').write_text(json.dumps(dict(rows=rows),indent=2));print(row,flush=True)
 (a.output/'results.json').write_text(json.dumps(dict(rows=rows,scope='Local action response diagnostics; no policy, world-model or fixed7 acceptance.'),indent=2)+'\n')
finally:env.close();disable_tw_velocity_limits(env)
