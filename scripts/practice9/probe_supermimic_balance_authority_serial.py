"""One fresh-process, one-world authority case; compare repeats externally."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np,torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import LIMITS,PHYSICS_CONTRACT,enable_tw_velocity_limits,disable_tw_velocity_limits
from supermimic.algo.state import SuperMimicStateSpec,StateExtractor
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--plan',type=Path,required=True);p.add_argument('--case-index',type=int,required=True);p.add_argument('--variant-index',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);plan=json.loads(a.plan.read_text());motions=plan['motions'];assert 0<=a.case_index<32 and 0<=a.variant_index<30
for r in motions:assert hashlib.sha256(Path(r['file']).read_bytes()).hexdigest()==r['sha256']
clip,phase=divmod(a.case_index,4);r=motions[clip];frame=int((r['frames']-1)*plan['phase_fractions'][phase]);cfg=tw_supermimic_env_cfg([r['file'] for r in motions],num_envs=1,play=True);cfg.scene.env_spacing=0.;cfg.events={};cfg.auto_reset=False;cfg.seed=42;cfg.commands['motion'].pose_range={};cfg.commands['motion'].velocity_range={};cfg.commands['motion'].joint_position_range=(0.,0.)
env=ManagerBasedRlEnv(cfg,device='cuda:0');enable_tw_velocity_limits(env)
try:
 robot=env.scene['robot'];cmd=env.command_manager.get_term('motion');term=env.action_manager.get_term('joint_pos');order=torch.arange(30,device='cuda:0')[term._target_ids];names=list(robot.joint_names);actions=[names[i] for i in order.tolist()];lower=[n for n in names if n.startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot'))];assert len(lower)==14 and term.cfg.clip is None
 delta=torch.zeros((1,30),device='cuda:0');joint=None;sign=0
 if a.variant_index>=2:joint=lower[(a.variant_index-2)//2];sign=1 if (a.variant_index-2)%2==0 else -1;delta[0,actions.index(joint)]=sign*.02
 spec=SuperMimicStateSpec.from_env(env,cmd.cfg.body_names);extract=StateExtractor(env,spec,torso_body_name='chest',left_foot_geom_pattern=r'^left_foot_collision(_\d+)?$',right_foot_geom_pattern=r'^right_foot_collision(_\d+)?$');caps=torch.tensor([LIMITS[n] for n in names],device='cuda:0');env.reset();cmd.reset_to_frame(torch.tensor([0],device='cuda:0'),clip,frame);env.sim.forward();cmd.update_relative_body_poses();data={k:[] for k in ['state','action','context','next_state','done','desired_joint_pos']}
 for step in range(min(8,r['frames']-frame-1)):
  s=extract.state().clone();ctx=extract.context().clone();target=cmd.joint_pos[:,order]+delta;action=(target-term._offset)/term._scale;assert float(abs(action*term._scale+term._offset-target).max())<=1e-6;result=env.step(action);nxt=extract.state().clone();assert bool((robot.data.joint_vel.abs()<=caps+1e-5).all())
  for k,v in [('state',s),('action',action),('context',ctx),('next_state',nxt),('done',(result[2]|result[3])),('desired_joint_pos',target)]:data[k].append(v[0].cpu().numpy().copy())
  if bool((result[2]|result[3]).any()):break
 arrays={k:np.stack(v) for k,v in data.items()};assert all(np.isfinite(v).all() for v in arrays.values());np.savez_compressed(a.output/'trace.npz',**arrays)
 receipt=dict(case_index=a.case_index,variant_index=a.variant_index,id=r['id'],split=r['split'],frame=frame,steps=len(arrays['state']),joint=joint,target_delta_rad=sign*.02,state_joint_names=names,action_joint_names=actions,scale=term._scale.cpu().tolist(),offset=term._offset.cpu().tolist(),source_sha256=r['sha256'],trace_sha256=hashlib.sha256((a.output/'trace.npz').read_bytes()).hexdigest(),asset_sha256=hashlib.sha256(Path(os.environ['TW_SUPERMIMIC_MJCF']).read_bytes()).hexdigest(),physics=PHYSICS_CONTRACT,done=bool(arrays['done'].any()))
 (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
finally:env.close();disable_tw_velocity_limits(env)
