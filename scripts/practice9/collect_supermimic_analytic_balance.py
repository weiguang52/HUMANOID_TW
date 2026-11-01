"""Matched bounded-reference/contact-balance short real-loop diagnostic."""
import argparse,hashlib,json,os
from pathlib import Path
import mujoco,numpy as np,torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import LIMITS,PHYSICS_CONTRACT,enable_tw_velocity_limits,disable_tw_velocity_limits
from supermimic.algo.state import SuperMimicStateSpec,StateExtractor
from supermimic_native_contacts import native_foot_contacts
from supermimic_contact_balance import ContactBalance
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=['bounded_reference','contact_balance'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);plan=json.loads(a.plan.read_text());motions=plan['motions'];references=[]
for r in motions:
 assert hashlib.sha256(Path(r['file']).read_bytes()).hexdigest()==r['sha256']
 with np.load(r['file']) as z:references.append({k:z[k].copy() for k in z.files})
cfg=tw_supermimic_env_cfg([r['file'] for r in motions],num_envs=1,play=True);cfg.scene.env_spacing=0.;cfg.events={};cfg.auto_reset=False;cfg.seed=42;cfg.commands['motion'].pose_range={};cfg.commands['motion'].velocity_range={};cfg.commands['motion'].joint_position_range=(0.,0.)
env=ManagerBasedRlEnv(cfg,device='cuda:0');enable_tw_velocity_limits(env);robot=env.scene['robot'];cmd=env.command_manager.get_term('motion');term=env.action_manager.get_term('joint_pos');order=torch.arange(30,device='cuda:0')[term._target_ids];names=list(robot.joint_names);caps=torch.tensor([LIMITS[n] for n in names],device='cuda:0');assert float(abs(env.scene.env_origins).max())==0 and term.cfg.clip is None
model=mujoco.MjModel.from_xml_path(os.environ['TW_SUPERMIMIC_MJCF']);controller=ContactBalance(model,names);spec=SuperMimicStateSpec.from_env(env,cmd.cfg.body_names);extract=StateExtractor(env,spec,torso_body_name='chest',left_foot_geom_pattern=r'^left_foot_collision(_\d+)?$',right_foot_geom_pattern=r'^right_foot_collision(_\d+)?$');reports=[]
(a.output/'input.json').write_text(json.dumps(dict(arm=a.arm,smoke=a.smoke,plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),asset_sha256=hashlib.sha256(Path(os.environ['TW_SUPERMIMIC_MJCF']).read_bytes()).hexdigest(),physics=PHYSICS_CONTRACT,state_joint_names=names,action_joint_names=[names[i] for i in order.tolist()],scale=term._scale.cpu().tolist(),offset=term._offset.cpu().tolist(),seed=42,contact_timing='action step k uses poststep contacts k-1; step1 empty after explicit reset'),indent=2)+'\n')
try:
 for clip,r in enumerate(motions[:1] if a.smoke else motions):
  ref=references[clip];base=ref['body_names'].tolist().index('base_link')
  for fraction in ([0.] if a.smoke else plan['phase_fractions']):
   frame=int((r['frames']-1)*fraction);env.reset();cmd.reset_to_frame(torch.tensor([0],device='cuda:0'),clip,frame);env.sim.forward();cmd.update_relative_body_poses();prior={s:dict(contacts=[]) for s in ['left','right']};records={k:[] for k in ['state','action','context','next_state','nominal_joint_pos','target_joint_pos','terminated','truncated','reference_frame']};logs=[];reasons=[];planned=min(3 if a.smoke else plan['max_steps'],r['frames']-frame-1)
   for step in range(planned):
    index=int(cmd.time_steps[0]);assert index==frame+step;s=extract.state().clone();ctx=extract.context().clone();nominal=cmd.joint_pos[0].cpu().numpy().copy();target,info=controller.target(s[0].cpu().numpy(),nominal,ref['root_quat_wxyz'][index],ref['body_ang_vel_w'][index,base],prior,a.arm=='contact_balance');target=torch.tensor(target[order.cpu().numpy()],device='cuda:0',dtype=torch.float32)[None];action=(target-term._offset)/term._scale;assert float(abs(action*term._scale+term._offset-target).max())<=1e-6
    result=env.step(action);nxt=extract.state().clone();feet=native_foot_contacts(env);assert torch.equal(nxt,extract.state());assert bool((robot.data.joint_vel.abs()<=caps+1e-5).all())
    actual_target=(action*term._scale+term._offset)[0];state_target=torch.empty_like(actual_target);state_target[order]=actual_target
    for k,v in [('state',s[0]),('action',action[0]),('context',ctx[0]),('next_state',nxt[0]),('nominal_joint_pos',torch.tensor(nominal)),('target_joint_pos',state_target),('terminated',result[2][0:1]),('truncated',result[3][0:1]),('reference_frame',torch.tensor([index]))]:records[k].append(v.detach().cpu().numpy().copy())
    logs.append(dict(action_step=step+1,contact_source_step=step,prior_feet=prior,post_feet=feet,solver=info));prior=feet
    if bool((result[2]|result[3]).any()):reasons=[n for n in env.termination_manager.active_terms if bool(env.termination_manager.get_term(n)[0])];break
   filename=r['id']+'_frame'+str(frame);arrays={k:np.stack(v) for k,v in records.items()};assert all(np.isfinite(v).all() for v in arrays.values());assert np.array_equal(arrays['next_state'][:-1],arrays['state'][1:]);np.savez_compressed(a.output/(filename+'.npz'),**arrays);(a.output/(filename+'_control.json')).write_text(json.dumps(logs,indent=2)+'\n');row=dict(id=r['id'],split=r['split'],source_family=r['source_family'],frame=frame,steps=len(logs),planned_steps=planned,termination_reasons=reasons,trace=filename+'.npz',control_file=filename+'_control.json',trace_sha256=hashlib.sha256((a.output/(filename+'.npz')).read_bytes()).hexdigest(),control_sha256=hashlib.sha256((a.output/(filename+'_control.json')).read_bytes()).hexdigest());reports.append(row);(a.output/'partial.json').write_text(json.dumps(dict(rows=reports),indent=2)+'\n');print(row,flush=True)
 (a.output/'results.json').write_text(json.dumps(dict(rows=reports,scope='Short analytic control diagnostic only, not learned policy or fixed7 acceptance.'),indent=2)+'\n')
except BaseException:
 if 'records' in locals() and records['state']:np.savez_compressed(a.output/'interrupted_trace.npz',**{k:np.stack(v) for k,v in records.items() if v})
 if 'logs' in locals():(a.output/'interrupted_control.json').write_text(json.dumps(logs,indent=2)+'\n')
 raise
finally:env.close();disable_tw_velocity_limits(env)
