"""Validate native GPU position-actuator controls/forces; no CPU contact recompute."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch,warp as wp,mujoco
from mjlab.envs import ManagerBasedRlEnv
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.robot import GROUPS,ACTION_SCALE
from supermimic.config.tw.velocity_limits import enable_tw_velocity_limits,disable_tw_velocity_limits,PHYSICS_CONTRACT,LIMITS
from supermimic.algo.state import SuperMimicStateSpec,StateExtractor
p=argparse.ArgumentParser(description=__doc__)
for key in ['manifest','source-checkpoint','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
c=torch.load(a.source_checkpoint,map_location='cpu',weights_only=False);extra=c['cfg']['extra'];asset=Path(os.environ['TW_SUPERMIMIC_MJCF']);assert extra['tw_physics_contract']==PHYSICS_CONTRACT and extra['tw_mjcf_sha256']==hashlib.sha256(asset.read_bytes()).hexdigest()
m=json.loads(a.manifest.read_text());assert m['synthetic_diagnostic'] and len(m['motions'])==1
plan=dict(warmups=[0,10],amplitudes=[.02,.05,8.],num_envs=61,dt=.02,formula='gain0*ctrl+bias0+bias1*cached_length+bias2*cached_velocity, then native force clamp',stress_scope='u8 only verifies saturation algebra; all original terminations retained, not a skill test or training data',physics_contract=PHYSICS_CONTRACT,asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
e=tw_supermimic_env_cfg([m['motions'][0]['file']],num_envs=61,play=True);e.seed=42;e.events={};e.auto_reset=False;e.commands['motion'].pose_range={};e.commands['motion'].velocity_range={};e.commands['motion'].joint_position_range=(0.,0.)
env=ManagerBasedRlEnv(e,device='cuda:0');enable_tw_velocity_limits(env);robot=env.scene['robot'];cmd=env.command_manager.get_term('motion');term=env.action_manager.get_term('joint_pos');mm=env.sim.mj_model;data=env.sim.wp_data
spec=SuperMimicStateSpec.from_env(env,cmd.cfg.body_names);extract=StateExtractor(env,spec,torso_body_name='chest',left_foot_geom_pattern=r'^left_foot_collision(_\d+)?$',right_foot_geom_pattern=r'^right_foot_collision(_\d+)?$')
assert mm.nu==30 and list(robot.joint_names)==extra['tw_state_joint_names'] and list(term.target_names)==extra['tw_action_joint_names']
assert np.all(mm.actuator_dyntype==int(mujoco.mjtDyn.mjDYN_NONE)) and np.all(mm.actuator_gaintype==int(mujoco.mjtGain.mjGAIN_FIXED)) and np.all(mm.actuator_biastype==int(mujoco.mjtBias.mjBIAS_AFFINE)) and np.all(mm.actuator_trntype==int(mujoco.mjtTrn.mjTRN_JOINT))
assert not np.any(mm.actuator_ctrllimited) and np.all(mm.actuator_forcelimited)
np.testing.assert_array_equal(mm.actuator_gear[:,0],np.ones(30));np.testing.assert_array_equal(mm.actuator_gear[:,1:],np.zeros((30,5)))
expected={name:(kp,kd,effort,armature,scale) for names,kp,kd,effort,armature,scale in GROUPS for name in names}
actuator_names=[mm.joint(int(mm.actuator_trnid[i,0])).name.rsplit('/',1)[-1] for i in range(30)];assert len(set(actuator_names))==30
action_indices=torch.tensor([list(term.target_names).index(n) for n in actuator_names],device='cuda:0');dof_indices=torch.tensor([int(mm.jnt_dofadr[int(mm.actuator_trnid[i,0])]) for i in range(30)],device='cuda:0')
for i,name in enumerate(actuator_names):
 kp,kd,limit,arm,scale=expected[name];np.testing.assert_allclose([mm.actuator_gainprm[i,0],mm.actuator_biasprm[i,0],mm.actuator_biasprm[i,1],mm.actuator_biasprm[i,2]],[kp,0,-kp,-kd],rtol=0,atol=1e-7);np.testing.assert_allclose(mm.actuator_forcerange[i],[-limit,limit],rtol=0,atol=1e-7)
 torch.testing.assert_close(term._scale[:,list(term.target_names).index(name)],torch.full((61,),ACTION_SCALE[name],device='cuda:0'),rtol=0,atol=1e-7)
metadata=dict(actuator_joint_names=actuator_names,state_joint_names=list(robot.joint_names),action_joint_names=list(term.target_names),action_index_for_actuator=action_indices.cpu().tolist(),dof_for_actuator=dof_indices.cpu().tolist(),gainprm=mm.actuator_gainprm.tolist(),biasprm=mm.actuator_biasprm.tolist(),forcerange=mm.actuator_forcerange.tolist(),gear=mm.actuator_gear.tolist(),offset=term._offset.cpu().tolist(),scale=term._scale.cpu().tolist(),physics_timestep_s=float(mm.opt.timestep),decimation=int(e.decimation),timing='Derived native fields refreshed by env.step forward after integration; confirm alignment in audit')
(a.output/'control_metadata.json').write_text(json.dumps(metadata,indent=2))
zero=torch.zeros((61,30),device='cuda:0');ids=torch.arange(61,device='cuda:0');caps=torch.tensor([LIMITS[n] for n in robot.joint_names],device='cuda:0');gain=torch.tensor(mm.actuator_gainprm[:,0],device='cuda:0',dtype=torch.float32);bias=torch.tensor(mm.actuator_biasprm[:,:3],device='cuda:0',dtype=torch.float32);ranges=torch.tensor(mm.actuator_forcerange,device='cuda:0',dtype=torch.float32);reports=[]
try:
 for warmup in plan['warmups']:
  for amplitude in plan['amplitudes']:
   env.reset();cmd.reset_to_frame(ids,0,0);env.sim.forward()
   for _ in range(warmup):
    result=env.step(zero);assert not bool((result[2]|result[3]).any())
   s=extract.state().clone();u=zero.clone();u[ids[1:],torch.tensor(np.repeat(np.arange(30),2),device='cuda:0')]=torch.tensor(np.tile([-1,1],30)*amplitude,device='cuda:0',dtype=torch.float32)
   result=env.step(u);n=extract.state().clone();assert torch.isfinite(n).all() and bool((robot.data.joint_vel.abs()<=caps+1e-5).all())
   expected_target=term._offset+term._scale*u;target_error=float((expected_target-term._processed_actions).abs().max());torch.testing.assert_close(expected_target,term._processed_actions,rtol=0,atol=1e-7)
   ctrl=wp.to_torch(data.ctrl).clone();length=wp.to_torch(data.actuator_length).clone();velocity=wp.to_torch(data.actuator_velocity).clone();force=wp.to_torch(data.actuator_force).clone();generalized=wp.to_torch(data.qfrc_actuator).clone();torch.cuda.synchronize()
   expected_ctrl=expected_target[:,action_indices];ctrl_error=float((ctrl-expected_ctrl).abs().max());torch.testing.assert_close(ctrl,expected_ctrl,rtol=0,atol=1e-7)
   raw=ctrl*gain+bias[:,0]+bias[:,1]*length+bias[:,2]*velocity;analytic=torch.maximum(torch.minimum(raw,ranges[:,1]),ranges[:,0]);force_error=float((analytic-force).abs().max())
   expected_generalized=torch.zeros_like(generalized);expected_generalized[:,dof_indices]=force;generalized_error=float((generalized-expected_generalized).abs().max())
   saturated=(raw<ranges[:,0])|(raw>ranges[:,1]);filename=f'warmup{warmup}_amplitude{amplitude:.2f}.npz';arrays={k:v.detach().cpu().numpy() for k,v in dict(state_before=s,next_state=n,action=u,expected_target=expected_target,ctrl=ctrl,cached_length=length,cached_velocity=velocity,native_force=force,analytic_force=analytic,unclamped_force=raw,generalized_force=generalized,saturated=saturated,terminated=result[2],truncated=result[3]).items()};assert all(np.isfinite(v).all() for v in arrays.values());np.savez_compressed(a.output/filename,**arrays)
   row=dict(warmup=warmup,amplitude=amplitude,file=filename,target_abs_error=target_error,ctrl_abs_error=ctrl_error,force_abs_error=force_error,generalized_force_abs_error=generalized_error,saturated_components=int(saturated.sum()),terminated_count=int(result[2].sum()),truncated_count=int(result[3].sum()),termination_counts={name:int(env.termination_manager.get_term(name).sum()) for name in env.termination_manager.active_terms})
   reports.append(row);(a.output/'partial.json').write_text(json.dumps(reports,indent=2));print(json.dumps(row),flush=True)
 (a.output/'results.json').write_text(json.dumps(dict(plan=plan,rows=reports),indent=2));(a.output/'status').write_text('collected_pending_audit\n')
except BaseException:
 (a.output/'status').write_text('failed\n');raise
finally:env.close();disable_tw_velocity_limits(env)
