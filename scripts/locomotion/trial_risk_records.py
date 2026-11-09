"""Issue11 opt-in all-trial observer. No sampling or simulation writes."""
import json,hashlib,random
from pathlib import Path
import numpy as np
import torch
from failure_traces import FailureRecorder
from isaaclab.utils.math import quat_apply_inverse

def rng_snapshot(device):
 return (torch.get_rng_state().clone(),torch.cuda.get_rng_state(device).clone() if str(device).startswith('cuda') else None)

def assert_rng_equal(a,b):
 assert torch.equal(a[0],b[0]) and (a[1] is None or torch.equal(a[1],b[1])),'observer changed Torch RNG'

def state_audit(base,path,label):
 r=base.scene['robot'];p=Path(path);p.mkdir(parents=True,exist_ok=True)
 ns=np.random.get_state();ps=random.getstate()
 np.savez_compressed(p/(label+'.npz'),root_state=r.data.root_state_w.cpu().numpy(),joint_pos=r.data.joint_pos.cpu().numpy(),joint_vel=r.data.joint_vel.cpu().numpy(),torch_rng=torch.get_rng_state().numpy(),cuda_rng=torch.cuda.get_rng_state(base.device).numpy(),numpy_rng=ns[1],numpy_rng_aux=np.asarray(ns[2:],dtype=np.float64),numpy_rng_algorithm=np.frombuffer(ns[0].encode('ascii'),dtype=np.uint8),python_rng=np.asarray(ps[1],dtype=np.uint64),python_rng_version=np.asarray([ps[0]],dtype=np.int64),python_rng_gauss_present=np.asarray([ps[2] is not None],dtype=np.uint8),python_rng_gauss_value=np.asarray([0. if ps[2] is None else ps[2]],dtype=np.float64))

class TrialRiskRecorder(FailureRecorder):
 def __init__(self,base,output):
  super().__init__(base,output)
  self.shape_counts=[]
  for link in self.robot.root_physx_view.link_paths[0]:
   view=self.robot._physics_sim_view.create_rigid_body_view(link)
   self.shape_counts.append(view.max_shapes)
  assert len(self.shape_counts)==len(self.robot.body_names)
  assert sum(self.shape_counts)==self.robot.root_physx_view.max_shapes
  self.shape_map={};start=0
  for name,count in zip(self.robot.body_names,self.shape_counts):self.shape_map[name]=list(range(start,start+count));start+=count
  self.base_id=self.robot.body_names.index('base_link')
 def begin(self,label,target):
  before=rng_snapshot(self.env.device);super().begin(label,target)
  self.push_rows=[];self.push_fields=None
  self.first_failure_step=torch.full((self.env.num_envs,),-1,dtype=torch.long,device=self.env.device)
  self.masses=self.robot.root_physx_view.get_masses().cpu().clone()
  self.nominal=self.robot.data.default_mass.cpu().clone()
  self.mass_gpu=self.masses.to(self.env.device)
  self.materials=self.robot.root_physx_view.get_material_properties().cpu().clone()
  assert self.masses.shape==self.nominal.shape and (self.masses>0).all()
  assert torch.isfinite(self.materials).all()
  foot_shapes=self.shape_map['left_foot']+self.shape_map['right_foot'];assert foot_shapes
  self.foot_static_min=self.materials[:,foot_shapes,0].min(-1).values
  self.foot_dynamic_min=self.materials[:,foot_shapes,1].min(-1).values
  self.rng_checks=1;assert_rng_equal(before,rng_snapshot(self.env.device))
 def compute(self):
  done=super().compute()
  if self.recording:
   fresh=self.failed&(self.first_failure_step<0)
   self.first_failure_step[fresh]=self.step
  return done
 def before_push(self,ids):
  if not self.recording:return
  before=rng_snapshot(self.env.device);r=self.robot;m=self.mass_gpu
  self.pre_com_velocity=((r.data.body_com_vel_w[:,:,:3]*m[:,:,None]).sum(1)/m.sum(1)[:,None])[ids].clone()
  assert_rng_equal(before,rng_snapshot(self.env.device))
 def push(self,ids,delta,q):
  if not self.recording:return
  before=rng_snapshot(self.env.device);super().push(ids,delta,q)
  ids=ids.to(device=self.env.device,dtype=torch.long);r=self.robot;env=self.env;n=len(ids)
  phase=torch.where(self.period>0,((self.step-self.onset)/self.period.clamp_min(1)).clamp(0,1),-1.)
  contact=self.previous_contact[ids];support=contact[:,0].long()+2*contact[:,1].long()
  masses=self.mass_gpu
  com=(r.data.body_com_pos_w*masses[:,:,None]).sum(1)/masses.sum(1)[:,None]
  com_vel=(r.data.body_com_vel_w[:,:,:3]*masses[:,:,None]).sum(1)/masses.sum(1)[:,None]
  forces=env.scene['contact_forces'].data.net_forces_w[ids][:,self.feet,:].norm(dim=-1)
  limits=r.data.soft_joint_pos_limits[ids];near=((r.data.joint_pos[ids]<=limits[:,:,0]+.02)|(r.data.joint_pos[ids]>=limits[:,:,1]-.02)).sum(-1)
  parts=[('env_id',ids[:,None].float()),('push_step',torch.full((n,1),float(self.step),device=env.device)),('first_trial_at_risk',(~self.failed[ids])[:,None].float()),('delta_world',delta),('delta_body',quat_apply_inverse(q,delta)),('root_quaternion',q),('root_position_local',r.data.root_pos_w[ids]-env.scene.env_origins[ids]),('root_velocity_before',r.data.root_vel_w[ids,:3]-delta),('root_velocity_after',r.data.root_vel_w[ids,:3]),('root_angular_velocity',r.data.root_vel_w[ids,3:]),('whole_body_com_local',com[ids]-env.scene.env_origins[ids]),('whole_body_com_velocity_before',self.pre_com_velocity),('whole_body_com_velocity_readback',com_vel[ids]),('foot_positions_local',(r.data.body_pos_w[ids][:,self.foot_bodies,:]-env.scene.env_origins[ids,None,:]).reshape(n,6)),('support_code',support[:,None].float()),('contact',contact.float()),('contact_force_norm',forces),('cycle_phase',phase[ids]),('delay20ms',self.action.delay[ids,:1].float()),('near_soft_limit_count',near[:,None].float())]
  values=torch.cat([v for _,v in parts],dim=-1).detach().clone()
  if self.push_fields is None:
   self.push_fields={};k=0
   for name,value in parts:self.push_fields[name]=[k,k+value.shape[1]];k+=value.shape[1]
   assert k==values.shape[1]
  self.push_rows.append(values);self.rng_checks+=1;assert_rng_equal(before,rng_snapshot(self.env.device))
 def finish(self):
  before=rng_snapshot(self.env.device);result=super().finish();self.rng_checks+=1
  push=torch.cat(self.push_rows).cpu().numpy() if self.push_rows else np.zeros((0,0),dtype=np.float32)
  path=self.output/(self.label+'_trials.npz')
  np.savez_compressed(path,env_ids=np.arange(self.env.num_envs,dtype=np.int32),masses=self.masses.numpy(),nominal_masses=self.nominal.numpy(),mass_ratios=(self.masses/self.nominal).numpy(),materials=self.materials.numpy(),foot_static_min=self.foot_static_min.numpy(),foot_dynamic_min=self.foot_dynamic_min.numpy(),initial_delay20ms=self.initial_delay.cpu().numpy(),failed=self.failed.cpu().numpy(),first_failure_step=self.first_failure_step.cpu().numpy(),pushes=push)
  terrain=self.env.cfg.scene.terrain.physics_material
  meta={'num_envs':self.env.num_envs,'command':self.label,'step_dt':self.dt,'horizon_seconds':20,'body_names':self.robot.body_names,'shape_counts_per_body':self.shape_counts,'body_shape_indices':self.shape_map,'push_fields':self.push_fields,'all_push_rows':len(push),'all_trials':self.env.num_envs,'failed_trials':int(self.failed.sum()),'torch_rng_invariance_checks':self.rng_checks,'material_columns':['static_friction','dynamic_friction','restitution'],'material_definition':'actual per-shape assigned PhysX coefficients; foot minima are analysis summaries, not inferred contact forces','terrain_material':{'static_friction':terrain.static_friction,'dynamic_friction':terrain.dynamic_friction,'friction_combine_mode':terrain.friction_combine_mode},'mass_definition':'actual PhysX link mass / nominal link mass; foot masses may be fixed','post_failure_definition':'all pushes retained; first_trial_at_risk=false after first failure, excluded from risk denominators','com_definition':'mass-weighted link COM getters read at push; velocity is immediate readback, not a counterfactual stable post-impulse state','phase_definition':'same as FailureRecorder, contact onset inferred, not prescribed gait phase','observer_no_sampling_or_physics_writes':True}
  path.with_suffix('.json').write_text(json.dumps(meta,indent=2));assert_rng_equal(before,rng_snapshot(self.env.device));result['all_trial_records']=str(path);return result
