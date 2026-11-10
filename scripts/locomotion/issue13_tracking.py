"""Read-only issue13 full-trial tracking moments and post-push time partitions."""
import hashlib,json,math,random
from pathlib import Path
import numpy as np
import torch
FIELDS=["xy_error","yaw_error","vx","vy","wz"]
SCOPES=["full_horizon_nonterminal","initial_episode_nonterminal"]
PARTS=["all","inside","outside","reward_aligned_inside","reward_aligned_outside"]
SENTINEL=-(2**60)
def window_masks(step,last,steps=50):
 age=step-last
 return (age>=0)&(age<steps),(age>0)&(age<=steps)
def digest(value):
 h=hashlib.sha256()
 def add(x):
  if isinstance(x,np.ndarray):
   h.update(str((str(x.dtype),tuple(x.shape))).encode());h.update(x.tobytes())
  elif isinstance(x,torch.Tensor):
   x=x.detach().cpu().contiguous();h.update(str((str(x.dtype),tuple(x.shape))).encode());h.update(x.numpy().tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=str):h.update(repr(k).encode());add(x[k])
  elif isinstance(x,(list,tuple)):
   h.update(type(x).__name__.encode())
   for v in x:add(v)
  else:h.update(repr(x).encode())
 add(value);return h.hexdigest()
def rng_state(device):
 return (torch.get_rng_state().clone(),torch.cuda.get_rng_state(device).clone() if str(device).startswith("cuda") else None,
         digest(np.random.get_state()),digest(random.getstate()))
def rng_same(a,b):
 assert torch.equal(a[0],b[0]) and (a[1] is None or torch.equal(a[1],b[1])) and a[2:]==b[2:],"Observer changed RNG"
class MomentAccumulator:
 def __init__(self,n,device="cpu"):
  self.count=torch.zeros((2,5,n),dtype=torch.long,device=device)
  self.sum=torch.zeros((2,5,n,5),dtype=torch.float64,device=device);self.sumsq=torch.zeros_like(self.sum)
 def add(self,values,valid,initial,inside,reward_inside):
  parts=torch.stack([torch.ones_like(valid),inside,~inside,reward_inside,~reward_inside])
  masks=torch.stack([parts&valid,parts&valid&initial])
  safe=torch.where(masks[:,:,:,None],values.double()[None,None],0.)
  self.count+=masks.long();self.sum+=safe;self.sumsq+=safe.square()
  return masks,safe.sum(2),safe.square().sum(2)
 def check(self):
  assert torch.equal(self.count[:,0],self.count[:,1]+self.count[:,2])
  assert torch.equal(self.count[:,0],self.count[:,3]+self.count[:,4])
  for a in [self.sum,self.sumsq]:
   assert torch.isfinite(a).all()
   for i,j in [(1,2),(3,4)]:assert torch.allclose(a[:,0],a[:,i]+a[:,j],atol=1e-9,rtol=1e-11)
class TrackingRecorder:
 def __init__(self,base,runner,output,checkpoint,condition,capture_env=-1,capture_command="forward_08"):
  self.base,self.runner,self.robot=base,runner,base.scene["robot"]
  self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
  self.checkpoint=checkpoint;self.condition=condition;self.capture_env=capture_env;self.capture_command=capture_command
  self.active=False;self.summaries=[];self.rng_checks=0
  before=rng_state(base.device);self.model_before=digest(runner.alg.save())
  from isaaclab.utils.io import dump_yaml
  dump_yaml(str(self.output/"env.yaml"),base.cfg)
  masses=self.robot.root_physx_view.get_masses().cpu().clone();nominal=self.robot.data.default_mass.cpu().clone()
  materials=self.robot.root_physx_view.get_material_properties().cpu().clone()
  np.savez_compressed(self.output/"startup_physics.npz",masses=masses.numpy(),nominal_masses=nominal.numpy(),materials=materials.numpy())
  assert torch.isfinite(materials).all() and (masses>0).all()
  action=base.action_manager.get_term("JointPositionAction")
  self.physics={"mass_equal_nominal":torch.equal(masses,nominal),
   "materials_equal_across_environments":torch.equal(materials,materials[:1].expand_as(materials)),
   "material_static_min":materials[:,:,0].min().item(),"material_static_max":materials[:,:,0].max().item(),
   "material_dynamic_min":materials[:,:,1].min().item(),"material_dynamic_max":materials[:,:,1].max().item(),
   "terrain_static":base.cfg.scene.terrain.physics_material.static_friction,
   "terrain_dynamic":base.cfg.scene.terrain.physics_material.dynamic_friction,
   "action_class":type(action).__name__,"pushes":base.cfg.events.push_robot is not None,
   "observation_noise":base.cfg.observations.policy.enable_corruption,
   "mass_event":base.cfg.events.add_base_mass is not None,"material_event":base.cfg.events.physics_material is not None}
  if condition=="clean":
   assert self.physics["mass_equal_nominal"] and self.physics["materials_equal_across_environments"]
   assert type(action).__name__=="JointPositionAction" and not hasattr(action,"delay")
   assert not any(self.physics[k] for k in ["pushes","observation_noise","mass_event","material_event"])
  else:assert condition=="combined" and type(action).__name__=="RandomOneStepDelayAction"
  self.original_compute=base.termination_manager.compute
  def compute():
   result=self.original_compute()
   if self.active:
    self.step+=1
    done=result.bool()
    self.last_push[done]=SENTINEL
    first=(self.first_failure<0)&base.termination_manager.terminated
    self.first_failure[first]=self.step
    self.initial_alive[done]=False
   return result
  base.termination_manager.compute=compute
  old=getattr(base,"_issue7_recorder",None)
  if old is not None:
   original_push=old.push
   def push(ids,delta,q):
    original_push(ids,delta,q)
    if self.active:
     assert old.step==self.step
     self.last_push[ids]=self.step
     self.push_rows.append(torch.stack([torch.full_like(ids,self.step),ids],dim=1).detach().clone())
   old.push=push
  elif condition!="clean":raise RuntimeError("Combined observer requires original issue11 full-trial recorder")
  rng_same(before,rng_state(base.device));self.rng_checks+=1
 def begin(self,label,target):
  assert not self.active
  before=rng_state(self.base.device)
  self.label,self.target=label,list(target);self.step=0;n=self.base.num_envs;device=self.base.device
  self.last_push=torch.full((n,),SENTINEL,dtype=torch.long,device=device)
  self.initial_alive=torch.ones(n,dtype=torch.bool,device=device);self.first_failure=torch.full((n,),-1,dtype=torch.long,device=device)
  self.acc=MomentAccumulator(n,device);self.push_rows=[];self.full_masks=[];self.step_sum=[];self.step_sumsq=[]
  self.ref_ids=torch.tensor(sorted({0,min(1,n-1),min(17,n-1),n-1}),device=device)
  self.reference=[];self.reference_masks=[];self.capture=[];self.capture_flags=[]
  self.active=True;rng_same(before,rng_state(device));self.rng_checks+=1
 def sample(self,values,valid,velocity,wz):
  assert self.active and self.step==len(self.full_masks)+1
  inside,reward_inside=window_masks(self.step,self.last_push,round(1/self.base.step_dt))
  fields=torch.stack([values["xy_error"],values["yaw_error"],velocity[:,0],velocity[:,1],wz],dim=1).detach()
  masks,sums,squares=self.acc.add(fields,valid,self.initial_alive,inside,reward_inside)
  self.full_masks.append(masks.detach().clone());self.step_sum.append(sums.detach());self.step_sumsq.append(squares.detach())
  self.reference.append(fields[self.ref_ids].clone());self.reference_masks.append(masks[:,:,self.ref_ids].clone())
  if self.capture_env>=0 and self.label==self.capture_command:
   i=self.capture_env;assert i<self.base.num_envs
   state=self.robot.data.root_state_w[i].clone();state[:3]-=self.base.scene.env_origins[i]
   self.capture.append(torch.cat([state,self.robot.data.joint_pos[i]]).detach().clone())
   self.capture_flags.append(torch.stack([valid[i],self.initial_alive[i],inside[i],reward_inside[i]]))
 def finish(self,result):
  assert self.active;before=rng_state(self.base.device);self.active=False;self.acc.check()
  assert self.step==round(20/self.base.step_dt)
  masks=torch.stack(self.full_masks);step_count=masks.sum(-1)
  ss=torch.stack(self.step_sum);sq=torch.stack(self.step_sumsq)
  counts=self.acc.count.cpu().numpy();sums=self.acc.sum.cpu().numpy();sumsq=self.acc.sumsq.cpu().numpy()
  reference=torch.stack(self.reference).cpu().numpy();refmask=torch.stack(self.reference_masks).cpu().numpy()
  refs=self.ref_ids.cpu().numpy()
  for si in range(2):
   for pi in range(5):
    for ri,env in enumerate(refs):
     chosen=reference[:,ri][refmask[:,si,pi,ri]]
     assert len(chosen)==counts[si,pi,env]
     assert np.allclose(chosen.astype("float64").sum(0),sums[si,pi,env],atol=1e-9,rtol=1e-11)
     assert np.allclose(np.square(chosen.astype("float64")).sum(0),sumsq[si,pi,env],atol=1e-9,rtol=1e-11)
  total=sums[0,0].sum(0);count=int(counts[0,0].sum())
  differences={k:abs(float(total[j]/count)-result["metrics"][k]) for j,k in enumerate(FIELDS[:2])}
  assert max(differences.values())<1e-10
  assert int((self.first_failure>=0).sum())==result["failed_trials"]
  pushes=torch.cat(self.push_rows).cpu().numpy() if self.push_rows else np.empty((0,2),dtype=np.int64)
  path=self.output/(self.label+".npz")
  np.savez_compressed(path,counts=counts,sums=sums,sumsq=sumsq,step_counts=step_count.cpu().numpy(),
   step_sums=ss.cpu().numpy(),step_sumsq=sq.cpu().numpy(),reference_env_ids=refs,reference_values=reference,
   reference_masks=refmask,packed_masks=np.packbits(masks.cpu().numpy(),axis=-1,bitorder="little"),
   push_steps_and_ids=pushes,first_failure_step=self.first_failure.cpu().numpy())
  meta={"command":self.label,"target":self.target,"num_envs":self.base.num_envs,"steps":self.step,"dt":self.base.step_dt,
   "fields":FIELDS,"scopes":SCOPES,"partitions":PARTS,"packed_mask_bitorder":"little",
   "observer_no_sampling_or_physics_writes":True,"reference_moment_reconstruction_passed":True,
   "original_metric_max_absolute_difference":differences,"first_failures":result["failed_trials"],
   "primary_window":"post-step [push,push+1s), ages0..49","reward_boundary_sensitivity":"ages1..50",
   "per_env_counts":[int((counts[si,pi]>0).sum()) for si in range(2) for pi in range(5)]}
  path.with_suffix(".json").write_text(json.dumps(meta,indent=2))
  if self.capture:
   target=self.output/(self.label+"_trajectory.npz");flags=torch.stack(self.capture_flags).cpu().numpy()
   np.savez_compressed(target,values=torch.stack(self.capture).cpu().numpy(),flags=flags,push_steps=pushes[pushes[:,1]==self.capture_env,0])
   target.with_suffix(".json").write_text(json.dumps({"command":self.label,"target":self.target,"env_id":self.capture_env,
    "checkpoint":self.checkpoint,"frames":self.step,"dt":self.base.step_dt,"joint_names":self.robot.joint_names,
    "fields":{"root_pose_local":[0,7],"root_velocity_world":[7,13],"joint_pos":[13,13+len(self.robot.joint_names)]},
    "source_sample":"exact post-step states, including any reset frames; no re-simulation in display",
    "first_failure_step":int(self.first_failure[self.capture_env]),"flags":["valid_nonterminal","initial_episode","window_inside","reward_window_inside"]},indent=2))
  rng_same(before,rng_state(self.base.device));self.rng_checks+=1;self.summaries.append(meta)
 def job_metadata(self):
  assert not self.active and self.summaries
  after=digest(self.runner.alg.save());assert after==self.model_before,"Inference changed model/optimizer state"
  return {"output":str(self.output),"condition":self.condition,"checkpoint_sha256":hashlib.sha256(Path(self.checkpoint).read_bytes()).hexdigest(),
   "model_optimizer_state_unchanged":True,"model_state_before_sha256":self.model_before,"model_state_after_sha256":after,
   "rng_invariance_checks":self.rng_checks,"physics_audit":self.physics,"commands":self.summaries,
   "training_updates":0,"version":"issue13_read_only_moments_v1"}
