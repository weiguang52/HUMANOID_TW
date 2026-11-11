"""Read-only issue15 yaw/heading/contact evidence layered on the frozen issue13 observer."""
import json,math
from pathlib import Path
import numpy as np
import torch
from issue13_tracking import TrackingRecorder,rng_state,rng_same
def yaw_from_quaternion(q):
 return torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
def wrapped_delta(a,b):
 return torch.atan2(torch.sin(a-b),torch.cos(a-b))
class YawRecorder(TrackingRecorder):
 def begin(self,label,target):
  super().begin(label,target)
  before=rng_state(self.base.device)
  self.start_yaw=yaw_from_quaternion(self.robot.data.root_quat_w).detach().clone()
  self.prev_yaw=self.start_yaw.clone()
  n=self.base.num_envs;d=self.base.device
  self.heading_sum=torch.zeros((2,n),dtype=torch.float64,device=d)
  self.heading_count=torch.zeros((2,n),dtype=torch.long,device=d)
  self.full_series=label in ["forward_03","forward_05","forward_08"]
  self.yaw_rates=[];self.heading_angles=[];self.foot_contacts=[];self.heading_refs=[]
  self.feet_ids=[self.base.scene["contact_forces"].body_names.index(n) for n in ["left_foot","right_foot"]]
  rng_same(before,rng_state(d))
 def sample(self,values,valid,velocity,wz):
  yaw=yaw_from_quaternion(self.robot.data.root_quat_w).detach()
  delta=wrapped_delta(yaw,self.prev_yaw).double()
  masks=torch.stack([valid,valid&self.initial_alive])
  self.heading_sum+=torch.where(masks,delta[None],0.)
  self.heading_count+=masks.long()
  self.prev_yaw=yaw.clone()
  self.heading_refs.append(torch.stack([yaw[self.ref_ids],delta[self.ref_ids].float()],dim=-1).clone())
  if self.full_series:
   contact=self.base.scene["contact_forces"].data.net_forces_w[:,self.feet_ids,:].norm(dim=-1)>1.
   self.yaw_rates.append(wz.detach().clone());self.heading_angles.append(yaw.clone());self.foot_contacts.append(contact.detach().clone())
  super().sample(values,valid,velocity,wz)
 def finish(self,result):
  before=rng_state(self.base.device)
  extra={"heading_start":self.start_yaw.cpu().numpy(),"heading_end":self.prev_yaw.cpu().numpy(),
   "heading_delta_sum":self.heading_sum.cpu().numpy(),"heading_count":self.heading_count.cpu().numpy(),
   "heading_reference_yaw_delta":torch.stack(self.heading_refs).cpu().numpy(),
   "reference_env_ids":self.ref_ids.cpu().numpy(),"first_failure_step":self.first_failure.cpu().numpy()}
  if self.full_series:
   extra.update(wz=torch.stack(self.yaw_rates).cpu().numpy(),heading_yaw=torch.stack(self.heading_angles).cpu().numpy(),
    packed_foot_contacts=np.packbits(torch.stack(self.foot_contacts).cpu().numpy(),axis=1,bitorder="little"))
  for si in range(2):
   assert torch.equal(self.heading_count[si],self.acc.count[si,0])
  # Endpoint heading is meaningful only for trials without reset over the full20s.
  survived=extra["first_failure_step"]<0
  endpoint=wrapped_delta(self.prev_yaw,self.start_yaw).cpu().numpy()
  assert np.allclose(np.arctan2(np.sin(extra["heading_delta_sum"][0,survived]),np.cos(extra["heading_delta_sum"][0,survived])),endpoint[survived],atol=2e-5)
  np.savez_compressed(self.output/(self.label+"_yaw.npz"),**extra)
  self.output.joinpath(self.label+"_yaw.json").write_text(json.dumps({
   "dt":self.base.step_dt,"steps":self.step,"num_envs":self.base.num_envs,"full_series":self.full_series,
   "yaw_error_formula":"mean valid post-step abs(root_ang_vel_b[:,2] - commanded_yaw_rate), units rad/s; exactly original evaluator",
   "yaw_body_rate_note":"Body-z angular velocity is not generally Euler heading derivative under tilt.",
   "heading_formula":"Sum atan2(sin(yaw_t-yaw_previous),cos(yaw_t-yaw_previous)) over nonterminal steps; reset jumps excluded. Both full episode-concatenated and initial-episode prefixes retained. Full20s endpoint statistics only on no-reset survivors, with denominator/falls and censoring caveat.",
   "contact_formula":"left/right foot net world force norm >1N, same as evaluator; packed axis1 little endian",
   "observer_no_sampling_or_writes":True},indent=2)+"\n")
  super().finish(result)
  rng_same(before,rng_state(self.base.device))
