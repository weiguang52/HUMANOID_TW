"""Full initial-trial metrics, phase, state digests, and fixed env0 video states."""
import numpy as np,torch,json,hashlib
from issue19_tracking import YawRecorder as Previous
from issue18_tracking import YawRecorder as PhaseObserver
from issue20_runtime import snapshot
class YawRecorder(Previous):
 def begin(self,label,target):
  super().begin(label,target);self.phases20=[];self.state_digests=[];self.replay20=[];self.initial20=snapshot(self.base)
  np.savez_compressed(self.output/(label+'_initial_state.npz'),env_origins=self.base.scene.env_origins.cpu().numpy(),**{k:v.cpu().numpy() for k,v in self.initial20.items()})
 def sample(self,values,valid,velocity,wz):
  k=self.base.cfg.issue20_phase_offset;self.phases20.append(((self.projected_policy.pre_action_episode_step+k)%20).detach().to(torch.int16).clone())
  state=snapshot(self.base);h=hashlib.sha256()
  for x in [self.projected_policy.last_output,*state.values()]:h.update(x.detach().cpu().numpy().tobytes())
  self.state_digests.append(h.hexdigest());self.replay20.append(torch.cat([state['root'][0,:7]-torch.cat([self.base.scene.env_origins[0],state['root'].new_zeros(4)]),state['joint_pos'][0]]).detach().clone())
  return PhaseObserver.sample(self,values,valid,velocity,wz)
 def finish(self,result):
  np.savez_compressed(self.output/(self.label+'_phase20.npz'),pre_action_idx=torch.stack(self.phases20).cpu().numpy())
  np.savez_compressed(self.output/(self.label+'_replay_env0.npz'),root_and_joints=torch.stack(self.replay20).cpu().numpy(),initial_root=self.initial20['root'][0].cpu().numpy(),initial_joint_pos=self.initial20['joint_pos'][0].cpu().numpy())
  (self.output/(self.label+'_step_sha256.json')).write_text(json.dumps(self.state_digests))
  (self.output/'mirrored_reset_audit.json').write_text(json.dumps(self.base._issue20_reset_audit))
  return PhaseObserver.finish(self,result)
