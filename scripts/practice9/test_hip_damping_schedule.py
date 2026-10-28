import sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
import torch
from types import SimpleNamespace as NS
from locomotion_damping import LocomotionDamping,speed_gain
names=['left_hip_pitch_joint','right_hip_pitch_joint','left_ankle_yaw_joint','right_ankle_yaw_joint']
class Robot:
 def __init__(self):
  self.joint_names=names;self.data=NS(joint_damping=torch.ones(2,4))
  self.actuators={'leg_major':NS(joint_names=names[::-1],damping=torch.ones(2,4))};self.writes=[]
 def write_joint_damping_to_sim(self,value,joint_ids):
  self.writes.append((value.clone(),list(joint_ids)))
  self.data.joint_damping[:,joint_ids]=value
r=Robot();cmd=NS(cfg=NS(body_names=['base_link']),body_lin_vel_w=torch.zeros(2,1,3))
env=NS(scene={'robot':r},command_manager=NS(get_term=lambda _:cmd))
a=LocomotionDamping(env,names[:2],1.1);a.step();assert not r.writes
cmd.body_lin_vel_w[:,0,0]=torch.tensor([.075,.1]);a.step()
assert torch.allclose(r.data.joint_damping[:,:2],torch.tensor([[1.05,1.05],[1.1,1.1]]))
assert torch.equal(r.data.joint_damping[:,2:],torch.ones(2,2))
assert torch.equal(r.actuators['leg_major'].damping[:,:2],torch.ones(2,2))
assert torch.equal(r.actuators['leg_major'].damping[:,2:],r.data.joint_damping[:,:2].flip(1))
a.reset();assert torch.equal(r.data.joint_damping,torch.ones(2,4)) and not a.records
for gain in [float('nan'),float('inf'),.9]:
 try:LocomotionDamping(env,names[:2],gain)
 except ValueError:pass
 else:raise AssertionError('Invalid gain accepted')
b=LocomotionDamping(env,names[:2],1.);before=len(r.writes);b.step();assert len(r.writes)==before
v=torch.linspace(0,.2,1001);u=((v-.05)/.05).clamp(0,1)
assert torch.equal(speed_gain(v),1.+.1*u*u*(3.-2.*u))
assert torch.equal(speed_gain(torch.tensor([0.,.05]),1.1),torch.ones(2))
print('PASS hip-only schedule, endpoints, independent envs, reset, synchronized buffers, identity and legacy default')
