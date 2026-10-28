"""Exercise actual Gym wrappers with asynchronous episode resets and eval filters."""
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
import torch,gymnasium as gym
from training_leg_filter import wrap_leg_environment,validate_resume_contract
from training_action_filter import wrap_environment,VectorAnkleFilter
from candidate_leg_filter import CandidateLegFilter
names=[s+'_'+j for s in ['left','right'] for j in ['hip_pitch_joint','hip_roll_joint','hip_yaw_joint','knee_pitch_joint','ankle_yaw_joint','ankle_pitch_joint','foot_roll']]+['other'+str(i) for i in range(16)]
class Dummy(gym.Env):
 def __init__(self):
  self.device='cpu';self.num_envs=2;self.index=0
  self.action_manager=SimpleNamespace(get_term=lambda _:SimpleNamespace(_joint_names=names))
 def reset(self,**kwargs):self.index=0;return {},{}
 def step(self,actions):
  self.last_actions=actions.clone();self.index+=1
  done=torch.tensor([self.index in [1,4],self.index==3])
  return {},torch.ones(2),done,torch.zeros(2,dtype=torch.bool),{}
base=Dummy();env=wrap_leg_environment(wrap_environment(base,.5),.8)
manual_leg=CandidateLegFilter(names,'cpu',.8);manual_ankle=VectorAnkleFilter(names,2,'cpu',.5)
env.reset();torch.manual_seed(42)
for _ in range(6):
 raw=torch.randn(2,30);snapshot=raw.clone();expected=manual_ankle(manual_leg(raw))
 result=env.step(raw);assert torch.equal(base.last_actions,expected)
 assert torch.equal(raw,snapshot)
 manual_leg.reset(result[2]|result[3]);manual_ankle.reset(result[2]|result[3])
 raw.add_(10000)  # caller mutation cannot corrupt either filter's history
# Explicit reset makes the next action identity through both filters.
env.reset();raw=torch.randn(2,30);env.step(raw);assert torch.equal(base.last_actions,raw)
contract=env.leg_action_filter.audit
assert contract['scope']=='legs14' and contract['order']=='candidate legs then existing ankles'
validate_resume_contract(contract,.8);validate_resume_contract(None,None)
for alpha in [None,.7,float('nan')]:
 try:validate_resume_contract(contract,alpha)
 except ValueError:pass
 else:raise AssertionError('Changed resumed filter accepted')
for field,value in [('mode','teacher_residual'),('scope','hips'),('order','reversed')]:
 bad=dict(contract);bad[field]=value
 try:validate_resume_contract(bad,.8)
 except ValueError:pass
 else:raise AssertionError('Invalid saved contract accepted')
print('Actual nested Gym wrappers equal eval leg+ankle transforms, async resets, explicit reset, immutable history and resume contract passed')
