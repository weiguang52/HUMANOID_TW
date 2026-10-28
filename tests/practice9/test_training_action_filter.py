from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/rsl_rl'))
from training_action_filter import VectorAnkleFilter
from policy_action_filter import LegActionFilter
names=['left_ankle_yaw_joint','left_ankle_pitch_joint','right_ankle_yaw_joint','right_ankle_pitch_joint','waist']
g=torch.Generator().manual_seed(71);f=VectorAnkleFilter(names,3,'cpu');single=[LegActionFilter(names,'cpu',.5,'ankles') for _ in range(3)]
for step in range(1000):
 a=torch.randn(3,5,generator=g);before=a.clone();out=f(a);expected=torch.cat([single[i](a[i:i+1]) for i in range(3)])
 assert torch.equal(out,expected) and torch.equal(a,before)
 done=torch.tensor([step%7==0,step%11==0,step%17==0]);f.reset(done)
 for i in range(3):single[i].reset(done[i:i+1])
print('1000-step vector/scalar bitwise equivalence with independent resets passed')
