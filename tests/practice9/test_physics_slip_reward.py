from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/rsl_rl'))
from physics_slip_reward import PhysicsSlipAccumulator
f=PhysicsSlipAccumulator(4)
# A mid-period spike must affect reward even when endpoint speed is zero.
for i,v in enumerate([0.,8.,0.,0.]):f.add(i,torch.tensor([v,2*v]))
assert torch.equal(f.consume(),torch.tensor([2.,4.]))
# Reset-side duplicate refresh must not contaminate the next control period.
f.add(3,torch.tensor([999.,999.]))
for i in range(4,8):f.add(i,torch.tensor([1.,0.]))
assert torch.equal(f.consume(),torch.tensor([1.,0.]))
for bad in ['incomplete','skip']:
 try:
  if bad=='incomplete':f.consume()
  else:f.add(10,torch.zeros(2))
 except RuntimeError:pass
 else:raise AssertionError(bad)
assert f.completed_windows==2
print('Physics reward spike, duplicate reset refresh, independent environments and missing-window guards passed')
