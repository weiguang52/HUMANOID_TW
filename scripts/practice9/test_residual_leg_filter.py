"""Current teacher feedthrough, residual recursion, reset and actual blend contract."""
import sys,tempfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
import torch,numpy as np
from candidate_leg_filter import TeacherResidualLegFilter
from reference_speed_blend import ReferenceSpeedBlend
names=[s+'_'+j for s in ['left','right'] for j in ['hip_pitch_joint','hip_roll_joint','hip_yaw_joint','knee_pitch_joint','ankle_yaw_joint','ankle_pitch_joint','foot_roll']]+['other'+str(i) for i in range(16)]
f=TeacherResidualLegFilter(names,'cpu',.8)
t0=torch.zeros(2,30);a0=torch.ones(2,30)*2
assert torch.equal(f(a0,t0),a0)
t1=torch.ones(2,30)*100;a1=t1+4
expected=torch.where(f.mask[None],t1+.8*4+.2*2,a1)
assert torch.allclose(f(a1,t1),expected,atol=1e-6)
f.reset(torch.tensor([True,False]));t2=torch.ones(2,30)*-200;a2=t2+8
result=f(a2,t2);assert torch.equal(result[0],a2[0])
assert torch.allclose(result[1,f.mask],(t2+.8*8+.2*3.6)[1,f.mask],atol=2e-5)
assert torch.equal(result[:,~f.mask],a2[:,~f.mask])
# With zero residual history, rapid teacher changes pass through without lag.
z=TeacherResidualLegFilter(names,'cpu',.8)
for value in [0.,100.,-100.,300.]:
 t=torch.full((2,30),value);assert torch.equal(z(t,t),t)
identity=TeacherResidualLegFilter(names,'cpu',1.)
assert torch.equal(identity(a0,t0),a0) and torch.equal(identity(a1,t1),a1)
class Teacher(torch.nn.Module):
 def forward(self,x):return x[:,0:1].repeat(1,30)
with tempfile.TemporaryDirectory() as folder:
 p=Path(folder)/'teacher.pt';torch.jit.trace(Teacher(),torch.zeros(2,2)).save(str(p))
 cmd=SimpleNamespace(cfg=SimpleNamespace(body_names=['base_link']),body_lin_vel_w=torch.zeros(2,1,3))
 obs={'policy':torch.zeros(2,10),'contact_base_policy':torch.ones(2,2)*10}
 b=ReferenceSpeedBlend(str(p),'cpu',cmd,1.,.05,.1,action_names=names,leg_alpha=.8,leg_mode='teacher_residual')
 raw=torch.ones(2,30)*12;assert torch.equal(b(obs,raw),torch.ones(2,30)*10)
 cmd.body_lin_vel_w[:,:,0]=.1;obs['contact_base_policy'].fill_(100);raw=torch.ones(2,30)*104
 filtered=torch.where(b.leg_filter.mask[None],torch.ones(2,30)*103.6,raw)
 assert torch.allclose(b(obs,raw),filtered,atol=1e-5)
 b.reset(torch.tensor([True,False]));cmd.body_lin_vel_w[:,0,0]=torch.tensor([.075,0.]);obs['contact_base_policy'].fill_(-100);raw=torch.ones(2,30)*-92
 result=b(obs,raw);assert torch.allclose(result[0],torch.ones(30)*-96,atol=1e-5) and torch.equal(result[1],torch.ones(30)*-100)
 b.save_trace(Path(folder)/'trace.npz');trace=np.load(Path(folder)/'trace.npz')
 assert trace['candidate_leg_reference'].shape==(3,2,14) and trace['candidate_leg_raw_filtered'].shape==(3,2,14,2)
 assert b.audit['candidate_leg_filter']['mode']=='teacher_residual' and b.audit['observation_key']=='contact_base_policy'
 for kwargs in [{'leg_mode':'unknown'},{'leg_mode':'teacher_residual'}]:
  try:ReferenceSpeedBlend(str(p),'cpu',cmd,**kwargs)
  except ValueError:pass
  else:raise AssertionError('Invalid filter contract accepted')
print('Residual recurrence, current-teacher feedthrough, zero-residual history, mixed resets, nonlegs, alpha1, real JIT teacher, speed gate and trace contract passed')
