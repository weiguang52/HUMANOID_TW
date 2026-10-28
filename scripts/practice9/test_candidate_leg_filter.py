"""Candidate-only14-leg filter: scope, per-environment reset and gate integrity."""
import sys,tempfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
import torch,numpy as np
from candidate_leg_filter import CandidateLegFilter
from reference_speed_blend import ReferenceSpeedBlend
legs=[s+'_'+j for s in ['left','right'] for j in ['hip_pitch_joint','hip_roll_joint','hip_yaw_joint','knee_pitch_joint','ankle_yaw_joint','ankle_pitch_joint','foot_roll']]
names=list(reversed(legs+['other_'+str(i) for i in range(16)]));f=CandidateLegFilter(names,'cpu',.8)
a=torch.arange(60).reshape(2,30).float();snap=a.clone();assert torch.equal(f(a),a);a.add_(1000);b=torch.ones(2,30)*5;expected=torch.where(f.mask[None],.8*b+.2*snap,b);assert torch.equal(f(b),expected)
f.reset(torch.tensor([True,False]));c=torch.ones(2,30)*9;want=torch.where(f.mask[None],.8*c+.2*expected,c);want[0]=c[0];assert torch.equal(f(c),want)
identity=CandidateLegFilter(names,'cpu',1.);assert torch.equal(identity(a),a) and torch.equal(identity(b),b)
for alpha in [-1,0,1.1,float('nan')]:
 try:CandidateLegFilter(names,'cpu',alpha)
 except ValueError:pass
 else:raise AssertionError('Invalid alpha accepted')
try:CandidateLegFilter(names[:-1],'cpu',.8)
except ValueError:pass
else:raise AssertionError('Missing leg action accepted')
class Teacher(torch.nn.Module):
 def __init__(self):super().__init__();self.register_buffer('weight',torch.arange(60).reshape(2,30).float()/100)
 def forward(self,x):return x@self.weight
with tempfile.TemporaryDirectory() as folder:
 p=Path(folder)/'teacher.pt';torch.jit.trace(Teacher(),torch.zeros(2,2)).save(str(p));command=SimpleNamespace(cfg=SimpleNamespace(body_names=['base_link']),body_lin_vel_w=torch.zeros(2,1,3));obs={'policy':torch.zeros(2,10),'contact_base_policy':torch.tensor([[1.,2.],[3.,4.]])};base=ReferenceSpeedBlend(str(p),'cpu',command,1.,.05,.1);blend=ReferenceSpeedBlend(str(p),'cpu',command,1.,.05,.1,action_names=names,leg_alpha=.8);teacher=Teacher()(obs['contact_base_policy']);raw=torch.ones(2,30)*2
 assert torch.equal(blend(obs,raw),teacher) and torch.equal(base(obs,raw),teacher)
 command.body_lin_vel_w[:,:,0]=.1;raw=torch.ones(2,30)*4;filtered=torch.where(f.mask[None],.8*raw+.2*torch.ones_like(raw)*2,raw)
 assert torch.equal(blend(obs,raw),filtered) and torch.equal(base(obs,raw),raw)
 command.body_lin_vel_w[:,0,0]=torch.tensor([.075,0.]);raw=torch.ones(2,30)*-1;filtered2=torch.where(f.mask[None],.8*raw+.2*filtered,raw);result=blend(obs,raw);weight=torch.tensor([[.5],[0.]]);expected=teacher+(filtered2-teacher)*weight
 assert torch.allclose(result,expected,atol=1e-6,rtol=0) and torch.equal(result[1],teacher[1])
 assert 'candidate_leg_filter' not in base.audit and blend.audit['candidate_leg_filter']['scope']=='legs14'
 blend.save_trace(Path(folder)/'trace.npz');z=np.load(Path(folder)/'trace.npz');assert z['candidate_leg_raw_filtered'].shape==(3,2,14,2) and len(z['candidate_leg_joints'])==14
 base.save_trace(Path(folder)/'legacy.npz');assert np.load(Path(folder)/'legacy.npz').files==['speed_and_candidate_weight']
 try:ReferenceSpeedBlend(str(p),'cpu',command,hip_alpha=.5,action_names=names,leg_alpha=.8)
 except ValueError:pass
 else:raise AssertionError('Two candidate filters accepted together')
print('Leg14 scope, identity, immutable state, mixed resets, alpha validation, frozen teacher zero gate, intermediate gate, unchanged default and auditable traces passed')
