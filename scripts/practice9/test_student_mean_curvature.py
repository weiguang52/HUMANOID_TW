from pathlib import Path
import sys,runpy,copy,importlib.util,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
from second_order_anchor import student_mean_curvature_mse,second_order_anchor_mse,PREVIOUS,OLDER,VALID
from anchored_ppo import install_anchor
from temporal_target_contract import validate_target,validate_resume_target
torch.set_num_threads(1)
x=torch.tensor([[1.,-1.],[0.,0.],[-1.,1.]],requires_grad=True);z=torch.zeros_like(x);mask=torch.tensor([[1.],[0.],[1.]])
loss=student_mean_curvature_mse(x,z,x,mask);assert loss==4
loss.backward();assert not x.grad[1].any() and x.grad[[0,2]].abs().sum()>0
x=torch.randn(3,2,requires_grad=True);loss=student_mean_curvature_mse(x,z,z,torch.zeros(3,1));loss.backward();assert loss==0 and not x.grad.any()
a=torch.ones(3,2);assert second_order_anchor_mse(a,a,-a,-a,a,a,torch.ones(3,1))==0
assert student_mean_curvature_mse(a,-a,a,torch.ones(3,1))==16
for order,coef in [(1,1.),(2,0.)]:
 try:validate_target(coef,order,'student_mean')
 except ValueError:pass
 else:raise AssertionError('Unsupported target accepted')
try:validate_resume_target({'temporal_coefficient':1.,'temporal_order':2},'student_mean')
except ValueError:pass
else:raise AssertionError('Legacy teacher-relative checkpoint silently reinterpreted')
validate_resume_target({'temporal_coefficient':1.,'temporal_target':'student_mean'},'student_mean');validate_resume_target({},'student_mean')
ns=runpy.run_path(str(Path(__file__).with_name('test_anchored_ppo.py')))
oldpath=Path('/root/gpufree-data/datasets/practice9/absolute_mean_curvature_20261026/source_before/anchored_ppo.py')
spec=importlib.util.spec_from_file_location('old_curvature_ppo',oldpath);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
alg,obs,batch=ns['make']();obs[PREVIOUS]=obs['policy']*.4-.2;obs[OLDER]=obs['policy']*-.3+.1;obs[VALID]=torch.ones(32,1);other=copy.deepcopy(alg)
old.install_anchor(alg,1.,temporal_coefficient=1.,temporal_order=2);install_anchor(other,1.,temporal_coefficient=1.,temporal_order=2)
rng=torch.random.get_rng_state();a=alg.update();after_rng=torch.random.get_rng_state();torch.random.set_rng_state(rng);b=other.update()
assert a==b and torch.equal(torch.random.get_rng_state(),after_rng)
for x,y in zip(alg.actor.parameters(),other.actor.parameters()):assert torch.equal(x,y)
for x,y in zip(alg.critic.parameters(),other.critic.parameters()):assert torch.equal(x,y)
for key,value in alg.optimizer.state_dict()['state'].items():
 for field,v in value.items():
  w=other.optimizer.state_dict()['state'][key][field];assert torch.equal(v,w) if torch.is_tensor(v) else v==w
alg,obs,batch=ns['make']();obs[PREVIOUS]=obs['policy']*.3-.1;obs[OLDER]=obs['policy']*-.2+.4;obs[VALID]=torch.ones(32,1);obs[VALID][:2]=0
meta=install_anchor(alg,0.,temporal_coefficient=1.,temporal_order=2,temporal_target='student_mean');assert meta['temporal_target']=='student_mean'
p=obs.clone(False);p['policy']=obs[PREVIOUS];q=obs.clone(False);q['policy']=obs[OLDER]
def objective():return student_mean_curvature_mse(alg.actor(obs),alg.actor(p),alg.actor(q),obs[VALID])
with torch.no_grad():before=objective().item()
frozen={k:v.clone() for k,v in alg.anchor_teacher.state_dict().items()};batch.advantages.zero_();alg.value_loss_coef=0.;alg.entropy_coef=0.
rng=torch.random.get_rng_state();result=alg.update()
with torch.no_grad():after=objective().item()
assert before>0 and abs(result['anchor_temporal_mse']-before)<1e-7 and after<before
assert all(torch.equal(v,alg.anchor_teacher.state_dict()[k]) for k,v in frozen.items()) and all(p.grad is None for p in alg.anchor_teacher.parameters())
print('PASS absolute loss/reset gradients, non-cancellation of shared teacher oscillation, target resume contract, real PPO legacy weights/Adam/RNG bitwise parity, direct mean loss descent and frozen teacher',before,after)
