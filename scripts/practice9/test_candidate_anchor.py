"""Selected17 MSE scaling and default/resume mapping without a new PPO update."""
from pathlib import Path
import sys,copy,math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
import torch
from anchored_ppo import candidate_anchor_weights,validate_candidate_resume,anchor_mse
from candidate_action_scope import EXPECTED_NAMES,make_scope_contract
names=list(EXPECTED_NAMES);scope=make_scope_contract(names,'lower_body_waist')
selected=[n in scope['candidate_joint_names'] for n in names]
weights=candidate_anchor_weights(names,8);assert weights==[8. if v else 1. for v in selected]
assert candidate_anchor_weights(names,1)==[1.]*30 and sum(selected)==17
saved={'candidate_multiplier':8.,'candidate_scope':'lower_body_waist','action_joint_names':names,'action_weights':weights}
validate_candidate_resume(saved,8.);validate_candidate_resume({},1.)
validate_candidate_resume({},8.,allow_frozen_initialization=True)
def rejects(call):
 try:call()
 except ValueError:return
 raise AssertionError('Invalid or mismatched anchor accepted')
rejects(lambda:validate_candidate_resume({},8.))
rejects(lambda:validate_candidate_resume(saved,1.))
rejects(lambda:validate_candidate_resume(saved,4.))
rejects(lambda:validate_candidate_resume(dict(saved,action_weights=[1.]*30),8.))
rejects(lambda:validate_candidate_resume(dict(saved,candidate_scope='all'),8.))
for value in [-1,0,.5,float('nan'),float('inf'),True,'8']:
 rejects(lambda:candidate_anchor_weights(names,value))
rejects(lambda:candidate_anchor_weights(names[:-1],8))
rejects(lambda:candidate_anchor_weights(names,8,existing=[0.]*30))
for device in ['cpu','cuda']:
 if device=='cuda' and not torch.cuda.is_available():continue
 torch.manual_seed(8044)
 mean=torch.randn(9,30,device=device,requires_grad=True);teacher=torch.randn_like(mean)
 rng=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
 baseline=anchor_mse(mean,teacher);base_grad=torch.autograd.grad(baseline,mean,retain_graph=True)[0]
 one=anchor_mse(mean,teacher,torch.tensor(candidate_anchor_weights(names,1),device=device));one_grad=torch.autograd.grad(one,mean,retain_graph=True)[0]
 assert torch.equal(one,baseline) and torch.equal(one_grad,base_grad)
 w=torch.tensor(weights,device=device);loss=anchor_mse(mean,teacher,w);grad=torch.autograd.grad(loss,mean)[0]
 torch.testing.assert_close(grad[:,selected],base_grad[:,selected]*8,rtol=0,atol=0)
 assert torch.equal(grad[:,[not v for v in selected]],base_grad[:,[not v for v in selected]])
 want=((mean-teacher).square()[:,selected].sum()*8+(mean-teacher).square()[:,[not v for v in selected]].sum())/(9*30)
 torch.testing.assert_close(loss,want,atol=2e-6,rtol=1e-6)
 permutation=list(reversed(range(30)));assert candidate_anchor_weights([names[i] for i in permutation],8)==[weights[i] for i in permutation]
 assert torch.equal(rng,torch.get_rng_state())
 if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
 print(device,'default MSE/gradient exact; selected17 x8 and remaining13 unchanged; original denominator30, ordering and RNG passed')
print('Saved weighted mapping and silent resume changes rejected; simulator integration is checked separately')
