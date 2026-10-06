from test_tw176_sole import s
import torch

def test_mask_keeps_swing_unknown_and_inputs():
 ref=torch.ones(2,2)*.01;act=torch.zeros(2,2,requires_grad=True)
 contact=torch.tensor([[True,False],[True,False]]);known=torch.tensor([[True,True],[False,False]])
 old=contact.clone();full=s.sole_cost(ref,act,.01)
 cost=s.stance_excluded_cost(ref,act,contact,known,.01)
 assert torch.allclose(cost,full*torch.tensor([.5,1.]))
 cost.sum().backward();assert act.grad[0,0]==0 and act.grad[0,1]!=0
 assert torch.equal(contact,old)

def test_both_stance_zero_without_nan():
 z=torch.zeros(3,2);ones=torch.ones(3,2,dtype=torch.bool)
 assert torch.equal(s.stance_excluded_cost(z+1,z,ones,ones,.01),torch.zeros(3))
