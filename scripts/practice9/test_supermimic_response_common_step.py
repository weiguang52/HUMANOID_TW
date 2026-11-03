"""State and replay contracts for the experimental five-loss step."""
import copy
import torch
from supermimic_response_common_step import response_common_step

def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

for mode in ['accepted','disabled','identity','rollback']:
    p=torch.nn.Parameter(torch.tensor([0.25,-0.5]))
    opt=torch.optim.Adam([p],lr=.01)
    p.grad=torch.tensor([.3,-.2]);opt.step();opt.zero_grad()
    start=p.detach().clone();old=copy.deepcopy(opt.state_dict())
    losses=[((p-1)**2).sum()*(i+1) for i in range(5)]
    gradients=[torch.autograd.grad(loss,[p],retain_graph=True) for loss in losses]
    baseline=tuple(float(loss) for loss in losses)
    reference=torch.nn.Parameter(start.clone());ref_opt=torch.optim.Adam([reference],lr=.01);ref_opt.load_state_dict(copy.deepcopy(old))
    reference.grad=gradients[1][0].clone();torch.nn.utils.clip_grad_norm_([reference],1.);ref_opt.step()
    calls=[]
    def evaluate():
        calls.append(p.detach().clone())
        if mode=='rollback':raise RuntimeError('injected failure')
        if mode=='identity' and not torch.equal(p,start):return tuple(v+1 for v in baseline)
        return tuple(float(((p-1)**2).sum()*(i+1)) for i in range(5))
    try:
        proof=response_common_step([p],opt,gradients,evaluate,baseline,1.,mode!='disabled')
    except RuntimeError as error:
        assert mode=='rollback' and str(error)=='injected failure'
        assert torch.equal(p,start) and equal(opt.state_dict(),old)
        continue
    assert len(calls)==9 and equal(opt.state_dict(),ref_opt.state_dict())
    assert torch.equal(calls[-1],start)
    if mode=='disabled':assert torch.equal(p,reference)
    elif mode=='identity':assert torch.equal(p,start) and proof['selected_factor']==0
    else:assert proof['selected_factor']>0 and all(v<=b for v,b in zip(proof['applied_values'],baseline))
print('4 contracts passed: accepted, disabled Adam/state exact, identity, exception rollback')
