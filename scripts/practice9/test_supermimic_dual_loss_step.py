"""Check mutation/rollback contracts of the experimental dual-loss optimizer."""
import copy,json,sys
from pathlib import Path
import torch
from supermimic_dual_loss_step import dual_loss_step

def equal(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def case(kind):
 p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64));opt=torch.optim.Adam([p],lr=.1)
 original=p.detach().clone();reference=torch.nn.Parameter(original.clone());ref=torch.optim.Adam([reference],lr=.1)
 reference.grad=torch.ones_like(reference);torch.nn.utils.clip_grad_norm_([reference],1.);ref.step()
 count=0
 def evaluate():
  nonlocal count
  count+=1;x=float(p)
  if kind=='identity':return (0.,0.) if torch.equal(p,original) else (1.,1.)
  if kind=='rollback':return (1.,1.)
  if kind=='half':return (2.,2.) if x<.94 else (x*x,x*x)
  return x*x,x*x
 baseline=(0.,0.) if kind in ['identity','rollback'] else (1.,1.)
 before_opt=copy.deepcopy(opt.state_dict())
 if kind=='rollback':
  try:dual_loss_step([p],opt,[torch.ones_like(p)],[torch.ones_like(p)],evaluate,baseline,1.,True)
  except AssertionError:pass
  else:raise AssertionError('Identity failure must reject')
  assert torch.equal(p,original) and equal(opt.state_dict(),before_opt)
 else:
  out=dual_loss_step([p],opt,[torch.ones_like(p)],[torch.ones_like(p)],evaluate,baseline,1.,kind!='disabled')
  assert count==13 and equal(opt.state_dict(),ref.state_dict())
  if kind=='identity':assert torch.equal(p,original) and out['selected_factor']==0
  elif kind=='half':assert torch.equal(p,original+.5*(reference.detach()-original)) and out['selected_factor']==.5
  else:assert torch.equal(p,reference) and out['selected_factor']==1
 return dict(case=kind,passed=True)
rows=[case(k) for k in ['full','half','identity','disabled','rollback']]
out=Path(sys.argv[1]);assert not out.exists();out.write_text(json.dumps(dict(passed=True,cases=rows),indent=2));print(json.dumps(rows))
