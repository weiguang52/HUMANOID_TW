from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"rsl_rl"))
from contact_history import ContactHistory
from physics_slip_reward import validate_history_resume,validate_history_samples
def test():
 for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
  h=ContactHistory();load=lambda v:torch.tensor(v,device=device,dtype=torch.float64)
  x=load([[6.,6.],[0.,0.]]);v=torch.ones_like(x)*.2;p=torch.ones_like(x,dtype=torch.long)
  assert torch.equal(h.update(x,v,p),load([[.2,.2],[0.,0.]]))
  h.reset(torch.tensor([1],device=device));x=load([[2.,0.],[2.,2.]])
  result=h.update(x,v,torch.tensor([[1,0],[1,1]],device=device));assert torch.equal(result,load([[.2,0.],[0.,0.]]))
  assert h.update(x,v,p)[0,0]==.2
  assert h.update(x,v,p)[0,0]==0
  h.update(load([[6.,6.],[6.,6.]]),v,p);h.reset();assert torch.equal(h.update(x,v,p),torch.zeros_like(v))
  h=ContactHistory(samples=1);assert torch.equal(h.update(load([[5.,5.0001],[0.,6.]]),v,p),load([[0.,.2],[0.,.2]]))
  h=ContactHistory();h.update(load([[6.,6.],[6.,6.]]),v,p);assert torch.equal(h.update(torch.zeros_like(x),torch.full_like(v,float('nan')),torch.zeros_like(p)),torch.zeros_like(v))
  print(device,'strict threshold,2-step hold expiry,per-owner reset,no-contact NaN exclusion,current velocity only PASS')
if __name__=='__main__':test()

for saved,request in [({},3),({"contact_history_samples":3},1)]:
 try:validate_history_resume(saved,request)
 except ValueError:pass
 else:raise AssertionError("silent history change")
 assert validate_history_resume(saved,request,True)["changed"]
for v in [0,-1,True,1.5]:
 try:validate_history_samples(v)
 except ValueError:pass
 else:raise AssertionError("invalid history")
assert not validate_history_resume({},1)["changed"]
print("history resume guards PASS")
