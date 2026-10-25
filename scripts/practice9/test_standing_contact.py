"""Pure reward gate checks without launching Isaac Sim."""
import importlib.util
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[2]/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/standing_contact.py'
spec=importlib.util.spec_from_file_location('standing_contact',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
f=torch.zeros(1,4,2);xy=torch.zeros(1,2);yaw=torch.zeros(1);yes=torch.ones(1,dtype=torch.bool);both=torch.ones(1,2,dtype=torch.bool)
def cost(force=f,speed=xy,rate=yaw,standing=yes,contact=both,known=both):return m.standing_contact_cost(force,speed,rate,standing,contact,known)
assert cost().item()==1
assert cost(force=torch.ones_like(f)).item()==0
assert cost(speed=torch.tensor([[.1,0.]])).item()==0
assert cost(rate=torch.ones(1)).item()==0
assert cost(standing=~yes).item()==0
assert cost(contact=torch.tensor([[True,False]])).item()==0
assert cost(known=torch.tensor([[False,True]])).item()==0
partial=f.clone();partial[:,0,:]=1
assert cost(force=partial).item()==.75
assert 0<cost(speed=torch.tensor([[.015,0.]])).item()<1
print('9 standing-contact checks passed')
