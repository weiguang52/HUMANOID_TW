import importlib.util
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[2]/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/point_slip.py'
s=importlib.util.spec_from_file_location('point_slip',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_point_geometry_and_environment_isolation():
    # Noncontiguous ownership: env 0 reads point 2, env 1 reads point 0.
    points=torch.tensor([[0.,0.,0.],[float('nan')]*3,[0.,0.,-1.]])
    forces=torch.tensor([10.,0.,10.])
    com=torch.zeros(3,3);linear=torch.tensor([[1.,0.,0.],[2.,0.,0.],[9.,9.,9.]])
    angular=torch.tensor([[0.,1.,0.],[0.,0.,0.],[0.,0.,0.]])
    out=m.point_speed_cost(forces,points,torch.tensor([1,1,0]),torch.tensor([2,0,0]),com,linear,angular)
    torch.testing.assert_close(out,torch.tensor([0.,2.,0.]))

def test_multiple_points_force_gate_and_capacity():
    points=torch.tensor([[0.,0.,-1.],[0.,0.,1.]])
    com=torch.zeros(1,3);v=torch.tensor([[1.,0.,0.]]);w=torch.tensor([[0.,1.,0.]])
    args=(points,torch.tensor([2]),torch.tensor([0]),com,v,w)
    torch.testing.assert_close(m.point_speed_cost(torch.tensor([2.,2.]),*args),torch.tensor([2.**.5]))
    torch.testing.assert_close(m.point_speed_cost(torch.tensor([.2,.2]),*args),torch.zeros(1))
    try:m.point_speed_cost(torch.tensor([2.,2.]),*args,capacity=1)
    except RuntimeError:pass
    else:raise AssertionError('Silent truncation')

if __name__=='__main__':
    test_point_geometry_and_environment_isolation();test_multiple_points_force_gate_and_capacity();print('point slip tests passed')
