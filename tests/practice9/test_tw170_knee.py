import importlib,sys,types,pathlib
import pytest
import torch
p=types.ModuleType('tw170_mdp');p.__path__=[str(pathlib.Path(__file__).resolve().parents[2]/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp')]
sys.modules[p.__name__]=p
m=importlib.import_module('tw170_mdp.swing_knee')

def test_unknown_stance_zero_and_no_gradient():
 e=torch.tensor([[1.,2.],[0.,3.]],requires_grad=True)
 c=torch.tensor([[False,True],[False,False]])
 k=torch.tensor([[False,True],[True,False]])
 out=m.masked_knee_cost(e,c,k,.35)
 assert out.tolist()==[0.,0.]
 out.sum().backward();assert e.grad.abs().sum()==0

def test_large_error_has_gradient_and_left_right_are_independent():
 e=torch.tensor([[1.,2.]],requires_grad=True)
 out=m.masked_knee_cost(e,torch.tensor([[False,True]]),torch.ones(1,2,dtype=torch.bool),.35)
 out.backward();assert e.grad[0,0]>1 and e.grad[0,1]==0
 assert out>0

def test_reference_mapping_and_phase_not_rewritten():
 labels=torch.tensor([[False,True],[True,False]])
 known=torch.ones_like(labels)
 c=types.SimpleNamespace(cfg=types.SimpleNamespace(joint_names=['right_knee_pitch_joint','left_knee_pitch_joint']),
 joint_pos=torch.tensor([[8.,1.]]),robot_joint_pos=torch.zeros(1,2),frame_indices=torch.tensor([0]),_contact_reference=(labels.clone(),known.clone()))
 env=types.SimpleNamespace(command_manager=types.SimpleNamespace(get_term=lambda _:c))
 expected=m.masked_knee_cost(torch.tensor([[1.,8.]]),labels[:1],known[:1],.35)
 assert torch.equal(m.swing_knee_cost(env,'motion'),expected)
 assert torch.equal(c._contact_reference[0],labels)

@pytest.mark.parametrize('scale',[0,-1,float('nan')])
def test_bad_scale(scale):
 with pytest.raises(ValueError):m.masked_knee_cost(torch.zeros(1,2),torch.zeros(1,2,dtype=torch.bool),torch.ones(1,2,dtype=torch.bool),scale)
