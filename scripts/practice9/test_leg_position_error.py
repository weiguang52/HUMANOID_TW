"""Reference gating, canonical joint lookup and gradient checks without simulator startup."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import torch
p=Path(__file__).resolve().parents[2]/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/leg_position_error.py'
spec=importlib.util.spec_from_file_location('position_cost_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=torch.ones(5,12,requires_grad=True);r=torch.zeros_like(a);speed=torch.tensor([0.,.05,.075,.1,.2]);before=a.detach().clone();rng=torch.random.get_rng_state().clone()
y=m.leg_position_cost(a,r,speed);torch.testing.assert_close(y,torch.tensor([0.,0.,2.,4.,4.]),rtol=1e-6,atol=1e-6);y.sum().backward();assert (a.grad[:2]==0).all() and (a.grad[2:]>0).all();assert torch.equal(a.detach(),before) and torch.equal(rng,torch.random.get_rng_state());assert not m.leg_position_cost(r,r,speed).any()
for scale in (0.,-1.,float('nan'),float('inf')):
 try:m.leg_position_cost(a,r,speed,scale)
 except ValueError:pass
 else:raise AssertionError('Invalid scale accepted')
canonical=[s+'_'+j+'_joint' for s in ('left','right') for j in ('hip_pitch','hip_roll','hip_yaw','knee_pitch','ankle_yaw','ankle_pitch')];names=['waist_yaw']+list(reversed(canonical))+['left_foot_roll'];actual=torch.zeros(2,len(names));actual[:,0]=100;actual[:,-1]=100
for i,n in enumerate(canonical):actual[:,names.index(n)]=(i+1)*.01
command=NS(robot=NS(joint_names=names),joint_indexes=torch.arange(len(names)),robot_joint_pos=actual,joint_pos=torch.zeros_like(actual),cfg=NS(body_names=['base_link']),body_lin_vel_w=torch.tensor([[[.1,0,0]],[[0,0,0]]]))
env=NS(command_manager=NS(get_term=lambda name:command));out=m.moving_leg_position_error(env);expected=(torch.arange(1,13)*.01/.5).square().mean();torch.testing.assert_close(out,torch.stack([expected,torch.tensor(0.)]));assert [names[i] for i in command._position_cost_leg_ids]==canonical
print('PASS: gate thresholds, scale, gradients, zero target, input/RNG preservation, shuffled lookup, excluded joints')
