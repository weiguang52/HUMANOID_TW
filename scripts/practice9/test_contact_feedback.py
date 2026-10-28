"""Synthetic physics contracts for the inactive contact feedback prototype."""
import sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
import torch
from contact_feedback import contact_statistics,contact_features
D=torch.float64
vec=lambda x:torch.tensor(x,dtype=D)
# Two environments own disjoint contact ranges. Opposing rotational velocities
# have zero weighted centroid velocity but nonzero RMS and must not disappear.
com=torch.zeros(2,3,dtype=D);linear=vec([[0,0,0],[.2,-.1,0]]);angular=vec([[0,0,1],[0,0,0]])
forces=vec([10,10,30]);points=vec([[1,0,0],[-1,0,0],[100,100,0]])
z=contact_statistics(forces,points,torch.tensor([2,1]),torch.tensor([0,2]),com,linear,angular)
assert torch.equal(z['load'],vec([20,30]))
assert torch.allclose(z['velocity'],linear,atol=1e-12)
assert torch.allclose(z['xy_rms'],vec([1,math.sqrt(.05)]),atol=1e-12)
# Force weighting differs from the deliberately unweighted point RMS.
w=contact_statistics(vec([10,30]),points[:2],torch.tensor([2]),torch.tensor([0]),com[:1],linear[:1],angular[:1])
assert torch.allclose(w['velocity'],vec([[0,-.5,0]])) and torch.equal(w['xy_rms'],vec([1]))
# Empty buffers and individual zero-contact environments stay finite and zero.
e=contact_statistics(vec([]),torch.empty(0,3,dtype=D),torch.tensor([0,0]),torch.tensor([0,0]),com,linear,angular)
assert all(torch.equal(v,torch.zeros_like(v)) for v in e.values())
e=contact_statistics(forces,points,torch.tensor([0,1]),torch.tensor([3,2]),com,linear,angular)
assert e['load'][0]==0 and e['xy_rms'][0]==0 and torch.equal(e['velocity'][0],vec([0,0,0]))
# A 90-degree base yaw maps world +X into heading -Y; load order is L then R.
load=vec([[20,0]]);velocity=vec([[[.1,0,0],[99,99,0]]]);rms=vec([[.1,99]])
quat=vec([[math.sqrt(.5),0,0,math.sqrt(.5)]])
f=contact_features(load,velocity,rms,quat)
assert torch.allclose(f,vec([[.2,0,-1,1,0,0,0,0]]),atol=1e-12)
assert torch.allclose(contact_features(load,velocity,rms,-quat),f,atol=1e-12)
assert contact_features(vec([[10000,10000]]),velocity,rms,quat).abs().max()<=5
for args in [(forces,points,torch.tensor([65,1]),torch.tensor([0,2]),com,linear,angular),(forces,points,torch.tensor([2,2]),torch.tensor([0,2]),com,linear,angular)]:
 try:contact_statistics(*args)
 except ValueError:pass
 else:raise AssertionError('Invalid contact range accepted')
try:contact_features(load,velocity,rms,torch.zeros_like(quat))
except ValueError:pass
else:raise AssertionError('Zero quaternion accepted')
print('Contact ownership, angular point velocity, RMS, weighting, empty reset, heading invariance, bounds and invalid inputs passed')
# Preserve the existing reward's numerical RMS contract over randomized buffers.
import importlib.util
path=Path(__file__).resolve().parents[2]/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/point_slip.py'
spec=importlib.util.spec_from_file_location('point_slip_reference',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
g=torch.Generator().manual_seed(42);n=9;counts=torch.randint(0,8,(n,),generator=g);starts=torch.cat([torch.zeros(1,dtype=torch.long),counts.cumsum(0)[:-1]]);p=int(counts.sum())
forces=torch.rand(p,1,generator=g,dtype=D)*10;points=torch.randn(p,3,generator=g,dtype=D);com=torch.randn(n,3,generator=g,dtype=D);linear=torch.randn(n,3,generator=g,dtype=D);angular=torch.randn(n,3,generator=g,dtype=D)
st=contact_statistics(forces,points,counts,starts,com,linear,angular)
expected=module.point_speed_cost(forces,points,counts,starts,com,linear,angular)
assert torch.equal(torch.where(st['load']>1,st['xy_rms'],0.),expected)
# Joint world rotation and base-yaw rotation leave heading features invariant.
angle=.73;c=math.cos(angle);s=math.sin(angle);v0=vec([[[.04,.03,0],[-.02,.01,0]]]);v1=v0.clone();v1[:,:,0]=c*v0[:,:,0]-s*v0[:,:,1];v1[:,:,1]=s*v0[:,:,0]+c*v0[:,:,1]
q0=vec([[1,0,0,0]]);q1=vec([[math.cos(angle/2),0,0,math.sin(angle/2)]]);loads=vec([[20,30]]);rr=vec([[.05,.03]])
assert torch.allclose(contact_features(loads,v0,rr,q0),contact_features(loads,v1,rr,q1),atol=1e-12)
# Sensor adapter discards a reset environment's stale physics buffer.
from types import SimpleNamespace
from contact_feedback import ground_contact_features
class View:
 def get_contact_data(self,dt):
  assert dt==.005
  return vec([20,20]),vec([[0,0,0],[0,0,0]]),None,None,torch.tensor([1,1]),torch.tensor([0,1])
body=SimpleNamespace(body_com_pos_w=torch.zeros(2,2,3,dtype=D),body_lin_vel_w=torch.ones(2,2,3,dtype=D)*.1,body_ang_vel_w=torch.zeros(2,2,3,dtype=D),root_quat_w=vec([[1,0,0,0],[1,0,0,0]]))
robot=SimpleNamespace(data=body,body_names=['left_foot','right_foot']);sensor=SimpleNamespace(contact_physx_view=View())
env=SimpleNamespace(scene={'robot':robot,'slip_left_foot':sensor,'slip_right_foot':sensor},cfg=SimpleNamespace(sim=SimpleNamespace(dt=.005)),episode_length_buf=torch.tensor([0,1]))
out=ground_contact_features(env);assert out.shape==(2,8) and torch.equal(out[0],torch.zeros(8,dtype=D)) and out[1,0]>.0
print('Existing reward RMS bitwise parity, global yaw invariance and per-environment stale-reset masking passed')
env.scene['diagnostic_left_foot']=sensor;env.scene['diagnostic_right_foot']=sensor
assert torch.equal(ground_contact_features(env,sensor_prefix='diagnostic_'),out)
print('Explicit training/evaluation sensor names passed')
