"""Loaded-foot threshold contract, legacy parity and real accumulator adapter."""
from pathlib import Path
import importlib.util,sys,types,json
import torch
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'scripts/rsl_rl'))
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
new=load('new_point_slip',root/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/point_slip.py')
old=load('old_point_slip',Path('/root/twrl_runs/load_aligned_20261027/source_before/point_slip.py'))
from physics_slip_reward import install,physics_mean_moving_slip,validate_load_resume,validate_minimum_load
from types import SimpleNamespace as NS
for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
 torch.manual_seed(711);n=512;d=torch.float64
 counts=torch.randint(0,8,(n,),device=device);starts=torch.cat([torch.zeros(1,device=device,dtype=torch.long),counts.cumsum(0)[:-1]]);p=int(counts.sum())
 force=torch.rand(p,1,device=device,dtype=d)*10;points=torch.randn(p,3,device=device,dtype=d);com=torch.randn(n,3,device=device,dtype=d);lin=torch.randn_like(com);ang=torch.randn_like(com)
 args=(force,points,counts,starts,com,lin,ang)
 assert torch.equal(old.point_speed_cost(*args),new.point_speed_cost(*args))
 assert torch.equal(new.point_speed_cost(*args),new.point_speed_cost(*args,minimum_load=1.))
 loads=torch.tensor([0.,1.,1.00001,5.,5.00001,30.],device=device,dtype=d);n=len(loads);pos=torch.zeros(n,3,device=device,dtype=d);vel=pos.clone();vel[:,0]=.2;rot=pos.clone();rot[:,2]=1;pts=pos.clone();pts[:,0]=1
 args=(loads,pts,torch.ones(n,device=device,dtype=torch.long),torch.arange(n,device=device),pos,vel,rot)
 speed=(.2**2+1)**.5
 for minimum in (1.,5.):
  expected=torch.where(loads>minimum,torch.full_like(loads,speed),0.)
  assert torch.allclose(new.point_speed_cost(*args,minimum_load=minimum),expected,atol=1e-12)
 # No-contact owners in a shared allocated PhysX buffer remain finite zero.
 zeros=torch.zeros(n,device=device,dtype=torch.long);z=new.point_speed_cost(loads,pts,zeros,zeros,pos,vel,rot,minimum_load=5.)
 assert torch.equal(z,torch.zeros_like(z))
 print(device,'default bit parity512 owners and strict1N/5N boundary, angular RMS, unloaded owners passed')
for x in (0.,-1.,float('nan'),float('inf')):
 for f in (validate_minimum_load,lambda v:new.point_speed_cost(*args,minimum_load=v)):
  try:f(x)
  except ValueError:pass
  else:raise AssertionError('Invalid load accepted')
assert not validate_load_resume({},1.)['changed']
assert not validate_load_resume({'minimum_normal_load':5.},5.)['changed']
for saved,requested in (({},5.),({'minimum_normal_load':5.},1.)):
 try:validate_load_resume(saved,requested)
 except ValueError:pass
 else:raise AssertionError('Silent threshold change accepted')
 assert validate_load_resume(saved,requested,True)['changed']
# Run the actual installer + ground sensor adapter without importing the simulator.
sys.modules['unitree_rl_lab.tasks.mimic.mdp.point_slip']=new
class Scene(dict):
 def update(self,*a,**kw):return 'updated'
class Manager:
 def __init__(self):self.term=NS(func=new.normalized_moving_point_slip,weight=-1.)
 def get_term_cfg(self,name):assert name=='moving_point_slip';return self.term
 def set_term_cfg(self,name,term):self.term=term
for threshold in (1.,5.):
 env=NS(num_envs=1,device='cpu',cfg=NS(decimation=4,sim=NS(dt=.005)),physics_dt=.005,_sim_step_counter=0);env.unwrapped=env
 body=NS(body_com_pos_w=torch.zeros(1,2,3),body_lin_vel_w=torch.zeros(1,2,3),body_ang_vel_w=torch.zeros(1,2,3));body.body_lin_vel_w[:,:,0]=.1
 class View:
  def get_contact_data(self,dt):
   assert dt==.005
   return torch.tensor([2. if env._sim_step_counter%2 else 6.]),torch.zeros(1,3),None,None,torch.ones(1,dtype=torch.long),torch.zeros(1,dtype=torch.long)
 env.scene=Scene(robot=NS(data=body,body_names=['left_foot','right_foot']),slip_left_foot=NS(contact_physx_view=View()),slip_right_foot=NS(contact_physx_view=View()))
 env.reward_manager=Manager();env.command_manager=NS(get_term=lambda name:NS(cfg=NS(body_names=['base_link']),body_lin_vel_w=torch.tensor([[[.2,0.,0.]]])))
 audit=install(env,minimum_load=threshold);assert audit['minimum_normal_load']==threshold
 for i in range(1,5):
  env._sim_step_counter=i;assert env.scene.update(.005)=='updated';env.scene.update(.005)
 assert env._physics_slip_accumulator.count==4
 result=env.reward_manager.term.func(env);assert torch.allclose(result,torch.tensor([2. if threshold==1. else 1.]))
 assert env._physics_slip_accumulator.completed_windows==1 and env._physics_slip_accumulator.count==0
 try:env.reward_manager.term.func(env)
 except RuntimeError:pass
 else:raise AssertionError('Double consume accepted')
 try:install(env,minimum_load=threshold)
 except RuntimeError:pass
 else:raise AssertionError('Double install accepted')
print('Physics4-step averaging, duplicate refresh, normalization, default+5N sensor forwarding, resume and invalid-load guards passed')
