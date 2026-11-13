"""CPU tests execute actual reward/observation function AST, with exact helper globals."""
import ast,importlib.util,json,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion'
spec=importlib.util.spec_from_file_location('clock',BASE/'mdp/integer_gait_clock.py');clock=importlib.util.module_from_spec(spec);spec.loader.exec_module(clock)
def actual(rel,name):
 s=(BASE/rel).read_text();f=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name);g={'torch':torch,**{k:getattr(clock,k) for k in ['integer_clock_enabled','integer_gait_phase','integer_leg_stance']}};exec('from __future__ import annotations\n'+ast.get_source_segment(s,f),g);return g[name]
gait=actual('mdp/rewards.py','feet_gait');swing=actual('robots/tw/rewards.py','feet_swing_height');obs=actual('mdp/observations.py','gait_phase')
def make(steps,contacts,heights,on=True):
 feet=NS(data=NS(current_contact_time=contacts.float(),body_pos_w=torch.stack([torch.zeros_like(heights),torch.zeros_like(heights),heights],-1)))
 scene=type('Scene',(dict,),{})(feet=feet);scene.sensors=scene
 return NS(cfg=NS(issue17_integer_clock=on),scene=scene,episode_length_buf=steps,step_dt=.02,num_envs=len(steps),device='cpu',command_manager=NS(get_command=lambda _:torch.tensor([.5,0.,0.]).repeat(len(steps),1)))
class Tests(unittest.TestCase):
 def test_all1000_steps_all_contact_patterns_and_heights(self):
  k=torch.arange(1000).repeat(4);pattern=torch.arange(4).repeat_interleave(1000);contacts=torch.stack([pattern%2>0,pattern//2>0],1);height=torch.linspace(0,.08,len(k)*2).reshape(-1,2)
  a=make(k,contacts,height);b=make(k+10,contacts.flip(1),height.flip(1));cfg=NS(name='feet',body_ids=[0,1])
  self.assertTrue(torch.equal(gait(a,.4,[0.,.5],cfg,.55,'base_velocity'),gait(b,.4,[0.,.5],cfg,.55,'base_velocity')))
  kw=dict(period=.4,offset=[0.,.5],threshold=.55,target_height=.03,rest_height=.011,asset_cfg=cfg,command_name='base_velocity')
  self.assertTrue(torch.equal(swing(a,**kw),swing(b,**kw)))
 def test_cycle_counts_and_double_support(self):
  e=make(torch.arange(1000),torch.zeros(1000,2),torch.zeros(1000,2));m=clock.integer_leg_stance(e,.4,[0.,.5],.55).reshape(50,20,2)
  self.assertTrue(torch.equal(m.sum(1),torch.full((50,2),11)));self.assertEqual(torch.where(m[0].all(1))[0].tolist(),[0,10]);self.assertTrue(torch.equal(m[:, :,0],m[:,:,1].roll(10,1)))
 def test_observation_clock_and_disabled_legacy(self):
  k=torch.arange(1000);e=make(k,torch.zeros(1000,2),torch.zeros(1000,2));expected=torch.stack([torch.sin((k%20).float()/20*torch.pi*2),torch.cos((k%20).float()/20*torch.pi*2)],1);self.assertTrue(torch.equal(obs(e,.4),expected))
  e.cfg.issue17_integer_clock=False;p=(k*.02)%.4/.4;legacy=torch.stack([torch.sin(p*torch.pi*2),torch.cos(p*torch.pi*2)],1);self.assertTrue(torch.equal(obs(e,.4),legacy))
if __name__=='__main__':unittest.main()
