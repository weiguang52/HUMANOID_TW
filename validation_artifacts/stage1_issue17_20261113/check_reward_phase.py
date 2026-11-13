import ast,sys,json
from pathlib import Path
import torch
from types import SimpleNamespace as NS
root=Path('/root/gpufree-data/tw_stage1/issue17_20261113')
def load(path,name):
 s=Path(path).read_text();f=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name);scope={'torch':torch};exec('from __future__ import annotations\n'+ast.get_source_segment(s,f),scope);return scope[name]
gait=load('source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py','feet_gait');swing=load('source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/tw/rewards.py','feet_swing_height')
N=4000;steps=torch.arange(N)%1000;contact=(torch.arange(N)[:,None]+torch.arange(2)[None,:])%3==0;heights=torch.rand(N,2)*.06
cfg=NS(name='feet',body_ids=[0,1])
def env(steps,contact,heights):
 feet=NS(data=NS(current_contact_time=contact.float(),body_pos_w=torch.stack([torch.zeros_like(heights),torch.zeros_like(heights),heights],dim=-1)))
 return NS(scene=type('Scene',(dict,),{})(feet=feet),episode_length_buf=steps,step_dt=.02,num_envs=N,device='cpu',command_manager=NS(get_command=lambda _:torch.tensor([.5,0.,0.]).repeat(N,1)))
a=env(steps,contact,heights);a.scene.sensors=a.scene;b=env(steps+10,contact.flip(1),heights.flip(1));b.scene.sensors=b.scene
results={}
for name,fn,kw in [('feet_gait',gait,{'sensor_cfg':cfg}),('feet_swing_height',swing,{'asset_cfg':cfg,'target_height':.03,'rest_height':.011})]:
 x=fn(a,period=.4,offset=[0.,.5],threshold=.55,command_name='base_velocity',**kw);y=fn(b,period=.4,offset=[0.,.5],threshold=.55,command_name='base_velocity',**kw)
 results[name]={'equal':bool(torch.equal(x,y)),'mismatch_count':int((x!=y).sum()),'max_abs_error':float(abs(x-y).max()),'mismatch_step_examples':steps[x!=y][:20].tolist()}
print(json.dumps(results,indent=2));(root/'reward_phase_initial_check.json').write_text(json.dumps(results,indent=2)+'\n')
