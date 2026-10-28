import sys,pathlib,types,torch,numpy as np,random
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'rsl_rl'))
from render_warmup import warm_up_renderer
class Fake:
 def __init__(self,device='cpu',physics=False,rng=False,black=False):
  self.device=device;self.physics=physics;self.rng=rng;self.black=black;self.calls=0;self._sim_step_counter=17;self.common_step_counter=3
  self.data=types.SimpleNamespace(joint_pos=torch.zeros(2,30,device=device),joint_vel=torch.ones(2,30,device=device),root_state_w=torch.zeros(2,13,device=device))
  self.scene={'robot':types.SimpleNamespace(data=self.data)}
  self.command=types.SimpleNamespace(time_steps=torch.tensor([0,4],device=device),motion_ids=torch.tensor([2,3],device=device));self.command_manager=types.SimpleNamespace(get_term=lambda _:self.command)
  self.action_manager=types.SimpleNamespace(action=torch.ones(2,30,device=device),prev_action=torch.zeros(2,30,device=device));self.sim=types.SimpleNamespace(current_time=.17,render=self.render_tick)
 def render_tick(self):
  self.calls+=1
  if self.physics:self.data.joint_pos+=1;self._sim_step_counter+=1
  if self.rng:random.random();np.random.rand();torch.rand(1,device=self.device)
 def render(self,recompute=False):
  assert recompute is True
  image=np.zeros((24,32,3),np.uint8)
  if not self.black and self.calls>2:image[4:20,8:24]=[200,30,80]
  return image
assert warm_up_renderer(object(),0) is None
for value in (-1,65,1.5,True,None):
 try:warm_up_renderer(object(),value);raise AssertionError('Invalid frames accepted')
 except ValueError:pass
f=Fake();audit=warm_up_renderer(f,8);assert f.calls==8 and audit['frames']==8 and audit['RNG_unchanged'] and audit['physics_and_command_unchanged'];assert audit['render_observations'][0]['max']==0 and audit['render_observations'][-1]['max']==200
for kw in ({'physics':True},{'rng':True},{'black':True}):
 try:warm_up_renderer(Fake(**kw),8);raise AssertionError('Unsafe/empty warm-up accepted')
 except RuntimeError:pass
print('PASS CPU: legacy no-op,8 render-only calls, unchanged physical/reference/action/RNG, invalid state/RNG/empty rejection')
if '--cuda' in sys.argv:
 assert torch.cuda.is_available();audit=warm_up_renderer(Fake('cuda'),8);assert audit['RNG_unchanged'];print('PASS CUDA RNG/state preservation')
