import importlib.util,math
from pathlib import Path
from types import SimpleNamespace
import pytest,torch
p=Path(__file__).resolve().parents[2]/'scripts/rsl_rl/diagnostic_perturbation.py'
s=importlib.util.spec_from_file_location('perturb',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
@pytest.mark.parametrize('seed',[42,123,2026])
def test_seed_and_magnitude(seed):
 assert m.vector(seed,.05)==m.vector(seed,.05)
 assert math.hypot(*m.vector(seed,.1))==pytest.approx(.1)
 assert m.vector(seed,0)==[0.,0.]
@pytest.mark.parametrize('v',[-1,.11,float('nan'),float('inf')])
def test_invalid(v):
 with pytest.raises(ValueError):m.vector(42,v)
def test_applies_additive_velocity_and_reads_back():
 class Robot:
  def __init__(self):self.v=torch.tensor([[.01,.02,.03,.04,.05,.06]]);self.root_physx_view=self
  def get_root_velocities(self):return self.v
  def write_root_velocity_to_sim(self,v):self.v=v.clone()
 r=Robot();e=SimpleNamespace(scene={'robot':r},step_dt=.02);before=r.v.clone()
 assert m.apply(e,{'push_speed':.05},249,42) is None
 a=m.apply(e,{'push_speed':.05},250,42)
 assert torch.equal(r.v[:,2:],before[:,2:]) and a['max_write_error']==0
 assert torch.allclose(r.v[0,:2]-before[0,:2],torch.tensor(m.vector(42,.05)))
