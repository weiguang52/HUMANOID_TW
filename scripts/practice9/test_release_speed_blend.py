"""Legacy zero-tau parity and independent release/reset on CPU and CUDA."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys,torch,numpy as np,json,hashlib,ast
repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo/'scripts/rsl_rl'))
from reference_speed_blend import ReferenceSpeedBlend
from release_speed_blend import ReleaseSpeedBlend
from release_gate import TorchReleaseGate
out=[]
def base(device):
 b=ReferenceSpeedBlend.__new__(ReferenceSpeedBlend);b.hip_filter=None;b.leg_filter=None;b.hip_trace=[];b.leg_trace=[];b.trace=[];b.audit={};b.index=0;b.maximum=1.;b.low=.05;b.high=.1
 b.command=NS(body_lin_vel_w=torch.zeros(2,1,3,device=device));b.base=NS(teacher=lambda x:x[:,:30]);return b
for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
 a=base(device);b=base(device);wrapped=ReleaseSpeedBlend(b,0.,.02);obs={'policy':torch.arange(390,device=device,dtype=torch.float32).reshape(2,195)*.001};obs['contact_base_policy']=obs['policy'];action=torch.ones(2,30,device=device)*.4;before=action.clone();rng=torch.random.get_rng_state().clone()
 for i in range(32):
  speed=torch.tensor([0.,.075,.15,.05][i%4],device=device);a.command.body_lin_vel_w[:,0,0]=speed;b.command.body_lin_vel_w[:,0,0]=speed
  x=a(obs,action);y=wrapped(obs,action);assert torch.equal(x,y)
  done=torch.tensor([i%3==0,i%5==0],device=device);a.reset(done);wrapped.reset(done)
 assert np.array_equal(np.asarray(a.trace),np.asarray(wrapped.trace));assert torch.equal(action,before) and torch.equal(rng,torch.random.get_rng_state())
 b=base(device);w=ReleaseSpeedBlend(b,.2,.02);b.command.body_lin_vel_w[:,0,0]=.15;assert torch.equal(w(obs,action),action)
 b.command.body_lin_vel_w.zero_();w(obs,action);first=w.gate.previous.clone();assert (first>0).all();w.reset(torch.tensor([True,False],device=device));w(obs,action);assert w.gate.previous[0]==0 and w.gate.previous[1]>0
 out.append({'device':device,'tau0_legacy_actions_exact':True,'raw_trace_exact':True,'per_environment_reset':True,'input_rng_unchanged':True})
for device in ['cpu']:
 b=base(device);b.hip_filter=object()
 try:ReleaseSpeedBlend(b,.2,.02)
 except ValueError:pass
 else:raise AssertionError('Unsupported combination accepted')
prototype=Path('/root/gpufree-data/datasets/practice9/release_gate_20261026/torch_gate.py');actual=repo/'scripts/rsl_rl/release_gate.py';
def executable_ast(path):
 t=ast.parse(path.read_text())
 if isinstance(t.body[0],ast.Expr) and isinstance(t.body[0].value,ast.Constant) and isinstance(t.body[0].value.value,str):t.body.pop(0)
 return ast.dump(t)
assert executable_ast(prototype)==executable_ast(actual)
root=prototype.parent;(root/'integration_unit_audit.json').write_text(json.dumps({'passed':True,'results':out,'tested_gate_sha256':hashlib.sha256(actual.read_bytes()).hexdigest(),'prototype_AST_exact':True,'simulation_identity':'pending'},indent=2));print(out)
