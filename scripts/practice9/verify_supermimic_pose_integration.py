"""Cross-check diagnostic Torch integration against float64 NumPy on recorded data."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from supermimic.algo.networks import WorldModel,WorldModelCfg
from supermimic_pose_integration_diagnostic import PoseIntegrationDiagnostic
p=argparse.ArgumentParser(description=__doc__)
for key in ['checkpoint','audit','branches','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();c=torch.load(a.checkpoint,map_location='cpu',weights_only=False);w=c['world_model'];cfg=c['cfg'];m=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda();m.load_state_dict(w);m.eval();wrapped=PoseIntegrationDiagnostic(m,a.audit,hashlib.sha256(a.checkpoint.read_bytes()).hexdigest())
parts=[]
for r in json.loads((a.branches/'results.json').read_text())['rows']:
 if r['split']=='validation':
  with np.load(a.branches/r['file']) as z:parts.append({k:z[k].copy() for k in ['state','action','context']})
d={k:np.concatenate([v[k] for v in parts]) for k in ['state','action','context']};s,u,ctx=[torch.tensor(d[k],device='cuda:0') for k in ['state','action','context']]
with torch.no_grad():pred=m(s,u,ctx);actual=wrapped(s,u,ctx)
for sl in [slice(7,13),slice(43,73)]:torch.testing.assert_close(actual[:,sl],pred[:,sl],rtol=0,atol=0)
x=s.cpu().numpy().astype(np.float64);n=pred.cpu().numpy().astype(np.float64);expected=n.copy();alpha=wrapped.alpha;dt=.02
expected[:,:3]=x[:,:3]+dt*((1-alpha['translation'])*x[:,7:10]+alpha['translation']*n[:,7:10]);expected[:,13:43]=x[:,13:43]+dt*((1-alpha['joint'])*x[:,43:73]+alpha['joint']*n[:,43:73]);rv=dt*((1-alpha['rotation'])*x[:,10:13]+alpha['rotation']*n[:,10:13]);theta=np.linalg.norm(rv,axis=-1,keepdims=True);q=np.concatenate([np.cos(theta/2),.5*np.sinc(theta/(2*np.pi))*rv],axis=-1);b=x[:,3:7];quat=np.concatenate([q[:,:1]*b[:,:1]-np.sum(q[:,1:]*b[:,1:],axis=-1,keepdims=True),q[:,:1]*b[:,1:]+b[:,:1]*q[:,1:]+np.cross(q[:,1:],b[:,1:])],axis=-1);expected[:,3:7]=quat/np.linalg.norm(quat,axis=-1,keepdims=True)
error=float(np.max(np.abs(actual.cpu().numpy()-expected)));assert error<1e-6
ua=u[:60].clone().requires_grad_();y=wrapped(s[:60],ua,ctx[:60]);grad=torch.autograd.grad(y.square().sum(),ua)[0];assert torch.isfinite(grad).all() and bool((grad!=0).any())
assert bool(torch.isfinite(actual).all());torch.testing.assert_close(actual[:,3:7].norm(dim=-1),torch.ones(len(s),device='cuda:0'),rtol=0,atol=1e-6)
a.output.write_text(json.dumps(dict(passed=True,recorded_validation_states=len(s),numpy_float64_vs_torch_max_abs_error=error,velocity_fields_exactly_unchanged=True,action_grad_finite=True,action_grad_norm=float(grad.norm()),unit_quaternions=True,scope='Numerical and differentiability checks only; not model-accuracy or control acceptance'),indent=2));print(a.output.read_text())
