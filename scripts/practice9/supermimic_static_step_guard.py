"""Fixed finite-step probes on fit-static loss; never a validation guarantee."""
import math
import torch
FACTORS=(1.,.5,.25,0.)
@torch.no_grad()
def guard_actual_step(parameters,before,evaluate,baseline,enabled):
 assert type(enabled) is bool and math.isfinite(baseline)
 assert len(parameters)==len(before)>0
 after=[p.detach().clone() for p in parameters]
 assert all(p.shape==b.shape and torch.isfinite(p).all() and torch.isfinite(b).all() for p,b in zip(after,before))
 delta=[p-b for p,b in zip(after,before)]
 def apply(factor):
  for parameter,start,end,change in zip(parameters,before,after,delta):
   parameter.copy_(start if factor==0 else end if factor==1 else start+factor*change)
 probes=[];chosen=None
 for factor in FACTORS:
  apply(factor);value=float(evaluate());finite=math.isfinite(value);accepted=finite and value<=baseline
  probes.append(dict(factor=factor,loss=value if finite else None,accepted=accepted))
  if chosen is None and accepted:chosen=factor
 if chosen is None:
  apply(0.);raise RuntimeError('Identity fit-static probe failed; preserve this run and investigate')
 applied=chosen if enabled else 1.;apply(applied)
 actual=next(p['loss'] for p in probes if p['factor']==applied)
 return dict(enabled=enabled,baseline_loss=baseline,probes=probes,selected_factor=chosen,applied_factor=applied,applied_loss=actual,applied_nonincrease=actual is not None and actual<=baseline,optimizer_moments_reverted=False,scope='Same fit batch finite loss check only; no validation/control guarantee.')
