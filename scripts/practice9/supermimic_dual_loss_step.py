"""Fixed fit-only finite probes; no independent or closed-loop guarantee."""
import copy,math
import torch
DIRECTIONS=('full_saved_momentum','full_zero_first_moment','static_saved_momentum','static_zero_first_moment')
FACTORS=(1.,.5,.25)
def dual_loss_step(parameters,optimizer,full_gradient,static_gradient,evaluate,baseline,clip,enabled):
 parameters=list(parameters);assert type(enabled) is bool
 assert len(parameters)==len(full_gradient)==len(static_gradient)>0
 assert all(math.isfinite(x) for x in baseline)
 before=[p.detach().clone() for p in parameters];old=copy.deepcopy(optimizer.state_dict())
 group=old['param_groups'][0];assert len(old['param_groups'])==1 and group['weight_decay']==0 and not group['amsgrad']
 assert all(torch.isfinite(g).all() for g in [*full_gradient,*static_gradient])
 proposals=[];chosen=None;full_state=None;full_after=None
 def apply(after,factor):
  with torch.no_grad():
   for p,b,v in zip(parameters,before,after):p.copy_(v if factor==1 else b if factor==0 else b+factor*(v-b))
 try:
  for name in DIRECTIONS:
   shadow=[torch.nn.Parameter(p.clone()) for p in before]
   shadow_opt=torch.optim.Adam(shadow,lr=group['lr']);shadow_opt.load_state_dict(copy.deepcopy(old))
   if 'zero_first' in name:
    for state in shadow_opt.state.values():state['exp_avg'].zero_()
   gradient=full_gradient if name.startswith('full_') else static_gradient
   for p,g in zip(shadow,gradient):p.grad=g.detach().clone()
   norm=torch.nn.utils.clip_grad_norm_(shadow,clip);assert torch.isfinite(norm);shadow_opt.step()
   after=[p.detach().clone() for p in shadow];state=copy.deepcopy(shadow_opt.state_dict())
   if name==DIRECTIONS[0]:full_after=after;full_state=state
   for factor in FACTORS:
    apply(after,factor)
    with torch.no_grad():values=tuple(float(x) for x in evaluate())
    accepted=all(math.isfinite(v) and v<=b for v,b in zip(values,baseline))
    proposals.append(dict(direction=name,factor=factor,static_loss=values[0] if math.isfinite(values[0]) else None,full_loss=values[1] if math.isfinite(values[1]) else None,accepted=accepted))
    if accepted and chosen is None:chosen=(name,factor,after,state,values)
  apply(before,0.)
  with torch.no_grad():identity=tuple(float(x) for x in evaluate())
  assert identity[0]==baseline[0] and abs(identity[1]-baseline[1])<=1e-6*max(1.,abs(baseline[1])),'Identity replay inconsistent'
  if enabled and chosen is not None:name,factor,after,state,values=chosen
  elif enabled:name,factor,after,state,values='identity',0.,before,full_state,identity
  else:name,factor,after,state,values=DIRECTIONS[0],1.,full_after,full_state,(proposals[0]['static_loss'],proposals[0]['full_loss'])
  apply(after,factor);optimizer.load_state_dict(state)
  return dict(enabled=enabled,baseline_static=baseline[0],baseline_full=baseline[1],proposals=proposals,identity_static=identity[0],identity_full=identity[1],selected_direction=name,selected_factor=factor,applied_static=values[0],applied_full=values[1],moments_contract='Selected shadow Adam state; reduced parameter step keeps full selected moments; identity advances full-gradient moments',scope='Same fit batch only; no validation guarantee')
 except BaseException:
  apply(before,0.);optimizer.load_state_dict(old);raise
