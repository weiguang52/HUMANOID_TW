"""Real PPO second-order loss, legacy regressions run separately."""
import runpy,sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
from second_order_anchor import PREVIOUS,OLDER,VALID,second_order_anchor_mse
from anchored_ppo import install_anchor
ns=runpy.run_path(str(Path(__file__).with_name('test_anchored_ppo.py')))
alg,obs,batch=ns['make']();obs[PREVIOUS]=obs['policy']*.4-.2;obs[OLDER]=obs['policy']*-.3+.1;obs[VALID]=torch.ones(32,1);obs[VALID][:3]=0
audit=install_anchor(alg,0.,temporal_coefficient=1.,temporal_order=2);assert audit['temporal_order']==2
with torch.no_grad():
 for p in alg.actor.mlp.parameters():p.add_(.1)
p=obs.clone(False);p['policy']=obs[PREVIOUS];q=obs.clone(False);q['policy']=obs[OLDER]
def loss(a,o,p,q):return second_order_anchor_mse(a.actor(o),a.anchor_teacher(o),a.actor(p),a.anchor_teacher(p),a.actor(q),a.anchor_teacher(q),o[VALID])
with torch.no_grad():before=loss(alg,obs,p,q).item()
state={k:v.clone() for k,v in alg.anchor_teacher.state_dict().items()};batch.advantages.zero_();alg.value_loss_coef=0;alg.entropy_coef=0
result=alg.update()
with torch.no_grad():after=loss(alg,obs,p,q).item()
assert before>0 and abs(result['anchor_temporal_mse']-before)<1e-7 and after<before
assert all(torch.equal(v,alg.anchor_teacher.state_dict()[k]) for k,v in state.items()) and all(p.grad is None for p in alg.anchor_teacher.parameters())
print('Real PPO second-order formula, optimizer descent and frozen teacher passed',before,after)
for order,coef in [(0,1),(3,1),(2,0)]:
 fresh,_,_=ns['make']()
 try:install_anchor(fresh,1.,temporal_coefficient=coef,temporal_order=order)
 except ValueError:pass
 else:raise AssertionError('Unsupported derivative contract accepted')
# Combined contact expansion and historical observations must work with preserved
# Adam state. Do not demand monotonic one-step loss with nonzero old moments.
c=runpy.run_path(str(Path(__file__).with_name('test_contact_policy.py')));cp=c['cp'];o=c['new_obs'].clone();fresh=c['make'](o);checkpoint,_=cp.migrate_checkpoint(fresh,c['checkpoint']);fresh.load(checkpoint,None,True);teacher,_=cp.expand_actor_state(fresh.actor,c['checkpoint']['actor_state_dict']);o[PREVIOUS]=o['policy']*.6-.2;o[OLDER]=o['policy']*.2+.3;o[VALID]=torch.ones(32,1);o[VALID][:2]=0
install_anchor(fresh,0.,teacher,temporal_coefficient=1.,temporal_order=2)
with torch.no_grad():
 first=dict(fresh.actor.named_parameters())['mlp.0.weight'];first[:,4:].add_(.1)
p=o.clone(False);p['policy']=o[PREVIOUS];q=o.clone(False);q['policy']=o[OLDER]
with torch.no_grad():expected=loss(fresh,o,p,q).item()
c['set_batch'](fresh,o);fresh.storage.batch.advantages.zero_();fresh.value_loss_coef=0;fresh.entropy_coef=0;teacher_state={k:v.clone() for k,v in fresh.anchor_teacher.state_dict().items()};result=fresh.update()
assert expected>0 and abs(result['anchor_temporal_mse']-expected)<1e-7
assert first.grad is not None and first.grad[:,4:].abs().sum()>0
assert all(torch.isfinite(v).all() for v in fresh.actor.state_dict().values())
assert all(torch.equal(v,fresh.anchor_teacher.state_dict()[k]) for k,v in teacher_state.items())
print('Contact-expanded original teacher, inherited Adam, second-order PPO and added-input gradient passed',expected)
