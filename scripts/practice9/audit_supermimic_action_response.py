"""Paired reset one-step action derivatives: recorded physics versus learned dynamics."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.trainer import SuperMimicTrainer,_squeeze
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import enable_tw_velocity_limits,disable_tw_velocity_limits,PHYSICS_CONTRACT

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
m=json.loads(a.manifest.read_text());assert m['synthetic_diagnostic']
c=torch.load(a.checkpoint,map_location='cuda:0',weights_only=False)
assert c['cfg']['extra']['tw_physics_contract']==PHYSICS_CONTRACT
asset=Path(os.environ['TW_SUPERMIMIC_MJCF'])
assert hashlib.sha256(asset.read_bytes()).hexdigest()==c['cfg']['extra']['tw_mjcf_sha256']
plan=dict(epsilons=[.02,.05],action_center='zero normalized residual',steps=1,
 checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
 manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
 scope='Diagnostic initial nominal state, central differences; no training or policy acceptance')
(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
cfg=SuperMimicCfg(**c['cfg']);cfg.compile=False;cfg.device='cuda:0'
e=tw_supermimic_env_cfg([m['motions'][0]['file']],num_envs=1,play=True)
e.seed=42;e.events={};e.auto_reset=False;e.commands['motion'].pose_range={}
e.commands['motion'].velocity_range={};e.commands['motion'].joint_position_range=(0.,0.)
env=ManagerBasedRlEnv(e,device='cuda:0');enable_tw_velocity_limits(env)
t=SuperMimicTrainer(env,cfg,log_dir=a.output/'runtime');t.load(a.checkpoint);t.initialize()
t.world_model.eval();t.actor_critic.eval();cmd=t.command
records=[]
def reset():
 env.reset();cmd.reset_to_frame(torch.tensor([0],device='cuda:0'),0,0);env.sim.forward()
 return t.extractor.state().detach().clone(),t.extractor.context().detach().clone()
def reward(s,action,reference):
 b=t.forward_kinematics(s[:,:3],s[:,3:7],s[:,7:10],s[:,10:13],s[:,13:43],s[:,43:])
 return t.reward.forward_from_bodies(s,action,torch.zeros_like(action),reference,*b)[0]
try:
 s0,context=reset();reference=_squeeze(t.library.gather(torch.tensor([0],device='cuda:0'),torch.tensor([0],device='cuda:0'),1))
 zero=torch.zeros((1,30),device='cuda:0')
 with torch.no_grad():
  baseline_pred=t.world_model(s0,zero,context)
  env.step(zero);baseline_real=t.extractor.state().detach().clone()
 bias=(baseline_real-baseline_pred).detach()
 action=torch.zeros((1,30),device='cuda:0',requires_grad=True)
 pred=t.world_model(s0,action,context);r=reward(pred,action,reference)
 gradient=torch.autograd.grad(r.sum(),action,retain_graph=True)[0].detach().cpu().numpy()[0]
 corrected_reward=reward(pred+bias,action,reference)
 corrected_gradient=torch.autograd.grad(corrected_reward.sum(),action)[0].detach().cpu().numpy()[0]
 for eps in plan['epsilons']:
  for j in range(30):
   for sign in [-1,1]:
    s,ctx=reset();torch.testing.assert_close(s,s0,rtol=0,atol=1e-7)
    torch.testing.assert_close(ctx,context,rtol=0,atol=1e-7)
    u=torch.zeros((1,30),device='cuda:0');u[0,j]=sign*eps
    with torch.no_grad():
     predicted=t.world_model(s,u,ctx);result=env.step(u);actual=t.extractor.state().clone()
     rp=reward(predicted,u,reference);ra=reward(actual,u,reference)
    records.append(dict(epsilon=eps,joint=j,sign=sign,state=s[0].cpu().numpy(),
     next_state=actual[0].cpu().numpy(),predicted=predicted[0].cpu().numpy(),
     actual_reward=float(ra),predicted_reward=float(rp),terminated=bool((result[2]|result[3])[0])))
   print(eps,j,flush=True)
 arrays={k:np.array([row[k] for row in records]) for k in records[0]}
 assert all(np.isfinite(v).all() for v in arrays.values())
 np.savez_compressed(a.output/'paired_responses.npz',**arrays)
 report={'plan':plan,'autograd_predicted_reward_gradient':gradient.tolist(),'diagnostic_bias_corrected_gradient':corrected_gradient.tolist(),'baseline_prediction_error':(baseline_pred-baseline_real)[0].cpu().numpy().tolist(),'bias_correction_scope':'Local diagnostic using actual zero-action response only, no policy or model update','rows':[]}
 def compare(x,y):
  den=np.linalg.norm(x)*np.linalg.norm(y)
  return dict(actual_norm=float(np.linalg.norm(x)),predicted_norm=float(np.linalg.norm(y)),
   cosine=float(np.sum(x*y)/den) if den>1e-12 else None,rmse=float(np.sqrt(np.mean((x-y)**2))))
 for eps in plan['epsilons']:
  subset=[x for x in records if x['epsilon']==eps]
  deriv={}
  for key in ['next_state','predicted','actual_reward','predicted_reward']:
   deriv[key]=np.array([(subset[j*2+1][key]-subset[j*2][key])/(2*eps) for j in range(30)])
  row={'epsilon':eps,'reward_gradient':compare(deriv['actual_reward'],deriv['predicted_reward']),
   'autograd_vs_finite_difference':compare(deriv['predicted_reward'],gradient),
   'diagnostic_bias_corrected_vs_actual_reward_gradient':compare(deriv['actual_reward'],corrected_gradient),
   'terminations':sum(x['terminated'] for x in subset)}
  for name,sl in [('root_linear_velocity',slice(7,10)),('root_angular_velocity',slice(10,13)),('joint_position',slice(13,43)),('joint_velocity',slice(43,73))]:
   row[name]=compare(deriv['next_state'][:,sl],deriv['predicted'][:,sl])
  report['rows'].append(row)
 (a.output/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
finally:
 env.close();disable_tw_velocity_limits(env)
