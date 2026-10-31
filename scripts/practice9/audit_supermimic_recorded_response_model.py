"""Score frozen dynamics against all recorded action branches; no policy update."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.trainer import SuperMimicTrainer,_squeeze
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import enable_tw_velocity_limits,disable_tw_velocity_limits
p=argparse.ArgumentParser(description=__doc__)
for name in ['collection','checkpoint','dynamics-checkpoint','manifest','output']:
 p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
plan=json.loads((a.collection/'plan.json').read_text())
assert plan['checkpoint_sha256']==hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
assert plan['manifest_sha256']==hashlib.sha256(a.manifest.read_bytes()).hexdigest()
c=torch.load(a.checkpoint,map_location='cuda:0',weights_only=False)
d=torch.load(a.dynamics_checkpoint,map_location='cuda:0',weights_only=False)
assert d['kind']=='offline_dynamics_calibration_not_policy_checkpoint'
for k,v in d['world_model'].items():
 if not k.startswith('net.'):assert torch.equal(v,c['world_model'][k]),k
asset=Path(os.environ['TW_SUPERMIMIC_MJCF'])
assert hashlib.sha256(asset.read_bytes()).hexdigest()==c['cfg']['extra']['tw_mjcf_sha256']
m=json.loads(a.manifest.read_text());assert m['synthetic_diagnostic']
cfg=SuperMimicCfg(**c['cfg']);cfg.compile=False;cfg.device='cuda:0'
e=tw_supermimic_env_cfg([m['motions'][0]['file']],num_envs=1,play=True)
e.seed=42;e.events={};e.auto_reset=False;e.commands['motion'].pose_range={}
e.commands['motion'].velocity_range={};e.commands['motion'].joint_position_range=(0.,0.)
env=ManagerBasedRlEnv(e,device='cuda:0');enable_tw_velocity_limits(env)
t=SuperMimicTrainer(env,cfg,log_dir=a.output/'runtime');t.load(a.checkpoint);t.initialize()
t.world_model.load_state_dict(d['world_model'],strict=True);t.world_model.eval()
try:
 env.reset();t.command.reset_to_frame(torch.tensor([0],device='cuda:0'),0,0);env.sim.forward()
 for _ in range(plan.get('warmup_steps',0)):
  result=env.step(torch.zeros((1,30),device='cuda:0'));assert not bool((result[2]|result[3])[0])
 ctx=t.extractor.context().detach();s=t.extractor.state().detach()
 with np.load(a.collection/'paired_responses.npz') as z:trace={k:z[k].copy() for k in z.files}
 count=len(plan['epsilons'])*60
 assert len(trace['state'])==count and all(np.isfinite(v).all() for v in trace.values())
 states=torch.tensor(trace['state'],device='cuda:0')
 torch.testing.assert_close(states,s.expand_as(states),rtol=0,atol=1e-5)
 reference=_squeeze(t.library.gather(torch.zeros(count,device='cuda:0',dtype=torch.long),torch.full((count,),plan.get('warmup_steps',0),device='cuda:0',dtype=torch.long),1))
 actions=torch.zeros((count,30),device='cuda:0')
 actions[torch.arange(count),torch.tensor(trace['joint'])]=torch.tensor(trace['epsilon']*trace['sign'],device='cuda:0',dtype=torch.float32)
 def reward(state,u):
  b=t.forward_kinematics(state[:,:3],state[:,3:7],state[:,7:10],state[:,10:13],state[:,13:43],state[:,43:])
  return t.reward.forward_from_bodies(state,u,torch.zeros_like(u),reference,*b)[0]
 with torch.no_grad():
  predicted=t.world_model(states,actions,ctx.expand(count,-1));actual=torch.tensor(trace['next_state'],device='cuda:0')
  actual_reward=reward(actual,actions)
  torch.testing.assert_close(actual_reward,torch.tensor(trace['actual_reward'],device='cuda:0',dtype=torch.float32),rtol=0,atol=1e-7)
  predicted_reward=reward(predicted,actions).cpu().numpy()
 pred=predicted.cpu().numpy();rows=[]
 def compare(x,y):
  den=np.linalg.norm(x)*np.linalg.norm(y)
  return dict(actual_norm=float(np.linalg.norm(x)),predicted_norm=float(np.linalg.norm(y)),cosine=float(np.sum(x*y)/den) if den>1e-12 else None,rmse=float(np.sqrt(np.mean((x-y)**2))))
 for eps in plan['epsilons']:
  ids=np.flatnonzero(trace['epsilon']==eps);assert len(ids)==60
  for j in range(30):assert list(trace['joint'][ids[j*2:j*2+2]])==[j,j] and list(trace['sign'][ids[j*2:j*2+2]])==[-1,1]
  dr=(trace['actual_reward'][ids][1::2]-trace['actual_reward'][ids][::2])/(2*eps)
  dp=(predicted_reward[ids][1::2]-predicted_reward[ids][::2])/(2*eps)
  row=dict(epsilon=eps,reward_gradient=compare(dr,dp))
  for name,sl in [('root_linear_velocity',slice(7,10)),('root_angular_velocity',slice(10,13)),('joint_position',slice(13,43)),('joint_velocity',slice(43,73))]:
   x=trace['next_state'][ids,sl];y=pred[ids,sl]
   row[name]=compare((x[1::2]-x[::2])/(2*eps),(y[1::2]-y[::2])/(2*eps))
  rows.append(row)
 np.savez_compressed(a.output/'model_responses.npz',predicted=pred,predicted_reward=predicted_reward,context=ctx.cpu().numpy())
 report=dict(rows=rows,collection=str(a.collection),dynamics_sha256=hashlib.sha256(a.dynamics_checkpoint.read_bytes()).hexdigest(),actual_reward_reconstruction='passed atol1e-7',scope='Same recorded physical branches; context reconstructed under identical deterministic events-disabled setup and warmup, original trace did not store context. Tracking reward only; no oracle correction or policy update.')
 (a.output/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
finally:env.close();disable_tw_velocity_limits(env)
