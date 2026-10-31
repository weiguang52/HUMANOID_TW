"""Check a measured-start checkpoint by restoring its real CUDA trainer."""
import argparse,hashlib,json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends
from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.trainer import SuperMimicTrainer
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import enable_tw_velocity_limits,disable_tw_velocity_limits
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();configure_torch_backends()
c=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
count=0
def finite(x):
 global count
 if isinstance(x,torch.Tensor):
  assert torch.isfinite(x).all();count+=1
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,(list,tuple)):
  for v in x:finite(v)
finite(c)
cfg=SuperMimicCfg(**c['cfg']);assert cfg.policy_anchor_collection is not None
files=tuple(x['file'] for x in json.loads(a.manifest.read_text())['motions'])
e=tw_supermimic_env_cfg(files,num_envs=8,play=False);e.seed=42;e.episode_length_s=30.
env=ManagerBasedRlEnv(e,device=cfg.device);enable_tw_velocity_limits(env)
try:
 t=SuperMimicTrainer(env,cfg,log_dir=a.output.parent/'restore_audit');t.load(a.checkpoint)
 try:
  pool=t.policy_anchor
  assert pool.metadata==c['policy_anchor_source']
  assert torch.equal(pool.generator.get_state().cpu(),c['policy_anchor_rng_state'])
  first=pool.sample_with_times(256)
  t.load(a.checkpoint)
  second=pool.sample_with_times(256)
  assert all(torch.equal(x,y) for x,y in zip(first,second))
  assert len(pool.metadata['fit_files'])==8 and pool.metadata['windows']==3217
  assert t.iteration==c['iteration']
  readout=t.actor_critic.decoder[-1]
  norm=float(torch.linalg.vector_norm(readout.weight).detach())
  assert norm>0
  report=dict(checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),iteration=t.iteration,finite_tensor_count=count,policy_source=pool.metadata,actual_cuda_trainer_restore=True,sampler_replay_exact=True,readout_weight_norm=norm,promoted=False)
  a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='policy_source'}))
 finally:t.close()
finally:env.close();disable_tw_velocity_limits(env)
