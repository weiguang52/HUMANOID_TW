"""Fixed PD factorial diagnostic; not a learned-control or curriculum acceptance."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import enable_tw_velocity_limits,disable_tw_velocity_limits,LIMITS,PHYSICS_CONTRACT
from supermimic.algo.state import SuperMimicStateSpec,StateExtractor
p=argparse.ArgumentParser(description=__doc__)
for name in ['manifest','training-config','output']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
m=json.loads(a.manifest.read_text());assert m['synthetic_diagnostic'] and len(m['motions'])==1
with np.load(m['motions'][0]['file'],allow_pickle=False) as z:
 assert np.allclose(z['joint_pos'],z['joint_pos'][0],rtol=0,atol=1e-7)
 assert np.allclose(z['body_pos_w'],z['body_pos_w'][0],rtol=0,atol=1e-7)
 assert all(np.max(np.abs(z[k]))<1e-7 for k in ['joint_vel','body_lin_vel_w','body_ang_vel_w'])
c=json.loads(a.training_config.read_text());extra=c['extra']
assert extra['tw_physics_contract']==PHYSICS_CONTRACT
assert extra['tw_mjcf_sha256']==hashlib.sha256(Path(os.environ['TW_SUPERMIMIC_MJCF']).read_bytes()).hexdigest()
noise_std=float(c['action_noise']);assert noise_std>=0
profiles=[('nominal_zero',False,False),('nominal_noise',False,True),('perturbed_zero',True,False),('perturbed_noise',True,True)]
plan=dict(seed=42,noise_seed=10042,num_envs=32,max_steps=1000,action_noise_std=noise_std,profiles=profiles,
 inactive_world_handling='Reset and nominally park only already stopped worlds to satisfy backend; assert live states/context/clock unchanged; never record another episode',physics_contract=PHYSICS_CONTRACT,training_config_sha256=hashlib.sha256(a.training_config.read_bytes()).hexdigest(),
 manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 scope='PD nominal targets only; events disabled, start frame fixed. Not exact training return-time/event distribution; PD failure is not proof no policy can recover. First episode only, no autoreset.')
(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
noise=np.random.default_rng(10042).normal(size=(1000,32,30)).astype(np.float32)*noise_std
np.savez_compressed(a.output/'fixed_noise.npz',noise=noise)
initials={};rows=[];summary=[]
for name,perturb,explore in profiles:
 cfg=tw_supermimic_env_cfg([m['motions'][0]['file']],num_envs=32,play=False)
 cfg.events={};cfg.seed=42;cfg.auto_reset=False;cfg.episode_length_s=30.
 cmd_cfg=cfg.commands['motion'];cmd_cfg.sampling_mode='start'
 ranges=dict(pose=cmd_cfg.pose_range,velocity=cmd_cfg.velocity_range,joint=cmd_cfg.joint_position_range)
 if not perturb:cmd_cfg.pose_range={};cmd_cfg.velocity_range={};cmd_cfg.joint_position_range=(0.,0.)
 env=ManagerBasedRlEnv(cfg,device='cuda:0');enable_tw_velocity_limits(env)
 records={k:[] for k in ['state','next_state','action','context']};parking_errors=[]
 try:
  assert abs(env.step_dt-.02)<1e-9
  robot=env.scene['robot'];cmd=env.command_manager.get_term('motion');term=env.action_manager.get_term('joint_pos')
  assert list(robot.joint_names)==extra['tw_state_joint_names']
  assert list(term.target_names)==extra['tw_action_joint_names']
  spec=SuperMimicStateSpec.from_env(env,cmd.cfg.body_names)
  extract=StateExtractor(env,spec,torso_body_name='chest',left_foot_geom_pattern=r'^left_foot_collision(_\d+)?$',right_foot_geom_pattern=r'^right_foot_collision(_\d+)?$')
  env.reset();env.sim.forward()
  assert bool((cmd.clip_ids==0).all())
  cmd.time_steps.zero_()  # Standardize episode clock without overwriting perturbed physical state.
  term.process_actions(torch.zeros((32,30),device='cuda:0'))
  torch.testing.assert_close(term._processed_actions,cmd.joint_pos[:,term._target_ids],rtol=0,atol=1e-5)
  s0=extract.state().clone();ctx0=extract.context().clone()
  np.savez_compressed(a.output/f'{name}_initial.npz',state=s0.cpu().numpy(),context=ctx0.cpu().numpy())
  if perturb in initials:
   torch.testing.assert_close(s0.cpu(),initials[perturb],rtol=0,atol=1e-6)
  else:initials[perturb]=s0.cpu().clone()
  caps=torch.tensor([LIMITS[n] for n in robot.joint_names],device='cuda:0')
  live=torch.ones(32,dtype=torch.bool,device='cuda:0');lengths=np.zeros(32,dtype=int);reasons=[[] for _ in range(32)]
  for step in range(1000):
   u=torch.tensor(noise[step],device='cuda:0') if explore else torch.zeros((32,30),device='cuda:0')
   u[~live]=0.0
   s=extract.state().clone();ctx=extract.context().clone();result=env.step(u);n=extract.state().clone()
   assert bool((robot.data.joint_vel.abs()<=caps+1e-5).all())
   for key,v in [('state',s),('next_state',n),('action',u),('context',ctx)]:
    assert torch.isfinite(v).all();records[key].append(v.detach().cpu().numpy().copy())
   done=(result[2]|result[3]) & live
   for i in done.nonzero().flatten().cpu().tolist():
    lengths[i]=step+1;reasons[i]=[term_name for term_name in env.termination_manager.active_terms if bool(env.termination_manager.get_term(term_name)[i])]
   live &= ~done
   if not bool(live.any()):break
   reset_ids=(result[2]|result[3]).nonzero().flatten()
   if len(reset_ids):
    active_before=n[live].clone();context_before=extract.context()[live].clone()
    clock_before=cmd.time_steps[live].clone();episode_clock_before=env.episode_length_buf[live].clone()
    env.reset(env_ids=reset_ids)
    cmd.reset_to_frame(reset_ids,0,0)
    after=extract.state().clone()
    torch.testing.assert_close(after[live],active_before,rtol=0,atol=1e-6)
    torch.testing.assert_close(extract.context()[live],context_before,rtol=0,atol=1e-7)
    assert torch.equal(cmd.time_steps[live],clock_before)
    assert torch.equal(env.episode_length_buf[live],episode_clock_before)
    parking_errors.append(float((after[live]-active_before).abs().max()))
    # Same physical instant after a scoped reset cache refresh. Preserve the
    # original failure endpoints, and align live next-state caches with next step.
    records['next_state'][-1][live.cpu().numpy()]=after[live].cpu().numpy()
  for i in range(32):
   length=int(lengths[i]) if lengths[i] else len(records['state'])
   trace={k:np.stack(v)[:length,i] for k,v in records.items()}
   assert np.array_equal(trace['next_state'][:-1],trace['state'][1:])
   file=f'{name}_env{i:02d}.npz';np.savez_compressed(a.output/file,**trace)
   rows.append(dict(profile=name,env_id=i,steps=length,seconds=length*.02,termination_reasons=reasons[i],completed20s=length==1000 and not reasons[i],trace=file))
  group=rows[-32:];summary.append(dict(profile=name,completed20s=sum(x['completed20s'] for x in group),episodes=32,median_seconds=float(np.median([x['seconds'] for x in group])),min_seconds=min(x['seconds'] for x in group),max_seconds=max(x['seconds'] for x in group),parking_max_active_state_error=max(parking_errors,default=0.),reset_ranges=ranges if perturb else dict(pose={},velocity={},joint=[0.,0.])))
  (a.output/'partial.json').write_text(json.dumps(dict(rows=rows,summary=summary),indent=2));print(json.dumps(summary[-1]),flush=True)
 except BaseException:
  if records['state']:
   np.savez_compressed(a.output/f'{name}_partial_arrays.npz',**{k:np.stack(v) for k,v in records.items()})
   (a.output/f'{name}_partial_meta.json').write_text(json.dumps(dict(first_failure_steps=lengths.tolist(),termination_reasons=reasons),indent=2))
  raise
 finally:env.close();disable_tw_velocity_limits(env)
(a.output/'results.json').write_text(json.dumps(dict(plan=plan,rows=rows,summary=summary),indent=2));print('PROBE_COMPLETE',flush=True)
