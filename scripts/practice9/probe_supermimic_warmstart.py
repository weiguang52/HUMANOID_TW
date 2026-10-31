"""Isolated one-step solver-cache intervention; no production physics changes."""
import argparse,hashlib,json,math,os
from pathlib import Path
import numpy as np
import torch,warp as wp,mujoco
from mjlab.envs import ManagerBasedRlEnv
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import enable_tw_velocity_limits,disable_tw_velocity_limits,PHYSICS_CONTRACT,LIMITS
from supermimic.algo.state import SuperMimicStateSpec,StateExtractor
p=argparse.ArgumentParser(description=__doc__)
for key in ['manifest','source-checkpoint','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
m=json.loads(a.manifest.read_text());assert m['synthetic_diagnostic'] and len(m['motions'])==1
c=torch.load(a.source_checkpoint,map_location='cpu',weights_only=False);extra=c['cfg']['extra'];assert extra['tw_physics_contract']==PHYSICS_CONTRACT
assert extra['tw_mjcf_sha256']==hashlib.sha256(Path(os.environ['TW_SUPERMIMIC_MJCF']).read_bytes()).hexdigest()
profiles=[('pd',0),('pd',10),('pd',50),('pd',100),('forcing_seed42_amp0.05',25)]
plan=dict(num_envs=61,profiles=profiles,epsilons=[.02,.05],conditions=['native','zero_warmstart','native_repeat'],seed=42,dt=.02,replay_atol=1e-5,intervention='Only initial qacc_warmstart zeroed, once before one control step; original substep solver remains unchanged',physics_contract=PHYSICS_CONTRACT,source_checkpoint_sha256=hashlib.sha256(a.source_checkpoint.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Isolated solver initialization diagnostic, not policy or skill acceptance')
(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
e=tw_supermimic_env_cfg([m['motions'][0]['file']],num_envs=61,play=True);e.seed=42;e.events={};e.auto_reset=False;e.commands['motion'].pose_range={};e.commands['motion'].velocity_range={};e.commands['motion'].joint_position_range=(0.,0.)
env=ManagerBasedRlEnv(e,device='cuda:0');enable_tw_velocity_limits(env);cmd=env.command_manager.get_term('motion');robot=env.scene['robot'];term=env.action_manager.get_term('joint_pos');spec=SuperMimicStateSpec.from_env(env,cmd.cfg.body_names);extract=StateExtractor(env,spec,torso_body_name='chest',left_foot_geom_pattern=r'^left_foot_collision(_\d+)?$',right_foot_geom_pattern=r'^right_foot_collision(_\d+)?$')
assert list(robot.joint_names)==extra['tw_state_joint_names'] and list(term.target_names)==extra['tw_action_joint_names']
zero=torch.zeros((61,30),device='cuda:0');ids=torch.arange(61,device='cuda:0');caps=torch.tensor([LIMITS[n] for n in robot.joint_names],device='cuda:0');cache=wp.to_torch(env.sim.wp_data.qacc_warmstart);assert cache.shape==(61,env.sim.mj_model.nv)
plan['solver_disableflags']=int(env.sim.mj_model.opt.disableflags);plan['warmstart_disabled']=bool(plan['solver_disableflags'] & int(mujoco.mjtDisableBit.mjDSBL_WARMSTART));(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
rng=np.random.default_rng(42);frequency=torch.tensor(rng.uniform(.2,1.,30),device='cuda:0');phase=torch.tensor(rng.uniform(-math.pi,math.pi,30),device='cuda:0');reports=[]
def reset(profile,warmup):
 env.reset();cmd.reset_to_frame(ids,0,0);env.sim.forward()
 for j in range(warmup):
  action=zero if profile=='pd' else (.05*min(j*.02,1.)*torch.sin(2*math.pi*frequency*(j*.02)+phase))[None].expand(61,-1)
  out=env.step(action);assert not bool((out[2]|out[3]).any()), 'Warmup terminated; no replacement'
 return extract.state().clone(),extract.context().clone()
try:
 for profile,warmup in profiles:
  for eps in plan['epsilons']:
   reference=None
   for condition in plan['conditions']:
    s,ctx=reset(profile,warmup);before=cache.clone();initial_error=0.
    if reference is None:reference=(s.clone(),ctx.clone())
    else:
     initial_error=float((s-reference[0]).abs().max());torch.testing.assert_close(s,reference[0],rtol=0,atol=plan['replay_atol']);torch.testing.assert_close(ctx,reference[1],rtol=0,atol=1e-7)
    if condition=='zero_warmstart':cache.zero_()
    torch.cuda.synchronize();after=cache.clone();assert torch.equal(s,extract.state()) and torch.equal(ctx,extract.context())
    if condition=='zero_warmstart':assert bool((after==0).all())
    else:assert torch.equal(before,after)
    u=zero.clone();u[ids[1:],torch.tensor(np.repeat(np.arange(30),2),device='cuda:0')]=torch.tensor(np.tile([-1,1],30)*eps,device='cuda:0',dtype=torch.float32)
    out=env.step(u);n=extract.state().clone();done=(out[2]|out[3]).clone();assert bool((robot.data.joint_vel.abs()<=caps+1e-5).all())
    filename=f'{profile}_warmup{warmup}_eps{eps:.2f}_{condition}.npz';arrays={k:v.detach().cpu().numpy() for k,v in dict(state=s,context=ctx,action=u,next_state=n,warmstart_before=before,warmstart_after=after,terminated=done).items()};assert all(np.isfinite(v).all() for v in arrays.values());np.savez_compressed(a.output/filename,**arrays)
    row=dict(profile=profile,warmup=warmup,epsilon=eps,condition=condition,file=filename,initial_state_error=initial_error,terminated_count=int(done.sum()),warmstart_before_abs_max=float(before.abs().max()),visible_state_unchanged_by_intervention=True)
    reports.append(row);print(json.dumps(row),flush=True);(a.output/'partial.json').write_text(json.dumps(reports,indent=2))
 (a.output/'results.json').write_text(json.dumps(dict(plan=plan,rows=reports),indent=2));(a.output/'status').write_text('collected_pending_audit\n')
except BaseException:
 (a.output/'status').write_text('failed\n');raise
finally:env.close();disable_tw_velocity_limits(env)
