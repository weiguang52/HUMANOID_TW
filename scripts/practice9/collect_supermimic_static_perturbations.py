"""Fixed synthetic nominal-pose perturbations; not learned-policy success."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import enable_tw_velocity_limits,disable_tw_velocity_limits,LIMITS
from supermimic.algo.state import SuperMimicStateSpec,StateExtractor
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
m=json.loads(a.manifest.read_text());assert m['synthetic_diagnostic'];plan=dict(seeds=[42,44,43,45],fit_seeds=[42,44],validation_seeds=[43,45],amplitudes=[0,.02,.05,.1],frequency_hz=[.2,1.],ramp_s=1.,steps=1000,note='Fixed before sampling. Synthetic forcing-seed split, not motion-source generalization; no learned policy. Original terminations retained.')
(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
cfg=tw_supermimic_env_cfg([m['motions'][0]['file']],num_envs=1,play=True);cfg.events={};cfg.seed=42;cfg.auto_reset=False
cfg.commands['motion'].pose_range={};cfg.commands['motion'].velocity_range={};cfg.commands['motion'].joint_position_range=(0.,0.)
env=ManagerBasedRlEnv(cfg,device='cuda:0');enable_tw_velocity_limits(env);robot=env.scene['robot'];cmd=env.command_manager.get_term('motion');term=env.action_manager.get_term('joint_pos');ids=term._target_ids
spec=SuperMimicStateSpec.from_env(env,cmd.cfg.body_names);extract=StateExtractor(env,spec,torso_body_name='chest',left_foot_geom_pattern=r'^left_foot_collision(_\d+)?$',right_foot_geom_pattern=r'^right_foot_collision(_\d+)?$');order=torch.arange(30,device='cuda:0')[ids].cpu().tolist()
(a.output/'control_mapping.json').write_text(json.dumps(dict(state_joint_names=list(robot.joint_names),action_joint_names=[robot.joint_names[i] for i in order],offset=term._offset.cpu().tolist(),scale=term._scale.cpu().tolist(),clip=repr(term.cfg.clip)),indent=2));caps=torch.tensor([LIMITS[n] for n in robot.joint_names],device='cuda:0');reports=[]
try:
 for seed in plan['seeds']:
  rng=np.random.default_rng(seed);frequency=torch.tensor(rng.uniform(.2,1.,30),device='cuda:0',dtype=torch.float32);phase=torch.tensor(rng.uniform(-math.pi,math.pi,30),device='cuda:0',dtype=torch.float32)
  for amp in plan['amplitudes']:
   env.reset();cmd.reset_to_frame(torch.tensor([0],device='cuda:0'),0,0);env.sim.forward();torch.testing.assert_close(robot.data.joint_pos,robot.data.default_joint_pos,atol=1e-6,rtol=0)
   records={k:[] for k in ('state','action','context','next_state')};reasons=[]
   for step in range(1000):
    t=step*.02;action=(amp*min(t,1.)*torch.sin(2*math.pi*frequency*t+phase))[None]
    s=extract.state().clone();c=extract.context().clone();result=env.step(action);n=extract.state().clone();assert torch.all(robot.data.joint_vel.abs()<=caps+1e-5)
    for k,v in [('state',s),('action',action),('context',c),('next_state',n)]:
     assert torch.isfinite(v).all();records[k].append(v[0].detach().cpu().numpy().copy())
    if bool((result[2]|result[3])[0]):reasons=[name for name in env.termination_manager.active_terms if bool(env.termination_manager.get_term(name)[0])];break
   filename=f'seed{seed}_amp{amp:.2f}.npz';np.savez_compressed(a.output/filename,**{k:np.stack(v) for k,v in records.items()})
   r=dict(id=f'synthetic_seed{seed}',split='fit' if seed in plan['fit_seeds'] else 'validation',mode=f'amplitude_{amp:.2f}',steps=len(records['state']),planned_steps=1000,termination_reasons=reasons,trace=filename,synthetic=True);reports.append(r);print(json.dumps(r),flush=True);(a.output/'partial.json').write_text(json.dumps(reports,indent=2))
 (a.output/'results.json').write_text(json.dumps(dict(rows=reports,note=plan['note']),indent=2))
finally:env.close();disable_tw_velocity_limits(env)
