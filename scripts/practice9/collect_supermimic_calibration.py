"""Real-physics diagnostic controls on frozen development sources, no resets after failure."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import enable_tw_velocity_limits,disable_tw_velocity_limits
from supermimic.algo.state import SuperMimicStateSpec,StateExtractor
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists();a.output.mkdir(parents=True)
rows=[]
for split in ('fit','validation'):
 m=json.loads((a.split_root/f'{split}.json').read_text())
 rows.extend(dict(r,calibration_split=split) for r in m['motions'][:4])
(a.output/'selection.json').write_text(json.dumps(dict(rule='First four manifest entries per frozen development partition, fixed before simulation; no replacement',motions=rows),indent=2))
cfg=tw_supermimic_env_cfg([r['file'] for r in rows],num_envs=1,play=True);cfg.events={};cfg.seed=42;cfg.auto_reset=False
cfg.commands['motion'].pose_range={};cfg.commands['motion'].velocity_range={};cfg.commands['motion'].joint_position_range=(0.,0.)
env=ManagerBasedRlEnv(cfg,device='cuda:0');enable_tw_velocity_limits(env)
cmd=env.command_manager.get_term('motion');robot=env.scene['robot'];term=env.action_manager.get_term('joint_pos');ids=term._target_ids
spec=SuperMimicStateSpec.from_env(env,cmd.cfg.body_names);extract=StateExtractor(env,spec,torso_body_name='chest',left_foot_geom_pattern=r'^left_foot_collision(_\d+)?$',right_foot_geom_pattern=r'^right_foot_collision(_\d+)?$')
target_order=torch.arange(30,device='cuda:0')[ids].cpu().tolist()
(a.output/'control_mapping.json').write_text(json.dumps(dict(state_joint_names=list(robot.joint_names),action_joint_names=[robot.joint_names[i] for i in target_order],offset=term._offset.cpu().tolist(),scale=term._scale.cpu().tolist(),clip=repr(term.cfg.clip)),indent=2))
reports=[]
try:
 for row in rows:
  clip=cmd.library.clip_names.index(Path(row['file']).stem)
  for mode in ('nominal','hold_initial','reference'):
   env.reset();cmd.reset_to_frame(torch.tensor([0],device='cuda:0'),clip,0);env.sim.forward();initial=robot.data.joint_pos.clone();records={k:[] for k in ('state','action','context','next_state')};reasons=[]
   limit=min(1000,int(cmd.library.clip_lengths_t[clip])-1)
   for step in range(limit):
    target=initial if mode=='hold_initial' else cmd.joint_pos
    action=torch.zeros((1,30),device='cuda:0') if mode=='nominal' else (target[:,ids]-term._offset)/term._scale
    term.process_actions(action)
    intended=term._offset.expand_as(action) if mode=='nominal' else target[:,ids]
    torch.testing.assert_close(term._processed_actions,intended,atol=1e-5,rtol=0)
    before=extract.state().clone();context=extract.context().clone()
    result=env.step(action);after=extract.state().clone()
    for k,v in [('state',before),('action',action),('context',context),('next_state',after)]:
     assert torch.isfinite(v).all();records[k].append(v[0].detach().cpu().numpy().copy())
    if bool((result[2]|result[3])[0]):
     reasons=[n for n in env.termination_manager.active_terms if bool(env.termination_manager.get_term(n)[0])];break
   name=f"{row['calibration_split']}_{row['id']}_{mode}.npz";np.savez_compressed(a.output/name,**{k:np.stack(v) for k,v in records.items()})
   report=dict(id=row['id'],split=row['calibration_split'],mode=mode,steps=len(records['state']),planned_steps=limit,termination_reasons=reasons,trace=name)
   reports.append(report);print(json.dumps(report),flush=True);(a.output/'partial.json').write_text(json.dumps(reports,indent=2))
 (a.output/'results.json').write_text(json.dumps(dict(rows=reports,note='Diagnostic PD controls, not learned skills; original termination gates retained. No final holdout used.'),indent=2))
finally:
 env.close();disable_tw_velocity_limits(env)
