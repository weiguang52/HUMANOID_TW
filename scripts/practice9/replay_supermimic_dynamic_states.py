"""Native archived dynamic replay; no model or policy fitting."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np,torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import enable_tw_velocity_limits,disable_tw_velocity_limits,LIMITS,PHYSICS_CONTRACT
from supermimic.algo.state import SuperMimicStateSpec,StateExtractor
p=argparse.ArgumentParser(description=__doc__)
for k in ['collection','manifest','output']:p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--phase',choices=['warm','cold'],required=True);p.add_argument('--smoke',action='store_true')
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
manifest=json.loads(a.manifest.read_text());assert manifest['synthetic_diagnostic'] and len(manifest['motions'])==1
source=json.loads((a.collection/'results.json').read_text())['rows'];assert len(source)==16
sources=[];loaded=[]
for row in source:
 path=(a.collection/row['trace']).resolve();assert path.is_relative_to(a.collection.resolve())
 with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in ['state','action','context','next_state']}
 n=len(data['state']);assert np.array_equal(data['next_state'][:-1],data['state'][1:])
 for key,width in [('state',73),('action',30),('context',34),('next_state',73)]:assert data[key].shape==(n,width) and np.isfinite(data[key]).all()
 sources.append(dict(file=row['trace'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),frames=n,split=row['split']))
 loaded.append((row,data))
plan=dict(phase=a.phase,smoke=a.smoke,source_files=sources,physics_contract=PHYSICS_CONTRACT,asset_sha256=hashlib.sha256(Path(os.environ['TW_SUPERMIMIC_MJCF']).read_bytes()).hexdigest(),num_envs=1,dt=.02,
 selection='First20/every50/final union for cold; all saved frames and original trace order for warm. Smoke first2 traces/first3 warm or0/middle/final cold only.',native_replay_atol=1e-5,state_write_atol=2e-6,context_atol=1e-7,scope='Native dynamic evidence only, no data replacement, model fitting, gate tuning or policy promotion. Warm cumulative differences must be interpreted together with input drift.')
(a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(a.output/'status').write_text('initializing\n')
cfg=tw_supermimic_env_cfg([manifest['motions'][0]['file']],num_envs=1,play=True);cfg.events={};cfg.seed=42;cfg.auto_reset=False
cfg.commands['motion'].pose_range={};cfg.commands['motion'].velocity_range={};cfg.commands['motion'].joint_position_range=(0.,0.)
env=ManagerBasedRlEnv(cfg,device='cuda:0');enable_tw_velocity_limits(env)
robot=env.scene['robot'];cmd=env.command_manager.get_term('motion');term=env.action_manager.get_term('joint_pos');ids=torch.tensor([0],device='cuda:0')
spec=SuperMimicStateSpec.from_env(env,cmd.cfg.body_names)
extract=StateExtractor(env,spec,torso_body_name='chest',left_foot_geom_pattern=r'^left_foot_collision(_\d+)?$',right_foot_geom_pattern=r'^right_foot_collision(_\d+)?$')
mapping=json.loads((a.collection/'control_mapping.json').read_text())
assert mapping['state_joint_names']==list(robot.joint_names) and mapping['action_joint_names']==list(term.target_names)
for key in ['offset','scale']:np.testing.assert_allclose(getattr(term,'_'+key).cpu().numpy(),np.array(mapping[key]),rtol=0,atol=1e-7)
assert abs(env.step_dt-.02)<1e-9
caps=torch.tensor([LIMITS[n] for n in robot.joint_names],device='cuda:0');reports=[];cleared=set();pending=None

def reset():
 env.reset();cmd.reset_to_frame(ids,0,0);env.sim.forward()

def clear_history():
 for name in ['qacc_warmstart','qacc','qfrc_applied','xfrc_applied','ctrl','act']:
  value=getattr(env.sim.data,name,None)
  if isinstance(value,torch.Tensor):value.zero_();cleared.add(name)

def save_pending(filename,records):
 if records and records['index']:
  np.savez_compressed(a.output/filename,**{k:np.stack(v) for k,v in records.items()})

try:
 (a.output/'status').write_text('running\n')
 for trace_index,(row,data) in enumerate(loaded[:2] if a.smoke else loaded):
  n=len(data['state'])
  selection=list(range(min(n,3))) if a.smoke and a.phase=='warm' else (sorted({0,n//2,n-1}) if a.smoke else (list(range(n)) if a.phase=='warm' else sorted(set(range(min(20,n)))|set(range(0,n,50))|{n-1})))
  records={k:[] for k in ['index','restored_state','context','actual_next_state','recorded_next_state','input_error','context_error','terminated','truncated','root_angular_error']}
  filename=Path(row['trace']).stem+'_'+a.phase+'.npz';pending=(filename,records)
  if a.phase=='warm':reset()
  for index in selection:
   desired=torch.tensor(data['state'][index:index+1],device='cuda:0');action=torch.tensor(data['action'][index:index+1],device='cuda:0')
   context=torch.tensor(data['context'][index:index+1],device='cuda:0')
   if a.phase=='cold':
    reset()
    robot.write_joint_state_to_sim(desired[:,13:43],desired[:,43:73])
    robot.write_root_link_pose_to_sim(torch.cat([desired[:,:3]+env.scene.env_origins,desired[:,3:7]],dim=1))
    robot.write_root_link_velocity_to_sim(desired[:,7:13]);env.sim.forward();clear_history()
    env.episode_length_buf[:]=index
   state=extract.state().clone();ctx=extract.context().clone()
   input_error=float((state-desired).abs().max());context_error=float((ctx-context).abs().max())
   if a.phase=='cold':assert input_error<=plan['state_write_atol'],input_error
   assert context_error<=plan['context_atol'],context_error
   out=env.step(action);actual=extract.state().clone();target=torch.tensor(data['next_state'][index:index+1],device='cuda:0')
   assert torch.isfinite(actual).all() and bool((robot.data.joint_vel.abs()<=caps+1e-5).all())
   angular_error=float((actual[:,10:13]-target[:,10:13]).square().mean().sqrt())
   values=dict(index=index,restored_state=state[0].cpu().numpy(),context=ctx[0].cpu().numpy(),actual_next_state=actual[0].cpu().numpy(),recorded_next_state=data['next_state'][index],input_error=input_error,context_error=context_error,terminated=bool(out[2][0]),truncated=bool(out[3][0]),root_angular_error=angular_error)
   for key,v in values.items():records[key].append(v)
  save_pending(filename,records);pending=None
  next_error=np.abs(np.stack(records['actual_next_state'])-np.stack(records['recorded_next_state']))
  report=dict(file=row['trace'],split=row['split'],source_frames=n,replayed_frames=len(selection),trace=filename,max_input_error=max(records['input_error']),max_next_state_error=float(next_error.max()),max_root_angular_rmse=max(records['root_angular_error']),original_termination_reasons=row['termination_reasons'],terminated_count=sum(records['terminated']),truncated_count=sum(records['truncated']))
  reports.append(report);print(json.dumps(report),flush=True);(a.output/'partial.json').write_text(json.dumps(reports,indent=2)+'\n')
 result=dict(phase=a.phase,smoke=a.smoke,rows=reports,cleared_transient_fields=sorted(cleared),native_replay_atol=plan['native_replay_atol'],all_next_states_within_native_atol=all(x['max_next_state_error']<=plan['native_replay_atol'] for x in reports),scope=plan['scope'])
 (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'status').write_text('completed_pending_review\n')
except BaseException:
 if pending:save_pending('interrupted_'+pending[0],pending[1])
 (a.output/'status').write_text('failed\n');raise
finally:env.close();disable_tw_velocity_limits(env)
