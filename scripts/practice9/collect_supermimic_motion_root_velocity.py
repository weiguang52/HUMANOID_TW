"""Native motion-root-velocity paired branches; no model fitting."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np,torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import enable_tw_velocity_limits,disable_tw_velocity_limits,LIMITS,PHYSICS_CONTRACT
from supermimic.algo.state import SuperMimicStateSpec,StateExtractor
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
a.collection=a.root/'motion_pd_collection_cache_v2/full'
manifest=json.loads((a.collection.parent/'plan.json').read_text())
source=json.loads((a.collection/'results.json').read_text())['rows'];assert len(source)==32
info=json.loads((a.collection/'input.json').read_text());assert info['physics']==PHYSICS_CONTRACT
assert info['asset_sha256']==hashlib.sha256(Path(os.environ['TW_SUPERMIMIC_MJCF']).read_bytes()).hexdigest()
plan=dict(contract='motion_root_velocity_branches_v1',smoke=a.smoke,source_files=source,source_plan_sha256=hashlib.sha256((a.collection.parent/'plan.json').read_bytes()).hexdigest(),levels=[1.,2.],base_delta=[.005,.005,.005,.02,.02,.02],assignments=2,num_envs=12,selection='All32 original episodes, indices first/middle/final; smoke first fit episode index0 only',state_write_atol=2e-6,replay_atol=1e-5,asset_sha256=info['asset_sha256'],physics=PHYSICS_CONTRACT,scope='One-step branches only; original terminal flags retained, no optimization or acceptance.')
(a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
cfg=tw_supermimic_env_cfg([x['file'] for x in manifest['motions']],num_envs=12,play=True);cfg.events={};cfg.seed=42;cfg.auto_reset=False
cfg.commands['motion'].pose_range={};cfg.commands['motion'].velocity_range={};cfg.commands['motion'].joint_position_range=(0.,0.)
env=ManagerBasedRlEnv(cfg,device='cuda:0');enable_tw_velocity_limits(env)
robot=env.scene['robot'];cmd=env.command_manager.get_term('motion');term=env.action_manager.get_term('joint_pos');ids=torch.arange(12,device='cuda:0')
spec=SuperMimicStateSpec.from_env(env,cmd.cfg.body_names)
extract=StateExtractor(env,spec,torso_body_name='chest',left_foot_geom_pattern=r'^left_foot_collision(_\d+)?$',right_foot_geom_pattern=r'^right_foot_collision(_\d+)?$')
mapping=json.loads((a.collection/'control_mapping.json').read_text())
assert mapping['state_joint_names']==list(robot.joint_names) and mapping['action_joint_names']==list(term.target_names)
for key in ['offset','scale']:np.testing.assert_allclose(getattr(term,'_'+key).cpu().numpy(),np.broadcast_to(np.array(mapping[key]),getattr(term,'_'+key).shape),rtol=0,atol=1e-7)
assert abs(env.step_dt-.02)<1e-9
caps=torch.tensor([LIMITS[n] for n in robot.joint_names],device='cuda:0');reports=[];cleared=set();pending=None

clip_ids={x['id']:i for i,x in enumerate(manifest['motions'])}
def reset(row,index=0):
 env.reset();cmd.reset_to_frame(ids,clip_ids[row['id']],row['frame']+index);env.sim.forward()

def clear_history():
 for name in ['qacc_warmstart','qacc','qfrc_applied','xfrc_applied','ctrl','act']:
  value=getattr(env.sim.data,name,None)
  if isinstance(value,torch.Tensor):value.zero_();cleared.add(name)

(a.output/'control_mapping.json').write_text((a.collection/'control_mapping.json').read_text())
reports=[]
try:
 (a.output/'status').write_text('running\n')
 selected=[x for x in source if x['split']=='fit'][:1] if a.smoke else source
 for row in selected:
  path=a.collection/row['trace'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
  with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in ['state','action','context']}
  n=len(data['state']);indices=[0] if a.smoke else sorted({0,n//2,n-1})
  for index in indices:
   for level,assignment in [(0.,0),(1.,0),(1.,1),(2.,0),(2.,1)]:
    reset(row,index)
    desired=torch.tensor(data['state'][index:index+1],device='cuda:0').repeat(12,1)
    action=torch.tensor(data['action'][index:index+1],device='cuda:0').repeat(12,1)
    delta=torch.zeros(12,6,device='cuda:0')
    signs=torch.tensor([-1.,1.]*6,device='cuda:0')*(1 if assignment==0 else -1)
    delta[torch.arange(12,device='cuda:0'),torch.arange(6,device='cuda:0').repeat_interleave(2)]=signs*torch.tensor(plan['base_delta'],device='cuda:0').repeat_interleave(2)*level
    desired[:,7:13]+=delta
    robot.write_joint_state_to_sim(desired[:,13:43],desired[:,43:73])
    robot.write_root_link_pose_to_sim(torch.cat([desired[:,:3]+env.scene.env_origins,desired[:,3:7]],1))
    robot.write_root_link_velocity_to_sim(desired[:,7:13]);env.sim.forward();clear_history();env.episode_length_buf[:]=index;cmd.update_relative_body_poses()
    state=extract.state().clone();ctx=extract.context().clone();context_error=float((ctx-torch.tensor(data['context'][index],device='cuda:0')).abs().max())
    assert context_error<=1e-7
    result=env.step(action);actual=extract.state().clone();assert torch.isfinite(actual).all() and bool((robot.data.joint_vel.abs()<=caps+1e-5).all())
    filename='%s_i%d_l%d_a%d.npz'%(Path(row['trace']).stem,index,level,assignment)
    np.savez_compressed(a.output/filename,state=state.cpu().numpy(),desired=desired.cpu().numpy(),delta=delta.cpu().numpy(),action=action.cpu().numpy(),context=ctx.cpu().numpy(),next_state=actual.cpu().numpy(),terminated=result[2].cpu().numpy(),truncated=result[3].cpu().numpy())
    reports.append(dict(file=filename,sha256=hashlib.sha256((a.output/filename).read_bytes()).hexdigest(),source_trace=row['trace'],source_family=row['source_family'],split=row['split'],index=index,level=level,assignment=assignment,count=12,write_error=float((state-desired).abs().max()),context_error=context_error,terminated_count=int(result[2].sum()),truncated_count=int(result[3].sum())))
   (a.output/'partial.json').write_text(json.dumps(reports,indent=2)+'\n')
  print(json.dumps(dict(trace=row['trace'],split=row['split'],batches=len(reports))),flush=True)
 result=dict(smoke=a.smoke,rows=reports,endpoints=len(reports)*12,write_failures=sum(x['write_error']>2e-6 for x in reports),scope=plan['scope'])
 (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'status').write_text('completed_pending_audit\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
except BaseException:
 (a.output/'status').write_text('failed\n');raise
finally:env.close();disable_tw_velocity_limits(env)
