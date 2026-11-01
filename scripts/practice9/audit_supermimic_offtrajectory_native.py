"""Frozen fit-only off-trajectory native comparisons; no policy fitting."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np,torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import enable_tw_velocity_limits,disable_tw_velocity_limits,LIMITS,PHYSICS_CONTRACT
from supermimic.algo.state import SuperMimicStateSpec,StateExtractor
from supermimic_geometry_model import load_geometry_artifact
from supermimic_motion_windows import load_motion_fit_windows
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
r=a.root;a.collection=r/'motion_pd_collection_cache_v2/full'
manifest=json.loads((a.collection.parent/'plan.json').read_text())
mapping=json.loads((a.collection/'control_mapping.json').read_text())
_,source_meta=load_motion_fit_windows(a.collection,a.collection.parent/'full_audit.json',mapping,window=1)
rows=[x for x in json.loads((a.collection/'results.json').read_text())['rows'] if x['split']=='fit'];assert len(rows)==16
models={};queries=[];sources=[]
for arm in ['control','candidate']:
 path=r/'motion_h1_anchor_fit_v1'/arm/'fit/world_model_calibrated.pt'
 c=torch.load(path,map_location='cpu',weights_only=False);m,predict=load_geometry_artifact(c);m.eval()
 for v in m.parameters():v.requires_grad_(False)
 info=json.loads((a.collection/'input.json').read_text())
 assert info['physics']==PHYSICS_CONTRACT==c['cfg']['extra']['tw_physics_contract']
 assert info['asset_sha256']==c['cfg']['extra']['tw_mjcf_sha256']==hashlib.sha256(Path(os.environ['TW_SUPERMIMIC_MJCF']).read_bytes()).hexdigest()
 assert mapping['state_joint_names']==c['cfg']['extra']['tw_state_joint_names'] and mapping['action_joint_names']==c['cfg']['extra']['tw_action_joint_names']
 models[arm]=(c,m,predict)
 sources.append(dict(arm=arm,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 with torch.no_grad():
  for row in rows[:1] if a.smoke else rows:
   with np.load(a.collection/row['trace'],allow_pickle=False) as z:data={k:torch.tensor(z[k],device='cuda:0') for k in ['state','action','context','next_state']}
   n=len(data['state']);selected=sorted({0,7,min(31,n-1)});pred=data['state'][:1]
   for j in range(max(selected)+1):
    if j in selected:
     mixed=pred.clone();mixed[:,7:13]=data['state'][j:j+1,7:13]
     for mode,state in [('true',data['state'][j:j+1]),('free',pred),('root_velocity_true',mixed)]:
      queries.append(dict(arm=arm,mode=mode,row=row,index=j,state=state.clone(),action=data['action'][j:j+1],context=data['context'][j:j+1],archived=data['next_state'][j:j+1]))
    pred=predict(pred,data['action'][j:j+1],data['context'][j:j+1])
    assert torch.isfinite(pred).all()
plan=dict(smoke=a.smoke,query_count=len(queries),fit_input=source_meta,models=sources,state_write_atol=2e-6,native_atol=1e-5,physics_contract=PHYSICS_CONTRACT,asset_sha256=hashlib.sha256(Path(os.environ['TW_SUPERMIMIC_MJCF']).read_bytes()).hexdigest(),selection='All16 fit episodes, first-start free rollouts, indices0/7/min31,n-1; smoke first episode. Two cold restores per query; hybrid probes never fed back.',scope='Off-trajectory finite-query physics diagnostic only; hybrid states may be inconsistent. No optimization or acceptance.')
(a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(a.output/'status').write_text('initializing\n')
cfg=tw_supermimic_env_cfg([x['file'] for x in manifest['motions']],num_envs=1,play=True);cfg.events={};cfg.seed=42;cfg.auto_reset=False
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

clip_ids={x['id']:i for i,x in enumerate(manifest['motions'])}
def reset(row,index=0):
 env.reset();cmd.reset_to_frame(ids,clip_ids[row['id']],row['frame']+index);env.sim.forward()

def clear_history():
 for name in ['qacc_warmstart','qacc','qfrc_applied','xfrc_applied','ctrl','act']:
  value=getattr(env.sim.data,name,None)
  if isinstance(value,torch.Tensor):value.zero_();cleared.add(name)

records=[]
try:
 (a.output/'status').write_text('running\n')
 for query in queries:
  desired=query['state'];outputs=[];restored=[];predictions=[];contexts=[];terminals=[]
  for repeat in range(2):
   reset(query['row'],query['index'])
   robot.write_joint_state_to_sim(desired[:,13:43],desired[:,43:73])
   robot.write_root_link_pose_to_sim(torch.cat([desired[:,:3]+env.scene.env_origins,desired[:,3:7]],1))
   robot.write_root_link_velocity_to_sim(desired[:,7:13]);env.sim.forward();clear_history()
   env.episode_length_buf[:]=query['index'];cmd.update_relative_body_poses()
   state=extract.state().clone();ctx=extract.context().clone()
   assert float((ctx-query['context']).abs().max())<=1e-7
   with torch.no_grad():pred=models[query['arm']][2](state,query['action'],ctx)
   result=env.step(query['action']);actual=extract.state().clone()
   assert torch.isfinite(actual).all() and torch.isfinite(pred).all()
   assert bool((robot.data.joint_vel.abs()<=caps+1e-5).all())
   restored.append(state[0].cpu().numpy());contexts.append(ctx[0].cpu().numpy());predictions.append(pred[0].cpu().numpy());outputs.append(actual[0].cpu().numpy());terminals.append([bool(result[2][0]),bool(result[3][0])])
  record=dict(arm=query['arm'],mode=query['mode'],trace=query['row']['trace'],index=query['index'],write_error=float(np.abs(np.stack(restored)-desired.cpu().numpy()).max()),repeat_error=float(np.abs(outputs[0]-outputs[1]).max()),archived_error=float(np.abs(outputs[0]-query['archived'][0].cpu().numpy()).max()) if query['mode']=='true' else None,terminals=terminals)
  filename='query_%03d.npz'%len(records)
  np.savez_compressed(a.output/filename,desired=desired.cpu().numpy(),restored=np.stack(restored),context=np.stack(contexts),action=query['action'].cpu().numpy(),native=np.stack(outputs),model=np.stack(predictions),archived=query['archived'].cpu().numpy())
  record['file']=filename;record['sha256']=hashlib.sha256((a.output/filename).read_bytes()).hexdigest()
  records.append(record);(a.output/'partial.json').write_text(json.dumps(records,indent=2)+'\n')
  if len(records)%18==0:print(json.dumps(dict(completed=len(records),total=len(queries))),flush=True)
 for c,m,_ in models.values():
  for k,v in m.state_dict().items():assert torch.equal(v.cpu(),c['world_model'][k]),k
 result=dict(smoke=a.smoke,queries=len(records),rows=records,weights_unchanged=True,validation_arrays_loaded=False,write_failures=sum(x['write_error']>2e-6 for x in records),repeat_failures=sum(x['repeat_error']>1e-5 for x in records),true_replay_failures=sum(x['archived_error']>1e-5 for x in records if x['archived_error'] is not None),cleared_transient_fields=sorted(cleared),scope=plan['scope'])
 (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'status').write_text('completed_pending_review\n')
 print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
except BaseException:
 (a.output/'status').write_text('failed\n');raise
finally:env.close();disable_tw_velocity_limits(env)
