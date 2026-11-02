"""Native labels for fixed fit-only rollout queries; no fitting or filtering."""
import argparse,hashlib,json,os
from pathlib import Path
from dataclasses import asdict
import numpy as np,torch,mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sim.sim_data import TorchArray
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import LIMITS,PHYSICS_CONTRACT,enable_tw_velocity_limits,disable_tw_velocity_limits
from supermimic.algo.state import SuperMimicStateSpec,StateExtractor
from supermimic_native_contacts import native_foot_contacts
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--queries',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
plan=json.loads((a.queries/'query_plan.json').read_text());assert plan['queries']==192 and not plan['validation_arrays_loaded'] and not plan['training_approved']
qfile=a.queries/'queries.npz';assert hashlib.sha256(qfile.read_bytes()).hexdigest()==plan['query_sha256']
with np.load(qfile,allow_pickle=False) as z:queries={k:z[k].copy() for k in z.files}
manifest=json.loads(Path(plan['manifest']).read_text());assert len(manifest['motions'])==8
collection=Path(plan['collection']);info=json.loads((collection/'input.json').read_text())
asset=Path(os.environ['TW_SUPERMIMIC_MJCF']);assert hashlib.sha256(asset.read_bytes()).hexdigest()==info['asset_sha256']
assert info['physics']==PHYSICS_CONTRACT
cfg=tw_supermimic_env_cfg([x['file'] for x in manifest['motions']],num_envs=1,play=True);cfg.events={};cfg.seed=42;cfg.auto_reset=False
cfg.commands['motion'].pose_range={};cfg.commands['motion'].velocity_range={};cfg.commands['motion'].joint_position_range=(0.,0.)
before=asdict(cfg.sim.mujoco);cfg.sim.mujoco.jacobian='dense';after=asdict(cfg.sim.mujoco);assert [k for k in before if before[k]!=after[k]]==['jacobian']
env=ManagerBasedRlEnv(cfg,device='cuda:0');enable_tw_velocity_limits(env)
assert env.sim.mj_model.opt.jacobian==mujoco.mjtJacobian.mjJAC_DENSE and not env.sim.wp_model.is_sparse
robot=env.scene['robot'];cmd=env.command_manager.get_term('motion');term=env.action_manager.get_term('joint_pos');ids=torch.tensor([0],device='cuda:0')
spec=SuperMimicStateSpec.from_env(env,cmd.cfg.body_names);extract=StateExtractor(env,spec,torso_body_name='chest',left_foot_geom_pattern=r'^left_foot_collision(_\d+)?$',right_foot_geom_pattern=r'^right_foot_collision(_\d+)?$')
mapping=json.loads((collection/'control_mapping.json').read_text());assert mapping['state_joint_names']==list(robot.joint_names) and mapping['action_joint_names']==list(term.target_names)
for key in ['offset','scale']:np.testing.assert_allclose(getattr(term,'_'+key).cpu().numpy(),mapping[key],atol=1e-7,rtol=0)
caps=torch.tensor([LIMITS[n] for n in robot.joint_names],device='cuda:0');assert env.step_dt==.02
clip_ids={x['id']:i for i,x in enumerate(manifest['motions'])};ranges=[]
for name in robot.joint_names:
 jid=mujoco.mj_name2id(env.sim.mj_model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+name);assert jid>=0
 ranges.append(env.sim.mj_model.jnt_range[jid].copy())
ranges=np.array(ranges);records={k:[] for k in ['query','warm_source','repeat','state','action','context','next_state','native_qpos','native_qvel','native_warmstart','terminated','truncated','state_write_error','context_error','archived_next_error','input_speed_excess','input_joint_range_excess']};contacts=[]
selected=plan['rows'][:4] if a.smoke else plan['rows'];assert all(x['split']=='fit' for x in selected)
metadata=dict(query_plan_sha256=hashlib.sha256((a.queries/'query_plan.json').read_bytes()).hexdigest(),queries_sha256=plan['query_sha256'],asset_sha256=info['asset_sha256'],physics=PHYSICS_CONTRACT,mujoco_before=before,mujoco_after=after,runtime_dense=True,smoke=a.smoke,planned_records=len(selected)*4,training_approved=False,promotion_approved=False,scope='All fixed fit queries, source/zero warmstart each repeated twice. Source solver state is not known true for predicted states. No input clamping/filtering, keep failures/terminations. No policy/model fitting.')
(a.output/'plan.json').write_text(json.dumps(metadata,indent=2));(a.output/'status').write_text('running')
def save(name):np.savez_compressed(a.output/name,**{k:np.stack(v) for k,v in records.items()})
try:
 for repeat in range(2):
  for warm_source in [True,False]:
   for row in selected:
    i=row['query'];desired=torch.tensor(queries['query_state'][i:i+1],device='cuda:0');action=torch.tensor(queries['action'][i:i+1],device='cuda:0')
    env.reset();cmd.reset_to_frame(ids,clip_ids[row['id']],row['reference_frame']);env.sim.forward()
    robot.write_joint_state_to_sim(desired[:,13:43],desired[:,43:73]);robot.write_root_link_pose_to_sim(torch.cat([desired[:,:3]+env.scene.env_origins,desired[:,3:7]],-1));robot.write_root_link_velocity_to_sim(desired[:,7:13]);env.sim.forward()
    if row['depth']==0:
     env.sim.data.qpos[:]=torch.tensor(queries['source_native_qpos'][i:i+1],device='cuda:0');env.sim.data.qvel[:]=torch.tensor(queries['source_native_qvel'][i:i+1],device='cuda:0');env.sim.forward()
    for name in ['qacc_warmstart','qacc','qfrc_applied','xfrc_applied','ctrl','act']:
     v=getattr(env.sim.data,name,None)
     if isinstance(v,torch.Tensor):v.zero_()
     elif isinstance(v,TorchArray):v[:]=torch.zeros_like(v[:])
    warm=queries['source_native_warmstart'][i:i+1] if warm_source else np.zeros_like(queries['source_native_warmstart'][i:i+1]);env.sim.data.qacc_warmstart[:]=torch.tensor(warm,device='cuda:0')
    env.episode_length_buf[:]=row['index'];cmd.update_relative_body_poses()
    state=extract.state().clone();context=extract.context().clone();qpos=env.sim.data.qpos[0].cpu().numpy().copy();qvel=env.sim.data.qvel[0].cpu().numpy().copy()
    input_error=float((state-desired).abs().max());context_error=float(np.max(np.abs(context[0].cpu().numpy()-queries['context'][i])))
    feet_before=native_foot_contacts(env);assert torch.equal(state,extract.state())
    result=env.step(action);nxt=extract.state().clone();feet_after=native_foot_contacts(env);assert torch.equal(nxt,extract.state())
    assert torch.isfinite(nxt).all() and bool((robot.data.joint_vel.abs()<=caps+1e-5).all())
    qp=desired[0,13:43].cpu().numpy();range_excess=float(np.maximum(ranges[:,0]-qp,qp-ranges[:,1]).clip(0).max())
    values=dict(query=i,warm_source=warm_source,repeat=repeat,state=state[0].cpu().numpy(),action=action[0].cpu().numpy(),context=context[0].cpu().numpy(),next_state=nxt[0].cpu().numpy(),native_qpos=qpos,native_qvel=qvel,native_warmstart=warm[0],terminated=bool(result[2][0]),truncated=bool(result[3][0]),state_write_error=input_error,context_error=context_error,archived_next_error=float(np.max(np.abs(nxt[0].cpu().numpy()-queries['source_next_state'][i]))) if row['depth']==0 else -1.,input_speed_excess=float((desired[:,43:73].abs()-caps).clamp_min(0).max()),input_joint_range_excess=range_excess)
    for k,v in values.items():records[k].append(v)
    contacts.append(dict(query=i,warm_source=warm_source,repeat=repeat,before=feet_before,after=feet_after))
    if len(records['query'])%16==0:save('partial.npz');print(json.dumps(dict(records=len(records['query']),planned=len(selected)*4)),flush=True)
 save('transitions.npz');(a.output/'contacts.json').write_text(json.dumps(contacts));(a.output/'status').write_text('complete_pending_independent_audit')
except BaseException:
 if records['query']:save('interrupted.npz')
 (a.output/'status').write_text('failed_preserve_all');raise
finally:env.close();disable_tw_velocity_limits(env)
