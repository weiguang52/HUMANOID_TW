"""Opt-in diagnostic only; unchanged production defaults and intrinsic actor."""
import json,numpy as np,torch,time
from pathlib import Path
from issue21_common import ROOT,OLD,sha,save
from issue20_runtime import configure as configure20,snapshot,reflected
from issue17_symmetry import mapping_for_env

def configure(cfg,mode):
 cfg.issue17_integer_clock=True;configure20(cfg,0,str(OLD/'strict_symmetric.usd'))
 if mode=='air':
  cfg.scene.robot.spawn.articulation_props.fix_root_link=True
  cfg.scene.robot.init_state.pos=(0.,0.,2.5)
  cfg.terminations.base_contact=None

def reset(base,command,phase,air=False):
 base.cfg.issue20_phase_offset=phase;ordinal=5 if command=='stand' else 3
 parent=OLD/'evaluation/symmetric/moments'/('symmetric_s%d_eval1_k0'%base._issue21_training_seed)/(command+'_initial_state.npz');z=np.load(parent);n=base.num_envs
 if not air:base.scene.env_origins[:]=torch.as_tensor(z['env_origins'][:n],device=base.device)
 base.reset(seed=1000+ordinal);robot=base.scene['robot'];ids=torch.arange(n,device=base.device);src=snapshot(base)
 for key in ['root','joint_pos','joint_vel']:src[key]=torch.as_tensor(z[key][:n],device=base.device).clone()
 if air:
  src['root'][:,:3]=base.scene.env_origins+src['root'].new_tensor([0,0,2.5]);src['root'][:,3:7]=src['root'].new_tensor([1,0,0,0]);src['root'][:,7:]=0
 expected=reflected(base,src,ids) if phase==10 else src
 robot.write_root_state_to_sim(expected['root']);robot.write_joint_state_to_sim(expected['joint_pos'],expected['joint_vel']);robot.set_joint_position_target(expected['joint_pos']);base.scene.write_data_to_sim();actual=snapshot(base)
 for key in ['root','joint_pos','joint_vel']:assert torch.equal(actual[key],expected[key]),key
 assert torch.count_nonzero(base.action_manager.action)==0
 base.command_manager.get_term('base_velocity').vel_command_b[:]=torch.tensor([0 if command=='stand' else .5,0,0],device=base.device);base.command_manager.get_term('base_velocity').is_standing_env[:]=False
 base._issue21_parent_initial={'path':str(parent),'sha256':sha(parent),'count':n,'ground_world_root_and_joints_exact_parent':not air,'air_joint_initial_states_exact_parent':air,'single_env_matches_env0_of16':True,'fresh_process_no_prior_trajectory':True}
 return actual

def capture(base):
 r=base.scene['robot'];s=base.scene['contact_forces'];root=r.data.root_state_w.clone();root[:,:3]-=base.scene.env_origins
 return {'root':root,'joint_pos':r.data.joint_pos.clone(),'joint_vel':r.data.joint_vel.clone(),'applied_torque':r.root_physx_view.get_dof_projected_joint_forces().clone(),'estimated_pd_torque':r.data.applied_torque.clone(),'explicit_actuation_force':r.root_physx_view.get_dof_actuation_forces().clone(),'joint_target':r.data.joint_pos_target.clone(),'contact':s.data.net_forces_w.clone()}

def cpu(x):return x.detach().cpu().numpy().copy()
def digest_actor(runner):
 import hashlib
 h=hashlib.sha256()
 for k,v in runner.alg.actor.state_dict().items():h.update(k.encode());h.update(cpu(v).tobytes())
 return h.hexdigest()
def run(env,policy,runner,args,checkpoint):
 base=env.unwrapped;r=base.scene['robot'];m=mapping_for_env(base);n=base.num_envs;mode=args.issue21_mode;ts=args.training_seed;base._issue21_training_seed=ts
 assert not base.cfg.observations.policy.enable_corruption and base.cfg.events.push_robot is None
 assert abs(base.step_dt-.02)<1e-9
 out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);before=digest_actor(runner);records=[]
 metadata={'mode':mode,'training_seed':ts,'num_envs':n,'steps':{'ground':1000,'air':200,'perturb':1000}[mode],'checkpoint':checkpoint,'checkpoint_sha256':sha(checkpoint),'strict_usd_sha256':sha(OLD/'strict_symmetric.usd'),'body_names':r.body_names,'contact_body_names':base.scene['contact_forces'].body_names,'joint_names':r.joint_names,'step_dt':base.step_dt,'policy_terms':base.observation_manager.active_terms['policy'],'policy_dims':base.observation_manager.group_obs_term_dim['policy'],'root_state_fields':['x','y','z','qw','qx','qy','qz','vx','vy','vz','wx','wy','wz'],'torque_semantics':{'applied_torque':'PhysX get_dof_projected_joint_forces: measured active component of incoming joint force in DOF direction; not claimed to be a separately exposed pure implicit drive torque','estimated_pd_torque':'Isaac Lab approximate implicit PD torque, not actual solver drive torque','explicit_actuation_force':'PhysX explicit actuation input; implicit drives may yield zero','joint_target':'Isaac Lab position target sent to simulation'},'training_updates':0,'air_fixed_root':mode=='air','clean':True}
 if mode=='perturb':
  command='forward_05'
  for epsilon in [args.issue21_epsilon]:
   reset(base,command,0);yaws=[];dones=[];vels=[];poses=[]
   for t in range(1000):
    if t<10 and epsilon!=0:
     v=r.data.root_vel_w.clone();v[:,5]=epsilon;r.write_root_velocity_to_sim(v)
    base.command_manager.get_term('base_velocity').vel_command_b[:]=r.data.root_pos_w.new_tensor([.5,0,0]);obs=base.observation_manager.compute()
    with torch.no_grad():a=policy(obs);_,_,done,_=env.step(a)
    q=r.data.root_quat_w;yaws.append(cpu(torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))));dones.append(cpu(base.reset_terminated));vels.append(cpu(r.data.root_ang_vel_w[:,2]));poses.append(cpu(torch.cat([r.data.root_state_w[0,:7]-torch.cat([base.scene.env_origins[0],q.new_zeros(4)]),r.data.joint_pos[0]])))
   name='epsilon_%+.2f'%epsilon;path=out.parent/(out.stem+'_'+name+'.npz');np.savez_compressed(path,yaw=np.stack(yaws),terminated=np.stack(dones),world_wz=np.stack(vels),replay_env0=np.stack(poses));records.append({'epsilon':epsilon,'path':str(path),'sha256':sha(path)});print('ISSUE21_PERTURB_DONE',ts,epsilon,flush=True)
 else:
  for command in [args.issue21_command]:
   for kind in [args.issue21_kind]:
    for phase in [args.issue21_phase]:
     reset(base,command,phase,air=mode=='air');arrays={k:[] for k in ['obs','action','counterfactual_action_error','root','joint_pos','joint_vel','applied_torque','estimated_pd_torque','explicit_actuation_force','joint_target','contact','terminated']};initial=capture(base);arrays_initial={k:cpu(v) for k,v in initial.items()};steps=1000 if kind=='closed' else 200
     if kind=='open':
      source=ROOT/'evaluation'/('ground_s%d_n%d_forward_05_closed_k0.npz'%(ts,n))
      source=source.with_name(source.name.replace('forward_05',command));actions=np.load(source)['action'][:200]
     for t in range(steps):
      base.command_manager.get_term('base_velocity').vel_command_b[:]=r.data.root_pos_w.new_tensor([0 if command=='stand' else .5,0,0]);obs=base.observation_manager.compute()
      with torch.no_grad():
       raw=policy(obs);counter=obs.clone() if hasattr(obs,'clone') else dict(obs);counter['policy']=m.actor(obs['policy']);counter_error=policy(counter)-m.action(raw)
       a=raw if kind=='closed' else torch.as_tensor(actions[t],device=base.device)
       if kind=='open' and phase==10:a=m.action(a)
       arrays['obs'].append(cpu(obs['policy']));arrays['action'].append(cpu(a));arrays['counterfactual_action_error'].append(cpu(counter_error))
       _,_,done,_=env.step(a)
      for k,v in capture(base).items():arrays[k].append(cpu(v))
      arrays['terminated'].append(cpu(base.reset_terminated))
     path=ROOT/'evaluation'/('%s_s%d_n%d_%s_%s_k%d.npz'%(mode,ts,n,command,kind,phase));np.savez_compressed(path,**{k:np.stack(v) for k,v in arrays.items()},**{'initial_'+k:v for k,v in arrays_initial.items()});records.append({'command':command,'kind':kind,'phase':phase,'steps':steps,'path':str(path),'sha256':sha(path)});print('ISSUE21_TRACE_DONE',mode,ts,n,command,kind,phase,flush=True)
 metadata.update(cold_start=True,parent_initial=base._issue21_parent_initial,records=records,model_unchanged=digest_actor(runner)==before);assert metadata['model_unchanged'];out.write_text(json.dumps(metadata,indent=2)+'\n');print('ISSUE21_JOB_DONE',mode,ts,n,flush=True)
