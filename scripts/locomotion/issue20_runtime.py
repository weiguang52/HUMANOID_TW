"""Evaluation-only phase offset and paired reflected reset state. No RNG draws."""
import torch,json,hashlib
from pathlib import Path
from issue17_symmetry import mapping_for_env
from issue13_tracking import rng_state,rng_same

def phase(env,period):
 from unitree_rl_lab.tasks.locomotion.mdp.observations import gait_phase
 k=getattr(env.cfg,'issue20_phase_offset',0)
 if k==0:return gait_phase(env,period)
 assert k==10 and round(period/env.step_dt)==20
 idx=(env.episode_length_buf+k)%20;q=idx.to(torch.float32)/20
 y=torch.zeros(env.num_envs,2,device=env.device);y[:,0]=torch.sin(q*torch.pi*2);y[:,1]=torch.cos(q*torch.pi*2);return y

def configure(cfg,k,usd):
 assert k in [0,10];cfg.issue20_phase_offset=k
 # Keep an explicit zero-offset branch that delegates the original function unchanged.
 cfg.observations.policy.gait_phase.func=phase;cfg.observations.critic.gait_phase.func=phase
 if usd:
  import isaaclab.sim as sim_utils,copy
  old=cfg.scene.robot.spawn;new=sim_utils.UsdFileCfg(usd_path=usd)
  for key in vars(new):
   if key not in ['func','usd_path'] and hasattr(old,key):setattr(new,key,copy.deepcopy(getattr(old,key)))
  cfg.scene.robot.spawn=new

def snapshot(base):
 robot=base.scene['robot'];return {'root':robot.data.root_state_w.detach().clone(),'joint_pos':robot.data.joint_pos.detach().clone(),'joint_vel':robot.data.joint_vel.detach().clone(),'last_action':base.action_manager.action.detach().clone()}

def reflected(base,state,ids):
 m=mapping_for_env(base).spatial;out={k:v[ids].clone() for k,v in state.items()};r=out['root'];o=base.scene.env_origins[ids];r[:,:3]=(r[:,:3]-o)*r.new_tensor([1,-1,1])+o;r[:,3:7]*=r.new_tensor([1,-1,1,-1]);r[:,7:10]*=r.new_tensor([1,-1,1]);r[:,10:13]*=r.new_tensor([-1,1,-1])
 for name in ['joint_pos','joint_vel']:out[name]=out[name][:,m.perm]*m.sign
 out['last_action']=m.action(out['last_action']);return out

def install_resets(base,k,folder):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);original=base._reset_idx;robot=base.scene['robot'];audit=[]
 def reset(ids):
  original(ids)
  if k!=10:return
  before=rng_state(base.device);source=snapshot(base);expected=reflected(base,source,ids)
  robot.write_root_state_to_sim(expected['root'],env_ids=ids);robot.write_joint_state_to_sim(expected['joint_pos'],expected['joint_vel'],env_ids=ids)
  # Standard reset actions and last actions are exactly zero; no artificial history.
  assert torch.count_nonzero(expected['last_action'])==0
  robot.set_joint_position_target(expected['joint_pos'],env_ids=ids);base.scene.write_data_to_sim();actual=snapshot(base)
  errors={key:float((actual[key][ids]-value).abs().max()) for key,value in expected.items()};assert max(errors.values())==0,errors
  rng_same(before,rng_state(base.device));audit.append({'count':len(ids),'max_errors':errors,'RNG_unchanged':True})
 base._reset_idx=reset;base._issue20_reset_audit=audit
 base._issue20_initial_snapshot=lambda:snapshot(base)
 return {'phase_offset':k,'initial_state_reflected':k==10,'noise':'clean protocol corruption disabled; no noise samples drawn or added','root_quaternion_M':'w,-x,y,-z','root_linear_M':'x,-y,z','root_angular_M':'-x,y,-z','clock_changed_only_in_gait_observation':True}
