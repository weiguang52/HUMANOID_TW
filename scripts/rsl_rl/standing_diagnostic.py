"""Opt-in controlled diagnostic only; never enabled in training by default."""
import torch

def configure(cfg,physics_dt):
 if physics_dt not in (.005,.0025):raise ValueError('Audited diagnostic steps only')
 cfg.sim.dt=physics_dt;cfg.decimation=round(.02/physics_dt);cfg.sim.render_interval=cfg.decimation
 cfg.commands.motion.pose_range={};cfg.commands.motion.velocity_range={};cfg.commands.motion.joint_position_range=(0.,0.)
 cfg.observations.policy.enable_corruption=False
 cfg.viewer.origin_type="world"
 cfg.events.add_joint_default_pos=None;cfg.events.base_com=None
 cfg.events.physics_material.params.update(static_friction_range=(1.,1.),dynamic_friction_range=(1.,1.),restitution_range=(0.,0.),num_buckets=1)
 # Arm tracking failure is not a physical fall criterion in the PD ablation.
 cfg.terminations.ee_body_pos=None

def reference_actions(env,command):
 action=env.action_manager.get_term('JointPositionAction')
 ids=action._joint_ids
 names=action._asset.joint_names if isinstance(ids,slice) else [action._asset.joint_names[i] for i in ids]
 cnames=[command.robot.joint_names[i] for i in command.joint_indexes]
 target=command.joint_pos[:,[cnames.index(n) for n in names]]
 scale=torch.as_tensor(action._scale,device=target.device)
 if (scale==0).any():raise ValueError('Zero action scale')
 return (target-action._offset)/scale
