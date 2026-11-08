"""Open-loop leg-lift test: can the TW legs physically lift a foot with the current actuators?

Each env gets a different (hip_pitch, knee) offset for the LEFT leg relative to the default pose; all other joints
stay at default. Targets are held for ~1.5 s. Reports foot height difference (left - right), achieved joint
angles, applied torque vs limit, and whether the robot stayed up.
"""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import itertools
import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

HIP = [-0.5, 0.0, 0.5]
KNEE = [-0.9, -0.5, 0.5, 0.9]
trials = list(itertools.product(HIP, KNEE))

cfg = parse_env_cfg("Unitree-TW-30dof-Velocity", device="cuda:0", num_envs=len(trials))
cfg.events.push_robot = None
cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
cfg.terminations.base_height = None
cfg.terminations.base_contact = None
cfg.commands.base_velocity.rel_standing_envs = 1.0
env = gym.make("Unitree-TW-30dof-Velocity", cfg=cfg)
env.reset()
u = env.unwrapped
robot = u.scene["robot"]
term = u.action_manager.get_term("JointPositionAction")
names = list(term._joint_names)
scale = term._scale if term._scale.ndim == 1 else term._scale[0]
print("ACTION_JOINTS", names, flush=True)
jn = robot.joint_names
hip_i, knee_i = names.index("left_hip_pitch_joint"), names.index("left_knee_pitch_joint")
lim = robot.data.soft_joint_pos_limits[0]
rj_hip, rj_knee = jn.index("left_hip_pitch_joint"), jn.index("left_knee_pitch_joint")
print("LIMITS left_hip_pitch", lim[rj_hip].tolist(), "left_knee_pitch", lim[rj_knee].tolist(), flush=True)
print("EFFORT_LIMIT hip/knee", robot.data.joint_effort_limits[0, rj_hip].item(), robot.data.joint_effort_limits[0, rj_knee].item(), flush=True)

act = torch.zeros(len(trials), len(names), device="cuda:0")
for e, (h, k) in enumerate(trials):
    act[e, hip_i] = h / scale[hip_i]
    act[e, knee_i] = k / scale[knee_i]
lf, rf = robot.body_names.index("left_foot"), robot.body_names.index("right_foot")
best = torch.full((len(trials),), -1.0, device="cuda:0")
minroot = torch.full((len(trials),), 9.0, device="cuda:0")
for t in range(75):
    u.step(act)
    lift = robot.data.body_pos_w[:, lf, 2] - robot.data.body_pos_w[:, rf, 2]
    best = torch.maximum(best, lift)
    minroot = torch.minimum(minroot, robot.data.root_pos_w[:, 2])
q = robot.data.joint_pos
tq = robot.data.applied_torque
for e, (h, k) in enumerate(trials):
    print(f"LIFT hip={h:+.1f} knee={k:+.1f} | max_lift={best[e].item()*1000:6.1f} mm | min_root_z={minroot[e].item():.3f} "
          f"| final hip={q[e, rj_hip].item():+.2f} knee={q[e, rj_knee].item():+.2f} "
          f"| torque hip={tq[e, rj_hip].item():+.2f} knee={tq[e, rj_knee].item():+.2f}", flush=True)
env.close()
app.close()