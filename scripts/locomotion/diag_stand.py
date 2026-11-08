"""Diagnostic: zero-action rollout of the TW task, print base height and key joint states."""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

cfg = parse_env_cfg("Unitree-TW-30dof-Velocity", device="cuda:0", num_envs=4)
# no randomization / pushes / terminations so we can watch the raw behaviour
cfg.events.push_robot = None
cfg.terminations.base_height = None
cfg.terminations.base_contact = None
env = gym.make("Unitree-TW-30dof-Velocity", cfg=cfg)
env.reset()
robot = env.unwrapped.scene["robot"]
print("BODIES", robot.body_names)
print("ROOT_BODY_IDX0 z", robot.data.root_pos_w[0].tolist())
print("DEFAULT_JOINT_POS", dict(zip(robot.joint_names, [round(v, 3) for v in robot.data.default_joint_pos[0].tolist()])))
for name in ("left_foot", "right_foot"):
    i = robot.body_names.index(name)
    print("FOOT", name, "z", robot.data.body_pos_w[0, i, 2].item())
zero = torch.zeros(env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim, device="cuda:0")
for t in range(60):
    env.step(zero)
    if t % 5 == 0 or t < 3:
        z = robot.data.root_pos_w[0, 2].item()
        fz = [robot.data.body_pos_w[0, robot.body_names.index(n), 2].item() for n in ("left_foot", "right_foot")]
        g = robot.data.projected_gravity_b[0].tolist()
        print(f"STEP {t:2d} root_z={z:.3f} feet_z={fz[0]:.3f},{fz[1]:.3f} gravity_b={[round(v,2) for v in g]}")
env.close()
app.close()