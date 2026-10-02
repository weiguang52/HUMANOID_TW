"""Velocity tracking must preserve intended movement and per-environment cost."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

def _check_velocity_residual():
    import torch
    path=Path(__file__).resolve().parents[2]/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/joint_velocity_cost.py'
    spec=importlib.util.spec_from_file_location('velocity_cost',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    cmd=SimpleNamespace(joint_vel=torch.tensor([[2.,-3.],[1.,1.]]),robot_joint_vel=torch.tensor([[2.,-3.],[3.,-1.]]))
    env=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name:cmd))
    result=mod.motion_joint_velocity_error_l2(env,'motion')
    torch.testing.assert_close(result,torch.tensor([0.,4.]))
    cmd.robot_joint_vel=cmd.joint_vel+torch.tensor([[1.,-1.],[-1.,1.]])
    torch.testing.assert_close(mod.motion_joint_velocity_error_l2(env,'motion'),torch.ones(2))


def test_velocity_residual_in_reference_joint_order():
    import subprocess, sys
    subprocess.run([sys.executable, str(Path(__file__).resolve())], check=True)

if __name__ == "__main__":
    _check_velocity_residual()
