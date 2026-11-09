# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import json
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--output", required=True)
parser.add_argument("--experiment_profile", default="default")
parser.add_argument("--mass_randomization", default="legacy", choices=["legacy","bounded"])
parser.add_argument("--mass_audit_output", default="")
parser.add_argument("--issue5_condition", default="legacy", choices=["legacy","clean","push","material","delay","noise","combined"])
parser.add_argument("--command_filter", default="")
parser.add_argument("--paired_resets", action="store_true")
parser.add_argument("--extended_commands", action="store_true")
parser.add_argument("--with_pushes", action="store_true")
parser.add_argument("--camera_view", choices=["front","side"], default="front")
parser.add_argument("--seconds", type=float, default=20.0)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
torch.set_num_threads(4)
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    from unitree_rl_lab.tasks.locomotion.robots.tw.experiments import configure_experiment
    configure_experiment(env_cfg, agent_cfg, args_cli.experiment_profile)
    if not args_cli.with_pushes:
        env_cfg.events.push_robot = None
    from unitree_rl_lab.tasks.locomotion.robots.tw.experiments import configure_issue5_condition
    configure_issue5_condition(env_cfg, args_cli.issue5_condition)
    from unitree_rl_lab.tasks.locomotion.robots.tw.mass_randomization import configure_mass_randomization, mass_distribution_report, AuditedMassRandomization
    configure_mass_randomization(env_cfg, args_cli.mass_randomization)
    if args_cli.mass_audit_output:
        assert env_cfg.events.add_base_mass is not None
        env_cfg.events.add_base_mass.func = AuditedMassRandomization
    env_cfg.episode_length_s = args_cli.seconds + 1.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.e9, 1.e9)
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.curriculum.lin_vel_cmd_levels = None
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.video:
        if env_cfg.scene.num_envs != 1:
            raise ValueError("Video evaluation requires --num_envs 1")
        import isaaclab.sim as sim_utils
        original_spawn = env_cfg.scene.robot.spawn
        env_cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
            usd_path="/root/gpufree-data/datasets/practice9/custom_robot/usd/urdf0711_training_30dof_tw123_yaw_waist_v3_colored.usd",
            activate_contact_sensors=True,
            rigid_props=original_spawn.rigid_props,
            articulation_props=original_spawn.articulation_props,
        )
        env_cfg.viewer.resolution = (960, 720)
        # Verify the precolored asset preserves the exact experiment physics.
        from pxr import Usd
        original=Usd.Stage.Open("/root/gpufree-data/datasets/practice9/custom_robot/usd/urdf0711_training_30dof_tw123_yaw_waist_v3.usd")
        colored=Usd.Stage.Open(env_cfg.scene.robot.spawn.usd_path)
        checked=0
        for prim in original.Traverse():
            other=colored.GetPrimAtPath(prim.GetPath())
            assert other, str(prim.GetPath())
            for attr in prim.GetAttributes():
                assert str(attr.Get())==str(other.GetAttribute(attr.GetName()).Get()),str(attr.GetPath())
                checked+=1
            for rel in prim.GetRelationships():
                if not rel.GetName().startswith("material:binding"):
                    assert rel.GetTargets()==other.GetRelationship(rel.GetName()).GetTargets()
        print(f"Colored USD integrity: {checked} original attributes unchanged",flush=True)
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.mass_audit_output:
        report = mass_distribution_report(env.unwrapped.scene["robot"], env.unwrapped._issue6_requested_masses)
        report.update(seed=env_cfg.seed, mass_randomization=args_cli.mass_randomization,
                      mass_parameters=env_cfg.events.add_base_mass.params["mass_distribution_params"],
                      operation=env_cfg.events.add_base_mass.params["operation"],
                      friction_static=env_cfg.events.physics_material.params["static_friction_range"],
                      friction_dynamic=env_cfg.events.physics_material.params["dynamic_friction_range"])
        Path(args_cli.mass_audit_output).parent.mkdir(parents=True,exist_ok=True)
        Path(args_cli.mass_audit_output).write_text(json.dumps(report,indent=2)+"\n")
        env.close()
        return

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # convert pre-5.0 published checkpoints to the layout expected by rsl-rl >= 5.0 (no-op otherwise)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # One independent, full-duration trial per environment and command.
    # Track any true termination across the whole horizon, rather than resets/step.
    base = env.unwrapped
    cmd_term = base.command_manager.get_term("base_velocity")
    robot = base.scene["robot"]
    feet = [robot.body_names.index(n) for n in ("left_foot", "right_foot")]
    sensor_feet = [base.scene["contact_forces"].body_names.index(n) for n in ("left_foot", "right_foot")]
    leg_joints = [[i for i,n in enumerate(robot.joint_names) if n.startswith(side) and any(k in n for k in ("hip", "knee", "ankle", "foot"))] for side in ("left_", "right_")]
    physics_diagnostics = None
    if args_cli.issue5_condition != "legacy":
        masses = robot.root_physx_view.get_masses()
        base_id = robot.body_names.index("base_link")
        physics_diagnostics = {
            "default_base_mass_min": robot.data.default_mass[:,base_id].min().item(),
            "default_base_mass_max": robot.data.default_mass[:,base_id].max().item(),
            "actual_base_mass_min": masses[:,base_id].min().item(),
            "actual_base_mass_max": masses[:,base_id].max().item(),
            "base_mass_below_1mg": int((masses[:,base_id] < 1.e-3).sum()),
            "nonpositive_body_mass_count": int((masses <= 0).sum()),
        }
        assert physics_diagnostics["nonpositive_body_mass_count"] == 0
    results = []
    commands = [("stand", (0.,0.,0.)), ("forward_01",(.1,0.,0.)),
                           ("forward_02",(.2,0.,0.)), ("forward_03",(.3,0.,0.)),
                           ("yaw_positive",(0.,0.,.3)), ("yaw_negative",(0.,0.,-.3))]
    if args_cli.extended_commands:
        commands += [("forward_05",(.5,0.,0.)),("forward_08",(.8,0.,0.)),("yaw_plus08",(0.,0.,.8)),("yaw_minus08",(0.,0.,-.8))]
    if args_cli.command_filter:
        selected=set(args_cli.command_filter.split(","))
        assert selected <= {x[0] for x in commands}, selected
        commands=[x for x in commands if x[0] in selected]
    for command_index, (label, command) in enumerate(commands):
        if args_cli.paired_resets:
            base.reset(seed=env_cfg.seed * 1000 + command_index)
        else:
            env.reset()
        writer = None
        if args_cli.video:
            import imageio.v2 as imageio
            video_dir=Path(args_cli.output).parent / "videos"
            video_dir.mkdir(parents=True,exist_ok=True)
            writer=imageio.get_writer(str(video_dir / f"{Path(args_cli.output).stem}_{label}_{args_cli.camera_view}.mp4"), fps=round(1/base.step_dt))
        failed = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
        totals = {k: torch.zeros((), device=base.device, dtype=torch.float64) for k in ["xy_error", "yaw_error", "base_height", "joint_acc_rms", "foot_contact_slide", "left_contact_duty", "right_contact_duty", "left_joint_acc_rms", "right_joint_acc_rms", "left_foot_height", "right_foot_height"]}
        count = torch.zeros((),device=base.device,dtype=torch.long)
        min_height = torch.full((),float("inf"),device=base.device)
        target = torch.tensor(command, device=base.device)
        for step in range(round(args_cli.seconds / base.step_dt)):
            cmd_term.vel_command_b[:] = target
            cmd_term.is_standing_env[:] = False
            # Recompute observations after command changes, including after auto-reset.
            obs = base.observation_manager.compute()
            with torch.no_grad():
                _, _, dones, _ = env.step(policy(obs))
            if writer is not None:
                import math
                q=robot.data.root_quat_w[0].detach().cpu().tolist()
                yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]))
                angle=yaw if args_cli.camera_view=="front" else yaw+math.pi/2
                pos=robot.data.root_pos_w[0].detach().cpu().tolist()
                eye=(pos[0]+1.2*math.cos(angle),pos[1]+1.2*math.sin(angle),pos[2]+.35)
                base.sim.set_camera_view(eye=eye,target=(pos[0],pos[1],pos[2]+.05))
                if step == 0:
                    for _ in range(12):
                        base.sim.render()
                frame = base.render()
                if step == 0:
                    for _ in range(60):
                        if frame.max() > 0:
                            break
                        base.sim.render()
                        frame = base.render()
                    assert frame.max() > 0, "Camera annotator returned a black first frame"
                writer.append_data(frame)
            failed |= base.reset_terminated
            valid = ~dones.bool()
            v = robot.data.root_lin_vel_b
            values = {
                "xy_error": torch.linalg.vector_norm(v[:,:2]-target[:2],dim=-1),
                "yaw_error": (robot.data.root_ang_vel_b[:,2]-target[2]).abs(),
                "base_height": robot.data.root_pos_w[:,2]-base.scene.env_origins[:,2],
                "joint_acc_rms": robot.data.joint_acc.square().mean(-1).sqrt(),
            }
            contact = base.scene["contact_forces"].data.net_forces_w[:,sensor_feet,:].norm(dim=-1)>1.
            slide = robot.data.body_lin_vel_w[:,feet,:2].norm(dim=-1)
            values["foot_contact_slide"]=(slide*contact).sum(-1)/contact.sum(-1).clamp_min(1)
            for side_index,side in enumerate(("left", "right")):
                values[side+"_contact_duty"]=contact[:,side_index].float()
                values[side+"_joint_acc_rms"]=robot.data.joint_acc[:,leg_joints[side_index]].square().mean(-1).sqrt()
                values[side+"_foot_height"]=robot.data.body_pos_w[:,feet[side_index],2]-base.scene.env_origins[:,2]
            for k,v in values.items(): totals[k] += torch.where(valid,v,0.).sum(dtype=torch.float64)
            count += valid.sum()
            min_height=torch.minimum(min_height,torch.where(valid,values["base_height"],float("inf")).min())
        if writer is not None:
            writer.close()
        results.append({"command":label,"target":command,"seed":env_cfg.seed,
                        "num_envs":base.num_envs,"seconds":args_cli.seconds,
                        "failed_trials":int(failed.sum()),"fall_rate":failed.float().mean().item(),
                        "minimum_base_height":min_height.item(),
                        "metrics":{k:(v/count.clamp_min(1)).item() for k,v in totals.items()}})
        print("EVAL",json.dumps(results[-1]),flush=True)
    output=Path(args_cli.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({"checkpoint":resume_path,"deterministic_policy":True,
        "runtime":{"python":sys.executable,"rsl_rl":metadata.version("rsl-rl-lib"),"torch":torch.__version__,"torch_path":torch.__file__},
        "physics_diagnostics":physics_diagnostics,
        "mass_randomization":args_cli.mass_randomization,
        "perturbation_config":{
            "mass_operation":None if base.cfg.events.add_base_mass is None else base.cfg.events.add_base_mass.params["operation"],
            "mass_distribution_params":None if base.cfg.events.add_base_mass is None else base.cfg.events.add_base_mass.params["mass_distribution_params"],
            "friction_static":None if base.cfg.events.physics_material is None else base.cfg.events.physics_material.params["static_friction_range"],
            "friction_dynamic":None if base.cfg.events.physics_material is None else base.cfg.events.physics_material.params["dynamic_friction_range"],
            "push_velocity_range":None if base.cfg.events.push_robot is None else base.cfg.events.push_robot.params["velocity_range"],
            "action_class":base.cfg.actions.JointPositionAction.class_type.__name__,
        },

        "mass_distribution":mass_distribution_report(robot),
        "issue5_condition":args_cli.issue5_condition,"paired_resets":args_cli.paired_resets,
        "pushes":base.cfg.events.push_robot is not None,"observation_noise":env_cfg.observations.policy.enable_corruption,
        "results":results},indent=2)+"\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()