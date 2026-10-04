# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.metadata as metadata
from importlib.metadata import version

from packaging import version as packaging_version

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
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--viewer_eye",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Override the viewport camera position relative to its configured origin.",
)
parser.add_argument(
    "--viewer_lookat",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Override the viewport camera target relative to its configured origin.",
)
parser.add_argument(
    "--viewer_follow_asset", type=str, default=None, help="Continuously follow this scene asset's root pose."
)
parser.add_argument("--evaluation_motion_id", type=int, default=None, help="Pin mimic replay to a clip from frame zero.")
parser.add_argument("--evaluation_output", type=str, default=None, help="Write per-step replay metrics and termination counts.")
parser.add_argument("--viewer_follow_tau", type=float, default=0.5, help="Horizontal follow smoothing in seconds; camera height stays fixed.")
parser.add_argument("--disable_observation_noise", action="store_true", help="Diagnostic ablation: disable policy observation noise.")
parser.add_argument("--seed", type=int, default=None, help="Seed the evaluation environment.")
parser.add_argument("--evaluation_steps", type=int, default=None, help="Bound headless evaluation without recording video.")
parser.add_argument("--telemetry_output", type=str, default=None, help="Save control/physics-rate mimic diagnostics (one environment).")
parser.add_argument("--evaluation_batch", type=str, help="JSON jobs; one simulator startup per seed, no video.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
parser.add_argument('--evaluation_upper_body_termination', action='store_true',
    help='TW100 matched criterion: anchor height/orientation + wrist height, no foot trajectory termination.')
args_cli = parser.parse_args()
if args_cli.evaluation_steps is not None and args_cli.evaluation_steps <= 0:
    parser.error("--evaluation_steps must be positive")
if args_cli.telemetry_output and not (args_cli.evaluation_steps or args_cli.video):
    parser.error("telemetry requires --evaluation_steps or --video")
if args_cli.evaluation_batch and (args_cli.video or args_cli.seed is None):
    parser.error("batch requires an explicit seed and does not support video")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    if args_cli.evaluation_upper_body_termination:
        if 'Custom-Humanoid-30dof' not in args_cli.task:
            raise ValueError('Upper-body validation criterion only supported for custom humanoid')
        env_cfg.terminations.ee_body_pos.params['body_names'] = ['left_wrist', 'right_wrist']
    if args_cli.disable_observation_noise:
        env_cfg.observations.policy.enable_corruption = False
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    if args_cli.viewer_eye is not None:
        env_cfg.viewer.eye = tuple(args_cli.viewer_eye)
    if args_cli.viewer_lookat is not None:
        env_cfg.viewer.lookat = tuple(args_cli.viewer_lookat)
    if args_cli.viewer_follow_asset is not None:
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.asset_name = args_cli.viewer_follow_asset
        env_cfg.viewer.env_index = 0
    if args_cli.evaluation_motion_id is not None or args_cli.evaluation_batch:
        env_cfg.commands.motion.adaptive_max_probability = None
        env_cfg.commands.motion.adaptive_uniform_ratio = 1.0
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)

    from control_runtime import restore as restore_control_runtime, snapshot as control_snapshot
    restore_control_runtime(env_cfg, resume_path)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.evaluation_motion_id is not None:
        env.unwrapped.command_manager.get_term("motion").set_evaluation_motion_ids(
            [args_cli.evaluation_motion_id] * env.unwrapped.num_envs
        )
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if packaging_version.parse(version("rsl-rl-lib")) >= packaging_version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        policy_nn = None
    else:
        # extract the neural network module
        # we do this in a try-except to maintain backwards compatibility.
        try:
            # version 2.3 onwards
            policy_nn = runner.alg.policy
        except AttributeError:
            # version 2.2 and below
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export policy to onnx/jit
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    if args_cli.evaluation_batch:
        from batch_mimic_evaluation import run
        run(env, policy, policy_nn, args_cli, resume_path, env_cfg)
        env.close()
        return

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    evaluation = {"steps": 0, "termination_counts": {}, "motion_metrics": {}}
    telemetry = None
    if args_cli.telemetry_output:
        from mimic_telemetry import MimicTelemetry
        telemetry = MimicTelemetry(env.unwrapped, args_cli.telemetry_output)
    camera_trace = []
    camera_follow = None
    if args_cli.viewer_follow_asset is not None:
        from camera_follow import SmoothCameraFollow
        camera_asset = env.unwrapped.scene[args_cli.viewer_follow_asset]
        camera_follow = SmoothCameraFollow(camera_asset.data.root_pos_w[0].cpu().numpy(), dt, args_cli.viewer_follow_tau)
        camera_reset = False
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        if camera_follow is not None:
            origin = camera_follow.update(camera_asset.data.root_pos_w[0].cpu().numpy(), reset=camera_reset)
            env.unwrapped.sim.set_camera_view(eye=origin + env_cfg.viewer.eye, target=origin + env_cfg.viewer.lookat)
            if args_cli.evaluation_output:
                camera_trace.append((origin.copy(), camera_asset.data.root_pos_w[0].cpu().numpy().copy(), camera_reset))
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if telemetry is not None:
                telemetry.begin(actions)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if camera_follow is not None:
                camera_reset = bool(dones[0].item())
            if telemetry is not None:
                telemetry.end(dones)
            if args_cli.evaluation_output:
                evaluation["steps"] += 1
                manager = env.unwrapped.termination_manager
                for name in manager.active_terms:
                    count = int(manager.get_term(name).sum().item())
                    evaluation["termination_counts"][name] = evaluation["termination_counts"].get(name, 0) + count
                for name, value in env.unwrapped.command_manager.get_term("motion").metrics.items():
                    if name.startswith("error_"):
                        evaluation["motion_metrics"][name] = evaluation["motion_metrics"].get(name, 0.0) + float(value.mean().item())
            if packaging_version.parse(version("rsl-rl-lib")) >= packaging_version.parse("4.0.0"):
                policy.reset(dones)
            elif policy_nn is not None:
                policy_nn.reset(dones)
        if args_cli.video or args_cli.evaluation_steps:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == (args_cli.evaluation_steps or args_cli.video_length):
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if telemetry is not None:
        telemetry.save()
    if args_cli.evaluation_output:
        import json
        from pathlib import Path
        if camera_trace:
            import numpy as np
            np.savez_compressed(str(args_cli.evaluation_output) + ".camera.npz", camera_origin=np.array([x[0] for x in camera_trace]), robot_origin=np.array([x[1] for x in camera_trace]), reset=np.array([x[2] for x in camera_trace]), dt=dt)
        steps = max(evaluation["steps"], 1)
        evaluation["motion_metrics"] = {name: value / steps for name, value in evaluation["motion_metrics"].items()}
        evaluation.update(control_runtime=control_snapshot(env_cfg), camera_follow_tau=args_cli.viewer_follow_tau, checkpoint=resume_path, motion_id=args_cli.evaluation_motion_id,
                          seconds=evaluation["steps"] * dt, num_envs=env.unwrapped.num_envs)
        Path(args_cli.evaluation_output).write_text(json.dumps(evaluation, indent=2) + "\n")
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
