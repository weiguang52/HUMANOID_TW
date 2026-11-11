# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument("--last_action_observation", default="requested", choices=["requested","applied"])
parser.add_argument("--mass_randomization", default="legacy", choices=["legacy","bounded"])
parser.add_argument("--post_push_pitch_weight", type=float, default=None)
parser.add_argument("--critic_recovery_clock", action="store_true", default=False)
parser.add_argument("--issue12_audit_dir", default="")
parser.add_argument("--issue12_calibration_steps", type=int, default=0)
parser.add_argument("--issue12_smoke_steps", type=int, default=0)
parser.add_argument("--issue5_training_mode", default="legacy", choices=["legacy","remove","strengthen"])
parser.add_argument("--issue5_factor", default="push", choices=["push","material","delay","noise"])
parser.add_argument("--experiment_profile", default="default", choices=["default","control","entropy","yaw","range05","range08","robust05","robust08"])
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401  (registers Unitree-TW-30dof-Velocity)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    from unitree_rl_lab.tasks.locomotion.robots.tw.experiments import configure_experiment
    configure_experiment(env_cfg, agent_cfg, args_cli.experiment_profile)
    if args_cli.issue5_training_mode != "legacy":
        assert args_cli.experiment_profile == "robust08"
        from unitree_rl_lab.tasks.locomotion.robots.tw.experiments import configure_issue5_training
        configure_issue5_training(env_cfg, args_cli.issue5_training_mode, args_cli.issue5_factor)

    from unitree_rl_lab.tasks.locomotion.robots.tw.mass_randomization import configure_mass_randomization
    configure_mass_randomization(env_cfg, args_cli.mass_randomization)
    from unitree_rl_lab.tasks.locomotion.robots.tw.action_observation import configure_action_observation
    configure_action_observation(env_cfg, args_cli.last_action_observation)
    from unitree_rl_lab.tasks.locomotion.robots.tw.post_push_recovery import configure_post_push_pitch_reward
    configure_post_push_pitch_reward(env_cfg, args_cli.post_push_pitch_weight)
    if args_cli.critic_recovery_clock:
        from unitree_rl_lab.tasks.locomotion.robots.tw.critic_recovery_clock import configure_critic_recovery_clock
        configure_critic_recovery_clock(env_cfg, True)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        if args_cli.critic_recovery_clock:
            from issue14_resume import construct_parent_width_runner
            runner = construct_parent_width_runner(env, agent_cfg.to_dict(), log_dir, agent_cfg.device)
        else:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        # TW fine-tune: after loading, optionally re-inflate the policy action std (the loaded std is ~0.02 and cannot explore).
        _ft_std = os.environ.get("TW_RESET_STD")
        if _ft_std:
            for _mn in ("actor", "critic", "policy", "actor_critic"):
                _m = getattr(runner.alg, _mn, None)
                if _m is None or not hasattr(_m, "named_parameters"):
                    continue
                for _pn, _p in _m.named_parameters():
                    if _pn.endswith("log_std_param"):
                        print(f"[TW] {_mn}.{_pn}: std {torch.exp(_p.data).mean().item():.4f} -> {float(_ft_std)}", flush=True)
                        _p.data.fill_(float(torch.log(torch.tensor(float(_ft_std)))))
                    elif _pn.endswith("std_param"):
                        print(f"[TW] {_mn}.{_pn}: std {_p.data.mean().item():.4f} -> {float(_ft_std)}", flush=True)
                        _p.data.fill_(float(_ft_std))

    monitor = None
    if args_cli.issue12_audit_dir:
        assert agent_cfg.resume and (args_cli.post_push_pitch_weight is not None or args_cli.issue12_smoke_steps)
        assert not os.environ.get("TW_RESET_STD"), "Issue12 forbids std reset"
        from pathlib import Path
        from issue12_monitor import RewardMonitor, audit_loaded
        from trial_risk_records import state_audit
        audit_dir = Path(args_cli.issue12_audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_loaded(runner, env.unwrapped, resume_path, audit_dir / "loaded_state_audit.json")
        state_audit(env.unwrapped, audit_dir / "initial_state", "initial")
        if args_cli.post_push_pitch_weight is not None:
            monitor = RewardMonitor(env.unwrapped, args_cli.post_push_pitch_weight, audit_dir)
        if monitor is not None and not (args_cli.issue12_calibration_steps or args_cli.issue12_smoke_steps):
            monitor.attach(runner)

    if args_cli.critic_recovery_clock:
        assert agent_cfg.resume and args_cli.issue12_audit_dir and not os.environ.get("TW_RESET_STD")
        from issue14_resume import expand_loaded_critic
        expand_loaded_critic(runner, env.unwrapped, Path(args_cli.issue12_audit_dir) / "critic_clock_migration.json")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    if args_cli.issue12_smoke_steps:
        from issue12_monitor import smoke
        smoke(runner, env, env.unwrapped, monitor, args_cli.issue12_smoke_steps, args_cli.issue12_audit_dir)
    elif args_cli.issue12_calibration_steps:
        assert monitor is not None
        from issue12_monitor import calibration
        calibration(runner, env, env.unwrapped, monitor, args_cli.issue12_calibration_steps,
                    Path(args_cli.issue12_audit_dir) / "calibration.json")
    else:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    if monitor is not None:
        expected = args_cli.issue12_calibration_steps or args_cli.issue12_smoke_steps or agent_cfg.max_iterations * agent_cfg.num_steps_per_env
        assert monitor.total_steps == expected, (monitor.total_steps, expected)
        monitor.close()
        Path(args_cli.issue12_audit_dir, "run_summary.json").write_text(__import__("json").dumps({
            "log_dir": log_dir, "parent_checkpoint": resume_path,
            "additional_updates": 0 if (args_cli.issue12_calibration_steps or args_cli.issue12_smoke_steps) else agent_cfg.max_iterations,
            "environment_steps": monitor.total_steps, "num_envs": env.unwrapped.num_envs,
            "final_iteration": runner.current_learning_iteration, "std_reset": False,
            "post_push_pitch_weight": args_cli.post_push_pitch_weight}, indent=2))

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
