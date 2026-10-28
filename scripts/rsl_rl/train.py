# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--save_interval", type=int, default=None, help="Checkpoint save interval in iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--physics_mean_slip", action="store_true", help="Opt-in slip reward averaged over all physics steps.")
parser.add_argument("--physics_slip_minimum_load", type=float, default=1.0)
parser.add_argument("--allow_physics_slip_load_change", action="store_true", help="Explicit new reward experiment when resuming a different loaded-foot threshold.")
parser.add_argument("--train_ankle_filter_alpha", type=float, default=None, help="Opt-in per-environment ankle action filter during training.")
parser.add_argument("--train_reference_speed_teacher_export", type=str, default=None, help="Opt-in frozen teacher speed mixture before existing ankle filter during training.")
parser.add_argument("--train_candidate_action_scope", choices=["all","lower_body_waist"], default="all", help="Explicit trained teacher/student joint action partition; default preserves full mixture.")
parser.add_argument("--training_minibatch_capture", action="store_true", help="Read-only diagnostic first/last minibatch capture; no additional policy calls or RNG draws.")
parser.add_argument("--train_controlled_action_likelihood", action="store_true", help="Opt-in active17 PPO score/entropy; raw30 sampling and full30 KL unchanged.")
parser.add_argument("--train_leg_filter_alpha", type=float, default=None, help="Opt-in legs14 filter before the existing ankle filter during training.")
parser.add_argument("--policy_anchor_checkpoint", type=str, default=None, help="Explicit frozen teacher checkpoint, retained across interrupted recovery.")
parser.add_argument("--policy_anchor_coefficient", type=float, default=0.0, help="Frozen resumed-policy mean-action MSE coefficient; opt-in.")
parser.add_argument("--policy_anchor_hip_multiplier", type=float, default=1.0, help="Opt-in four hip pitch/roll anchor weight multiplier, at least one.")
parser.add_argument("--policy_anchor_candidate_multiplier", type=float, default=1.0, help="Opt-in teacher MSE weight for controlled17 leg/foot/waist only; requires scoped controlled likelihood.")
parser.add_argument("--policy_anchor_arm_multiplier", type=float, default=1.0, help="Opt-in all10 arm teacher anchor multiplier, at least one; training loss only.")
parser.add_argument("--policy_anchor_temporal_coefficient", type=float, default=0.0, help="Training-only consecutive teacher-relative mean-action increment MSE; opt-in.")
parser.add_argument("--policy_anchor_temporal_order", type=int, choices=[1,2], default=1, help="Opt-in mean-action temporal derivative order; order2 uses three valid observations.")
parser.add_argument("--policy_anchor_temporal_target", choices=["teacher_relative","student_mean"], default="teacher_relative", help="Opt-in direct student mean curvature; requires temporal order2.")
parser.add_argument("--policy_anchor_temporal_observations", choices=["policy","clean_critic"], default="policy", help="Opt-in same-state clean critic reconstruction for temporal loss only; PPO observations unchanged.")
parser.add_argument("--contact_feedback_mode", choices=["actual","zeros"], default=None, help="Opt-in eight actual contact features or matched zero-feature control.")
parser.add_argument("--restore_optimizer_learning_rate", action="store_true", help="Restore adaptive PPO learning rate from loaded optimizer state.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.save_interval is not None and args_cli.save_interval <= 0:
    parser.error("--save_interval must be greater than zero")

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

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
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

import gymnasium as gym
import inspect
import os
import shutil
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner  # TODO: Consider printing the experiment name in the terminal.

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval

    # Convert legacy policy configs to the actor/critic format expected by newer rsl-rl releases.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

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
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

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

    # Keep the filter contract explicit for resumed and deployed policies.
    import json
    from pathlib import Path
    from temporal_target_contract import validate_target,validate_resume_target
    validate_target(args_cli.policy_anchor_temporal_coefficient,args_cli.policy_anchor_temporal_order,args_cli.policy_anchor_temporal_target)
    if args_cli.policy_anchor_temporal_order==2 and not args_cli.policy_anchor_temporal_coefficient:
        raise ValueError('Second-order mean anchor requires positive coefficient')
    if args_cli.contact_feedback_mode and args_cli.policy_anchor_temporal_coefficient and args_cli.policy_anchor_temporal_order!=2:
        raise ValueError('Contact temporal combination audited only for second order')
    if args_cli.policy_anchor_temporal_observations=='clean_critic':
        if args_cli.policy_anchor_temporal_order!=2 or args_cli.policy_anchor_temporal_target!='student_mean' or args_cli.policy_anchor_temporal_coefficient<=0 or args_cli.contact_feedback_mode!='actual' or not args_cli.train_controlled_action_likelihood:
            raise ValueError('Clean temporal requires positive student_mean order2, actual contact and controlled likelihood')
    if agent_cfg.resume:
        from clean_temporal_policy import validate_resume_source
        prior_anchor=Path(resume_path).parent/'policy_anchor.json'
        validate_resume_source(json.loads(prior_anchor.read_text()) if prior_anchor.exists() else {},args_cli.policy_anchor_temporal_observations)
        prior_clean=Path(resume_path).parent/'params/clean_temporal_policy.json'
        if prior_clean.exists() and args_cli.policy_anchor_temporal_observations!='clean_critic':
            raise ValueError('Resume cannot remove clean temporal observations')
        if prior_anchor.exists() and json.loads(prior_anchor.read_text()).get('temporal_observation_source')=='clean_critic' and not prior_clean.exists():
            raise ValueError('Saved clean temporal contract missing')
    if agent_cfg.resume:
        saved_contact=Path(resume_path).parent/'params/contact_feedback.json'
        if saved_contact.exists() and json.loads(saved_contact.read_text())['mode']!=args_cli.contact_feedback_mode:
            raise ValueError('Resume must preserve contact observation mode')
        saved_anchor=Path(resume_path).parent/'policy_anchor.json'
        if saved_anchor.exists():
            from anchored_ppo import validate_arm_resume
            validate_arm_resume(json.loads(saved_anchor.read_text()), args_cli.policy_anchor_arm_multiplier)
            saved_candidate_anchor=json.loads(saved_anchor.read_text())
            if args_cli.policy_anchor_candidate_multiplier!=1 or saved_candidate_anchor.get('candidate_multiplier',1.)!=1:
                from anchored_ppo import validate_candidate_resume
                import hashlib
                frozen_initialization=bool(args_cli.policy_anchor_checkpoint) and hashlib.sha256(Path(resume_path).read_bytes()).hexdigest()==hashlib.sha256(Path(args_cli.policy_anchor_checkpoint).read_bytes()).hexdigest()
                validate_candidate_resume(saved_candidate_anchor,args_cli.policy_anchor_candidate_multiplier,allow_frozen_initialization=frozen_initialization)
            validate_resume_target(json.loads(saved_anchor.read_text()),args_cli.policy_anchor_temporal_target)
            previous_temporal=json.loads(saved_anchor.read_text()).get('temporal_coefficient',0.0)
            previous_order=json.loads(saved_anchor.read_text()).get('temporal_order',1)
            if previous_temporal and previous_order!=args_cli.policy_anchor_temporal_order:
                raise ValueError('Resume must preserve temporal derivative order')
            if previous_temporal and previous_temporal != args_cli.policy_anchor_temporal_coefficient:
                raise ValueError('Resume must preserve training temporal anchor coefficient')
        if (Path(resume_path).parent/'params/physics_slip.json').exists() and not args_cli.physics_mean_slip:
            raise ValueError('Resume must preserve physics-rate slip reward')
        saved_physics_slip=Path(resume_path).parent/'params/physics_slip.json'
        if saved_physics_slip.exists():
            from physics_slip_reward import validate_load_resume
            validate_load_resume(json.loads(saved_physics_slip.read_text()),args_cli.physics_slip_minimum_load,args_cli.allow_physics_slip_load_change)
        saved_filter=Path(resume_path).parent/'params/action_filter.json'
        if saved_filter.exists() and args_cli.train_ankle_filter_alpha != json.loads(saved_filter.read_text())['alpha']:
            raise ValueError('Resume must preserve checkpoint training action filter')
    if args_cli.train_ankle_filter_alpha is not None:
        from training_action_filter import wrap_environment
        env=wrap_environment(env,args_cli.train_ankle_filter_alpha)
        path=Path(log_dir)/'params/action_filter.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(env.action_filter.audit,indent=2))

    leg_filter_environment=None
    from training_leg_filter import validate_resume_contract
    if agent_cfg.resume:
        saved_leg_filter=Path(resume_path).parent/'params/leg_action_filter.json'
        if saved_leg_filter.exists():
            validate_resume_contract(json.loads(saved_leg_filter.read_text()),args_cli.train_leg_filter_alpha)
    if args_cli.train_leg_filter_alpha is not None:
        if args_cli.train_ankle_filter_alpha is None:
            raise ValueError('Audited training leg filter requires explicit existing ankle filter')
        from training_leg_filter import wrap_leg_environment
        env=wrap_leg_environment(env,args_cli.train_leg_filter_alpha)
        leg_filter_environment=env
        path=Path(log_dir)/'params/leg_action_filter.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(env.leg_action_filter.audit,indent=2))

    from physics_slip_reward import validate_minimum_load
    validate_minimum_load(args_cli.physics_slip_minimum_load)
    if not args_cli.physics_mean_slip and (args_cli.physics_slip_minimum_load!=1. or args_cli.allow_physics_slip_load_change):
        raise ValueError('Slip load configuration requires physics_mean_slip')
    if args_cli.physics_mean_slip:
        from physics_slip_reward import install
        physics_slip_audit=install(env,minimum_load=args_cli.physics_slip_minimum_load)
        path=Path(log_dir)/'params/physics_slip.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(physics_slip_audit,indent=2))

    from highpass_reward_contract import prepare as prepare_highpass_reward
    highpass_reward_audit=prepare_highpass_reward(env_cfg,log_dir,resume_path if agent_cfg.resume else None)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args_cli.contact_feedback_mode:
        from contact_policy import ContactPolicyObservations
        env=ContactPolicyObservations(env,args_cli.contact_feedback_mode)

    import math
    if not math.isfinite(args_cli.policy_anchor_temporal_coefficient) or args_cli.policy_anchor_temporal_coefficient < 0:
        raise ValueError('Temporal anchor coefficient must be finite and nonnegative')
    clean_temporal_contract=None
    if args_cli.policy_anchor_temporal_coefficient:
        if args_cli.policy_anchor_temporal_order==1:
            from temporal_anchor import PreviousPolicyTraining
            env = PreviousPolicyTraining(env)
        else:
            if args_cli.policy_anchor_temporal_observations=='clean_critic':
                from clean_temporal_policy import CleanPolicyObservations,CleanTwoPoliciesTraining
                env=CleanPolicyObservations(env)
                clean_temporal_contract=env.contract
                if agent_cfg.resume:
                    previous_clean=Path(resume_path).parent/'params/clean_temporal_policy.json'
                    if previous_clean.exists() and json.loads(previous_clean.read_text())!=clean_temporal_contract:
                        raise ValueError('Resume clean temporal reconstruction contract changed')
                path=Path(log_dir)/'params/clean_temporal_policy.json';path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(clean_temporal_contract,indent=2))
                env=CleanTwoPoliciesTraining(env)
            else:
                from second_order_anchor import PreviousTwoPoliciesTraining
                env = PreviousTwoPoliciesTraining(env)

    controlled_activity_environment=None
    controlled_likelihood_contract=None
    saved_likelihood=Path(resume_path).parent/'params/controlled_action_likelihood.json' if agent_cfg.resume else None
    if saved_likelihood is not None and saved_likelihood.exists() and not args_cli.train_controlled_action_likelihood:
        raise ValueError('Resume cannot remove controlled action likelihood')
    if args_cli.train_controlled_action_likelihood:
        if not agent_cfg.resume or args_cli.train_candidate_action_scope!='lower_body_waist' or args_cli.train_reference_speed_teacher_export is None or args_cli.policy_anchor_coefficient<=0:
            raise ValueError('Controlled likelihood requires resumed scoped frozen-teacher mixture')
        if not args_cli.policy_anchor_checkpoint:
            raise ValueError('Controlled likelihood requires explicit frozen teacher checkpoint')
        from controlled_action_likelihood import ControlledActionActivity,make_contract as make_likelihood_contract,validate_resume as validate_likelihood_resume
        import hashlib
        likelihood_names=list(env.unwrapped.action_manager.get_term('JointPositionAction')._joint_names)
        if len(likelihood_names)!=env.unwrapped.action_manager.total_action_dim:
            raise ValueError('Controlled likelihood requires one explicit30 joint action term')
        controlled_likelihood_contract=make_likelihood_contract(likelihood_names)
        prior=json.loads(saved_likelihood.read_text()) if saved_likelihood is not None and saved_likelihood.exists() else None
        exact_teacher_initialization=(hashlib.sha256(Path(resume_path).read_bytes()).hexdigest()==hashlib.sha256(Path(args_cli.policy_anchor_checkpoint).read_bytes()).hexdigest())
        validate_likelihood_resume(prior,controlled_likelihood_contract,allow_teacher_initialization=exact_teacher_initialization)
        command=env.unwrapped.command_manager.get_term('motion')
        env=ControlledActionActivity(env,command)
        controlled_activity_environment=env

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args_cli.contact_feedback_mode:
        if 'policy' in runner.alg.critic.obs_groups:
            raise ValueError('Contact experiment must preserve separate critic inputs')
        path=Path(log_dir)/'params/contact_feedback.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'version':1,'mode':args_cli.contact_feedback_mode,'base_dim':env.base_dim,'extra_dim':8,'load_scale_N':100.,'speed_scale_m_s':.1,'loaded_threshold_N':5.,'reset':'zero first per-environment observation','sensor_prefix_training':'slip_','teacher_input':'original policy only'},indent=2))
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    if args_cli.policy_anchor_temporal_coefficient:
        if runner.alg.actor.obs_groups != ['policy']:
            raise ValueError('Temporal anchor requires the original policy-only actor observation group')
        path=Path(log_dir)/'params/temporal_anchor.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'coefficient':args_cli.policy_anchor_temporal_coefficient,
            'training_auxiliary_keys':['anchor_previous_policy','anchor_previous_valid'],
            'actor_groups':runner.alg.actor.obs_groups,'policy_input_unchanged':True,
            'reset':'exclude first action after each per-environment reset'},indent=2))
        if args_cli.policy_anchor_temporal_order==2:
            from second_order_anchor import PREVIOUS,OLDER,VALID
            meta=json.loads(path.read_text());meta.update(order=2,training_auxiliary_keys=[PREVIOUS,OLDER,VALID],reset='exclude first two actions after each per-environment reset')
            if args_cli.policy_anchor_temporal_target=='student_mean':meta['target']='student_mean'
            path.write_text(json.dumps(meta,indent=2))
    if clean_temporal_contract is not None:
        from clean_temporal_policy import CLEAN
        path=Path(log_dir)/'params/temporal_anchor.json';meta=json.loads(path.read_text())
        meta.update(observation_source='clean_critic',training_auxiliary_keys=[CLEAN,PREVIOUS,OLDER,VALID])
        path.write_text(json.dumps(meta,indent=2))
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if args_cli.contact_feedback_mode:
            from contact_policy import load_contact_checkpoint
            migration=load_contact_checkpoint(runner,resume_path)
            (Path(log_dir)/'contact_checkpoint_migration.json').write_text(json.dumps(migration,indent=2))
        else:
            runner.load(resume_path)
        if args_cli.restore_optimizer_learning_rate:
            import json
            from resume_learning_rate import restore_learning_rate
            resume_audit=restore_learning_rate(runner.alg)
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "resume_learning_rate.json"), "w") as stream:
                json.dump(resume_audit, stream, indent=2)
            print(f"[INFO] Restored scheduler learning rate: {resume_audit}")

    import math
    if not math.isfinite(args_cli.policy_anchor_hip_multiplier) or args_cli.policy_anchor_hip_multiplier < 1:
        raise ValueError('Hip anchor multiplier must be finite and at least one')
    if args_cli.policy_anchor_hip_multiplier != 1 and args_cli.policy_anchor_coefficient == 0:
        raise ValueError('Hip anchor multiplier requires an enabled policy anchor')
    if not math.isfinite(args_cli.policy_anchor_arm_multiplier) or args_cli.policy_anchor_arm_multiplier < 1:
        raise ValueError('Arm anchor multiplier must be finite and at least one')
    if args_cli.policy_anchor_arm_multiplier != 1 and args_cli.policy_anchor_coefficient == 0:
        raise ValueError('Arm anchor multiplier requires an enabled policy anchor')
    if not math.isfinite(args_cli.policy_anchor_candidate_multiplier) or args_cli.policy_anchor_candidate_multiplier < 1:
        raise ValueError('Candidate anchor multiplier must be finite and at least one')
    if args_cli.policy_anchor_candidate_multiplier != 1:
        if args_cli.policy_anchor_coefficient <= 0 or not args_cli.train_controlled_action_likelihood or args_cli.train_candidate_action_scope!='lower_body_waist':
            raise ValueError('Candidate anchor multiplier requires controlled17 frozen teacher mixture')
        if args_cli.policy_anchor_hip_multiplier != 1 or args_cli.policy_anchor_arm_multiplier != 1:
            raise ValueError('Candidate anchor calibration cannot combine other joint multipliers')
    if args_cli.policy_anchor_coefficient != 0 or args_cli.policy_anchor_temporal_coefficient != 0:
        if not agent_cfg.resume:
            raise ValueError('Policy anchor requires a resumed teacher checkpoint')
        import json, hashlib
        from anchored_ppo import install_anchor
        teacher_path=args_cli.policy_anchor_checkpoint or resume_path
        teacher_state=torch.load(teacher_path,map_location=agent_cfg.device,weights_only=False)['actor_state_dict']
        teacher_contact_audit=None
        if args_cli.contact_feedback_mode:
            from contact_policy import expand_actor_state
            teacher_state,teacher_contact_audit=expand_actor_state(runner.alg.actor,teacher_state)
        action_weights = None
        if args_cli.policy_anchor_hip_multiplier != 1 or args_cli.policy_anchor_arm_multiplier != 1 or args_cli.policy_anchor_candidate_multiplier != 1:
            from anchored_ppo import hip_anchor_weights, arm_anchor_weights, candidate_anchor_weights
            names = list(env.unwrapped.action_manager.get_term('JointPositionAction')._joint_names)
            if len(names) != env.unwrapped.action_manager.total_action_dim:
                raise ValueError('Hip anchor requires one joint position action term')
            action_weights = hip_anchor_weights(names, args_cli.policy_anchor_hip_multiplier)
            if args_cli.policy_anchor_arm_multiplier != 1:
                action_weights = arm_anchor_weights(names, args_cli.policy_anchor_arm_multiplier, action_weights)
            if args_cli.policy_anchor_candidate_multiplier != 1:
                action_weights = candidate_anchor_weights(names, args_cli.policy_anchor_candidate_multiplier, action_weights)
        audit=install_anchor(runner.alg,args_cli.policy_anchor_coefficient,teacher_state,action_weights,temporal_coefficient=args_cli.policy_anchor_temporal_coefficient,temporal_order=args_cli.policy_anchor_temporal_order,temporal_target=args_cli.policy_anchor_temporal_target)
        if clean_temporal_contract is not None:
            from clean_temporal_policy import configure_clean_temporal
            clean_install=configure_clean_temporal(runner.alg,clean_temporal_contract)
            audit['temporal_observation_source']='clean_critic'
            (Path(log_dir)/'clean_temporal_install_audit.json').write_text(json.dumps(clean_install,indent=2))
        if action_weights is not None:
            audit['action_joint_names'] = names
            audit['hip_multiplier'] = args_cli.policy_anchor_hip_multiplier
            if args_cli.policy_anchor_arm_multiplier != 1:
                audit['arm_multiplier'] = args_cli.policy_anchor_arm_multiplier
            if args_cli.policy_anchor_candidate_multiplier != 1:
                audit['candidate_multiplier'] = args_cli.policy_anchor_candidate_multiplier
                audit['candidate_scope'] = 'lower_body_waist'
        if teacher_contact_audit is not None:audit['teacher_input_expansion']=teacher_contact_audit
        audit['teacher_checkpoint']=teacher_path
        with open(teacher_path,'rb') as source:
            audit['teacher_checkpoint_sha256']=hashlib.sha256(source.read()).hexdigest()
        with open(os.path.join(log_dir,'policy_anchor.json'),'w') as stream:
            json.dump(audit,stream,indent=2)
        print(f'[INFO] Policy anchor: {audit}')

    if controlled_likelihood_contract is not None:
        from controlled_likelihood_ppo import install_controlled_likelihood
        likelihood_audit=install_controlled_likelihood(runner.alg,controlled_likelihood_contract)
        (Path(log_dir)/'params/controlled_action_likelihood.json').write_text(json.dumps(controlled_likelihood_contract,indent=2))
        (Path(log_dir)/'controlled_likelihood_install_audit.json').write_text(json.dumps(likelihood_audit,indent=2))

    training_speed_environment=None
    from training_speed_blend import TrainingSpeedBlend,make_contract,validate_resume,audit_teacher_binding
    saved_speed_contract=Path(resume_path).parent/'params/training_speed_blend.json' if agent_cfg.resume else None
    if saved_speed_contract is not None and saved_speed_contract.exists() and args_cli.train_reference_speed_teacher_export is None:
        raise ValueError('Resume cannot remove trained reference-speed mixture')
    if args_cli.train_candidate_action_scope!='all' and args_cli.train_reference_speed_teacher_export is None:
        raise ValueError('Action partition requires explicit training mixture teacher')
    if args_cli.train_reference_speed_teacher_export is not None:
        if not agent_cfg.resume or args_cli.policy_anchor_coefficient<=0 or args_cli.train_ankle_filter_alpha!=.5 or args_cli.train_leg_filter_alpha is not None:
            raise ValueError('Training mixture requires frozen anchor, ankle .5, and no candidate leg filter')
        teacher_binding=audit_teacher_binding(args_cli.train_reference_speed_teacher_export,teacher_path)
        scope_action_names=None
        if args_cli.train_candidate_action_scope!='all':
            scope_action_names=list(env.unwrapped.action_manager.get_term('JointPositionAction')._joint_names)
            if len(scope_action_names)!=env.unwrapped.action_manager.total_action_dim:raise ValueError('Action partition requires one joint position term')
        contract=make_contract(args_cli.train_reference_speed_teacher_export,teacher_path,scope_action_names,args_cli.train_candidate_action_scope)
        if controlled_likelihood_contract is not None:
            from controlled_action_likelihood import attach_training_contract
            contract=attach_training_contract(contract,controlled_likelihood_contract)
        if args_cli.train_candidate_action_scope!='all':
            from candidate_scope_contract import validate_scope_resume
            previous=json.loads(saved_speed_contract.read_text()) if saved_speed_contract is not None and saved_speed_contract.exists() else None
            if previous is None and hashlib.sha256(Path(resume_path).read_bytes()).hexdigest()!=hashlib.sha256(Path(teacher_path).read_bytes()).hexdigest():
                raise ValueError('New action partition must initialize from exact frozen teacher checkpoint')
            validate_scope_resume(previous.get('candidate_action_scope') if previous is not None else None,contract['candidate_action_scope'],allow_legacy_initialization=previous is None)
        (Path(log_dir)/'training_speed_teacher_binding.json').write_text(json.dumps(teacher_binding,indent=2))
        if saved_speed_contract is not None and saved_speed_contract.exists():
            validate_resume(json.loads(saved_speed_contract.read_text()),contract)
        frozen_teacher=torch.jit.load(args_cli.train_reference_speed_teacher_export,map_location=env.unwrapped.device)
        command=env.unwrapped.command_manager.get_term('motion')
        env=TrainingSpeedBlend(env,frozen_teacher,command,contract,action_names=scope_action_names)
        training_speed_environment=env;runner.env=env
        (Path(log_dir)/'params/training_speed_blend.json').write_text(json.dumps(contract,indent=2))

    training_batch_capture=None
    if args_cli.training_minibatch_capture:
        if controlled_likelihood_contract is None:
            raise ValueError('Actual minibatch capture audited only for controlled likelihood probe')
        from training_minibatch_capture import install_capture
        manifest=Path(os.environ['PRACTICE9_CUSTOM_MOTION_MANIFEST'])
        motions=json.loads(manifest.read_text())['motions']
        provenance={'purpose':'diagnostic only; generated checkpoint is not a candidate',
            'initial_checkpoint':str(resume_path),
            'initial_checkpoint_sha256':hashlib.sha256(Path(resume_path).read_bytes()).hexdigest(),
            'teacher_checkpoint_sha256':hashlib.sha256(Path(teacher_path).read_bytes()).hexdigest(),
            'training_manifest':str(manifest),'training_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'training_motion_ids':[m['id'] for m in motions],
            'training_frames':sum(m['frames'] for m in motions),
            'training_contract':contract,'likelihood_contract':controlled_likelihood_contract,
            'initial_runner_iteration':runner.current_learning_iteration}
        batches=runner.alg.num_learning_epochs*runner.alg.num_mini_batches
        training_batch_capture=install_capture(runner.alg,Path(log_dir)/'training_minibatches',slots=(0,batches-1),provenance=provenance)
        (Path(log_dir)/'params/training_minibatch_capture.json').write_text(json.dumps({'enabled':True,'ordinals':[0,batches-1],'provenance':provenance},indent=2))

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    from control_runtime import save as save_control_runtime
    save_control_runtime(env_cfg, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    if training_batch_capture is not None:
        batch_audit=training_batch_capture.finish()
        (Path(log_dir)/'training_minibatch_capture_completed.json').write_text(json.dumps(batch_audit,indent=2))
    if controlled_activity_environment is not None:
        activity_audit=controlled_activity_environment.save_trace(Path(log_dir)/'controlled_activity_trace.npz')
        (Path(log_dir)/'controlled_activity_completed.json').write_text(json.dumps(activity_audit,indent=2))
    if training_speed_environment is not None:
        speed_audit=training_speed_environment.save_trace(Path(log_dir)/'training_speed_blend_trace.npz')
        (Path(log_dir)/'training_speed_blend_completed.json').write_text(json.dumps(speed_audit,indent=2))
    if leg_filter_environment is not None:
        leg_filter_environment.save_trace(Path(log_dir)/'training_leg_filter_trace.npz')

    # close the simulator
    if args_cli.physics_mean_slip:
        accumulator=env.unwrapped._physics_slip_accumulator
        if accumulator.count != 0 or accumulator.completed_windows == 0:
            raise RuntimeError('Invalid final physics slip accumulator state')
        (Path(log_dir)/'physics_slip_completed.json').write_text(json.dumps({
            'completed_windows':accumulator.completed_windows,
            'unconsumed_samples':accumulator.count},indent=2))
    from highpass_reward_contract import finish as finish_highpass_reward
    finish_highpass_reward(env,log_dir,highpass_reward_audit)
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
