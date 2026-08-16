#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a mimic policy with one fixed HumanML3D motion per environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

PRACTICE9_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "practice9"
sys.path.insert(0, str(PRACTICE9_SCRIPT_DIR))

from evaluation_utils import (  # isort: skip
    ERROR_METRICS,
    FailureTraceBuffer,
    TERMINATION_TERMS,
    atomic_write_reports,
    build_report,
    parse_motion_indices,
)

parser = argparse.ArgumentParser(description="Evaluate every selected HumanML3D motion independently.")
parser.add_argument(
    "--task",
    default="Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D",
    help="Registered Isaac Lab task.",
)
parser.add_argument("--motion_manifest", required=True, help="Validated motion manifest used by the policy.")
parser.add_argument(
    "--motion_indices",
    default=None,
    help="Optional comma-separated indexes and inclusive ranges, for example 0,3-7.",
)
parser.add_argument("--episodes_per_motion", type=int, default=3)
parser.add_argument("--output_dir", required=True, help="New data-disk directory for JSON and CSV results.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument(
    '--failure_trace_frames',
    type=int,
    default=0,
    help='Record this many pre-terminal state frames for failed episodes; zero disables diagnostics.',
)
parser.add_argument(
    '--failure_trace_max_episodes',
    type=int,
    default=100,
    help='Maximum number of failed episodes retained in failure_traces.json.',
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.episodes_per_motion <= 0:
    parser.error("--episodes_per_motion must be positive")
if args_cli.failure_trace_frames < 0:
    parser.error('--failure_trace_frames must be non-negative')
if args_cli.failure_trace_max_episodes <= 0:
    parser.error('--failure_trace_max_episodes must be positive')
manifest_path = Path(args_cli.motion_manifest).expanduser().resolve()
if not manifest_path.is_file():
    parser.error(f"Motion manifest does not exist: {manifest_path}")
manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest_motions = manifest_payload.get("motions")
if manifest_payload.get("schema_version") != 1 or not isinstance(manifest_motions, list) or not manifest_motions:
    parser.error(f"Invalid schema-v1 motion manifest: {manifest_path}")
manifest_ids = [str(item.get("id", Path(str(item["file"])).stem)) for item in manifest_motions]
if len(set(manifest_ids)) != len(manifest_ids):
    parser.error("Motion manifest contains duplicate ids")
selected_indices = parse_motion_indices(args_cli.motion_indices, len(manifest_ids))
args_cli.num_envs = len(selected_indices)
os.environ["PRACTICE9_CUSTOM_MOTION_MANIFEST"] = str(manifest_path)
os.umask(0o077)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
installed_version = metadata.version("rsl-rl-lib")

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_error_magnitude
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from packaging import version as packaging_version
from rsl_rl.runners import OnPolicyRunner
from unitree_rl_lab.tasks.mimic.mdp.commands import MotionCommand
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


class _TerminalFailureRecorder:
    '''Capture exact terminal state before Isaac Lab automatically resets an environment.'''

    def __init__(self, command: MotionCommand, raw_env, ee_body_names: list[str]):
        self.command = command
        self.raw_env = raw_env
        self.ee_body_names = ee_body_names
        self.ee_body_indexes = [command.cfg.body_names.index(name) for name in ee_body_names]
        self.latest_actions: torch.Tensor | None = None
        self.pending: dict[int, dict] = {}
        self.contact_sensor = raw_env.scene['contact_forces']
        command._evaluation_failure_recorder = self

    @staticmethod
    def _list(value: torch.Tensor):
        return value.detach().cpu().tolist()

    def _contact_snapshot(self, env_index: int) -> dict:
        try:
            force_norms = torch.norm(self.contact_sensor.data.net_forces_w[env_index], dim=-1)
            names = list(self.contact_sensor.body_names)
            values = self._list(force_norms)
            return {
                'available': True,
                'force_norm_by_body': {name: float(value) for name, value in zip(names, values, strict=True)},
            }
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as error:
            return {'available': False, 'error': f'{type(error).__name__}: {error}'}

    def _terminal_snapshot(self, env_index: int) -> dict:
        command = self.command
        reference_body = command.body_pos_relative_w[env_index]
        robot_body = command.robot_body_pos_w[env_index]
        reference_joint = command.joint_pos[env_index]
        robot_joint = command.robot_joint_pos[env_index]
        limits = command.robot.data.soft_joint_pos_limits
        limits = limits[env_index] if limits.ndim == 3 else limits
        limits = limits[command.joint_indexes]
        joint_margin = torch.minimum(robot_joint - limits[:, 0], limits[:, 1] - robot_joint)
        snapshot = {
            'motion_frame': int(command.time_steps[env_index].item()),
            'reference_joint_pos': self._list(reference_joint),
            'robot_joint_pos': self._list(robot_joint),
            'reference_joint_vel': self._list(command.joint_vel[env_index]),
            'robot_joint_vel': self._list(command.robot_joint_vel[env_index]),
            'joint_soft_limit_margin': self._list(joint_margin),
            'reference_body_pos_w': self._list(reference_body),
            'robot_body_pos_w': self._list(robot_body),
            'body_position_error': self._list(torch.norm(reference_body - robot_body, dim=-1)),
            'body_z_error': self._list(torch.abs(reference_body[:, 2] - robot_body[:, 2])),
            'anchor_position_error_xyz': self._list(command.anchor_pos_w[env_index] - command.robot_anchor_pos_w[env_index]),
            'anchor_rotation_error': float(
                quat_error_magnitude(
                    command.anchor_quat_w[env_index : env_index + 1],
                    command.robot_anchor_quat_w[env_index : env_index + 1],
                )[0].item()
            ),
            'contact': self._contact_snapshot(env_index),
            'termination_measurements': {},
        }
        if self.latest_actions is not None:
            snapshot['policy_action'] = self._list(self.latest_actions[env_index])
        return snapshot

    def record(
        self,
        term_name: str,
        failed: torch.Tensor,
        measure: torch.Tensor,
        body_names: list[str] | None,
    ) -> None:
        for env_index_tensor in torch.where(failed)[0]:
            env_index = int(env_index_tensor.item())
            frame = int(self.command.time_steps[env_index].item())
            snapshot = self.pending.get(env_index)
            if snapshot is None or snapshot['motion_frame'] != frame:
                snapshot = self._terminal_snapshot(env_index)
                self.pending[env_index] = snapshot
            value = self._list(measure[env_index])
            if body_names is None:
                snapshot['termination_measurements'][term_name] = value
            else:
                snapshot['termination_measurements'][term_name] = {
                    name: float(item) for name, item in zip(body_names, value, strict=True)
                }

    def history_snapshots(self, active: torch.Tensor, episode_steps: torch.Tensor) -> list[dict | None]:
        command = self.command
        result: list[dict | None] = [None] * command.num_envs
        for env_index_tensor in torch.where(active)[0]:
            env_index = int(env_index_tensor.item())
            ee_reference = command.body_pos_relative_w[env_index, self.ee_body_indexes, 2]
            ee_robot = command.robot_body_pos_w[env_index, self.ee_body_indexes, 2]
            anchor_delta = command.anchor_pos_w[env_index] - command.robot_anchor_pos_w[env_index]
            action = self.latest_actions[env_index] if self.latest_actions is not None else None
            result[env_index] = {
                'motion_frame': int(command.time_steps[env_index].item()),
                'episode_step': int(episode_steps[env_index].item()),
                'anchor_position_error_xyz': self._list(anchor_delta),
                'anchor_rotation_error': float(command.metrics['error_anchor_rot'][env_index].item()),
                'ee_body_z_error': {
                    name: float(value)
                    for name, value in zip(
                        self.ee_body_names,
                        self._list(torch.abs(ee_reference - ee_robot)),
                        strict=True,
                    )
                },
                'joint_position_error_norm': float(command.metrics['error_joint_pos'][env_index].item()),
                'joint_velocity_error_norm': float(command.metrics['error_joint_vel'][env_index].item()),
                'policy_action_max_abs': float(torch.abs(action).max().item()) if action is not None else None,
            }
        return result

    def set_actions(self, actions: torch.Tensor) -> None:
        self.latest_actions = actions.detach()

    def close(self) -> None:
        if getattr(self.command, '_evaluation_failure_recorder', None) is self:
            delattr(self.command, '_evaluation_failure_recorder')

    def pop(self, env_index: int) -> dict | None:
        return self.pending.pop(env_index, None)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_path() -> Path:
    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required for per-motion evaluation")
    path = Path(retrieve_file_path(args_cli.checkpoint)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    data_root = Path(os.environ.get("GPUFREE_DATA_ROOT", "/root/gpufree-data")).resolve()
    if output_dir == data_root or data_root not in output_dir.parents:
        raise ValueError(f"Evaluation output must be a child of the data disk {data_root}: {output_dir}")
    if (output_dir / "results.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed evaluation: {output_dir}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=len(selected_indices),
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = args_cli.seed
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    checkpoint = _checkpoint_path()
    checkpoint_for_runner = handle_deprecated_rsl_rl_checkpoint(str(checkpoint), installed_version)

    env = None
    terminal_recorder = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        raw_env = env.unwrapped
        command = raw_env.command_manager.get_term("motion")
        if not isinstance(command, MotionCommand):
            raise TypeError(f"Expected MotionCommand, got {type(command).__name__}")
        if command.motion.clip_ids != manifest_ids:
            raise ValueError("Loaded motion order differs from the evaluation manifest")
        assignment = torch.tensor(selected_indices, dtype=torch.long, device=raw_env.device)
        command.set_evaluation_motion_ids(assignment)

        trace_buffer = None
        terminal_recorder = None
        if args_cli.failure_trace_frames > 0:
            ee_body_names = list(raw_env.cfg.terminations.ee_body_pos.params['body_names'])
            trace_buffer = FailureTraceBuffer(
                len(selected_indices),
                args_cli.failure_trace_frames,
                args_cli.failure_trace_max_episodes,
            )
            terminal_recorder = _TerminalFailureRecorder(command, raw_env, ee_body_names)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint_for_runner)
        policy = runner.get_inference_policy(device=raw_env.device)

        obs = env.get_observations()
        if metadata.version("rsl-rl-lib").startswith("2.3."):
            obs, _ = env.get_observations()

        count = len(selected_indices)
        device = raw_env.device
        episodes = torch.zeros(count, dtype=torch.long, device=device)
        successes = torch.zeros_like(episodes)
        episode_steps = torch.zeros_like(episodes)
        completion_sums = torch.zeros(count, dtype=torch.float64, device=device)
        reward_sums = torch.zeros(count, dtype=torch.float64, device=device)
        metric_steps = torch.zeros_like(episodes)
        metric_sums = {
            name: torch.zeros(count, dtype=torch.float64, device=device) for name in ERROR_METRICS
        }
        termination_counts = {
            name: torch.zeros(count, dtype=torch.long, device=device) for name in TERMINATION_TERMS
        }
        clip_lengths = command.motion.clip_lengths[assignment]
        max_steps = args_cli.episodes_per_motion * int(clip_lengths.max().item()) + 50
        global_step = 0

        while not bool(torch.all(episodes >= args_cli.episodes_per_motion)):
            if global_step >= max_steps:
                unfinished = torch.where(episodes < args_cli.episodes_per_motion)[0].detach().cpu().tolist()
                raise RuntimeError(f"Evaluation exceeded {max_steps} steps; unfinished envs={unfinished}")
            active = episodes < args_cli.episodes_per_motion
            command._update_metrics()
            for name in ERROR_METRICS:
                metric_sums[name] += command.metrics[name].double() * active
            metric_steps += active.long()
            episode_steps += active.long()

            with torch.inference_mode():
                actions = policy(obs)
                if terminal_recorder is not None and trace_buffer is not None:
                    terminal_recorder.set_actions(actions)
                    trace_buffer.append(terminal_recorder.history_snapshots(active, episode_steps))
                obs, rewards, dones, _ = env.step(actions)
                done_mask = dones.bool() & active
                reward_sums += rewards.double() * active
                if packaging_version.parse(installed_version) >= packaging_version.parse("4.0.0"):
                    policy.reset(dones)

            if bool(torch.any(done_mask)):
                motion_end = raw_env.termination_manager.get_term("motion_end").bool()
                successful = done_mask & motion_end & ~raw_env.reset_terminated.bool()
                successes += successful.long()
                fraction = torch.minimum(episode_steps.double() / clip_lengths.double(), torch.ones(count, device=device))
                completion_sums += fraction * done_mask
                cause_masks = {}
                for name in TERMINATION_TERMS:
                    cause = raw_env.termination_manager.get_term(name).bool()
                    cause_masks[name] = cause
                    termination_counts[name] += (cause & done_mask).long()
                if terminal_recorder is not None and trace_buffer is not None:
                    for env_index_tensor in torch.where(done_mask)[0]:
                        env_index = int(env_index_tensor.item())
                        is_successful = bool(successful[env_index].item())
                        causes = [name for name, mask in cause_masks.items() if bool(mask[env_index].item())]
                        terminal = terminal_recorder.pop(env_index)
                        trace_buffer.complete(
                            env_index=env_index,
                            motion_index=selected_indices[env_index],
                            motion_id=manifest_ids[selected_indices[env_index]],
                            episode=int(episodes[env_index].item()) + 1,
                            successful=is_successful,
                            terminal_snapshot=terminal,
                            termination_causes=causes,
                        )
                episodes += done_mask.long()
                episode_steps[done_mask] = 0

            global_step += 1
            if global_step % 250 == 0:
                completed = int(episodes.sum().item())
                target = count * args_cli.episodes_per_motion
                print(f"[EVAL] step={global_step} episodes={completed}/{target}", flush=True)

        selected_clip_ids = [command.motion.clip_ids[index] for index in selected_indices]
        selected_clip_lengths = clip_lengths.detach().cpu().tolist()
        summary, rows = build_report(
            selected_indices=selected_indices,
            clip_ids=selected_clip_ids,
            clip_lengths=selected_clip_lengths,
            fps=float(command.motion.fps),
            episodes=episodes.detach().cpu().tolist(),
            successes=successes.detach().cpu().tolist(),
            completion_sums=completion_sums.detach().cpu().tolist(),
            reward_sums=reward_sums.detach().cpu().tolist(),
            metric_steps=metric_steps.detach().cpu().tolist(),
            metric_sums={name: value.detach().cpu().tolist() for name, value in metric_sums.items()},
            termination_counts={
                name: value.detach().cpu().tolist() for name, value in termination_counts.items()
            },
        )
        payload = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": args_cli.task,
            "seed": args_cli.seed,
            "episodes_per_motion": args_cli.episodes_per_motion,
            "sim_steps": global_step,
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "selected_motion_indices": selected_indices,
            "summary": summary,
            "motions": rows,
        }
        failure_trace_payload = None
        if trace_buffer is not None:
            cause_counts: dict[str, int] = {}
            motion_counts: dict[str, int] = {}
            for record in trace_buffer.records:
                motion_id = record['motion_id']
                motion_counts[motion_id] = motion_counts.get(motion_id, 0) + 1
                for cause in record['termination_causes']:
                    cause_counts[cause] = cause_counts.get(cause, 0) + 1
            total_failures = int(summary['episodes'] - summary['successes'])
            failure_trace_payload = {
                'schema_version': 1,
                'created_at_utc': payload['created_at_utc'],
                'checkpoint': payload['checkpoint'],
                'checkpoint_sha256': payload['checkpoint_sha256'],
                'manifest': payload['manifest'],
                'manifest_sha256': payload['manifest_sha256'],
                'history_frames': trace_buffer.history_frames,
                'max_records': args_cli.failure_trace_max_episodes,
                'records_captured': len(trace_buffer.records),
                'total_failed_episodes': total_failures,
                'truncated': len(trace_buffer.records) < total_failures,
                'joint_names': list(command.cfg.joint_names),
                'body_names': list(command.cfg.body_names),
                'ee_body_names': list(terminal_recorder.ee_body_names),
                'cause_counts': cause_counts,
                'motion_counts': motion_counts,
                'records': trace_buffer.records,
            }
            payload['failure_trace'] = {
                'file': 'failure_traces.json',
                'records_captured': len(trace_buffer.records),
                'total_failed_episodes': total_failures,
                'truncated': failure_trace_payload['truncated'],
            }
        atomic_write_reports(output_dir, payload, rows, failure_trace_payload)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        print(f"[EVAL] results={output_dir / 'results.json'}", flush=True)
    finally:
        if terminal_recorder is not None:
            terminal_recorder.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
