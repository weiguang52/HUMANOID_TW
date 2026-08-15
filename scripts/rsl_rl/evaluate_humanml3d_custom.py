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
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.episodes_per_motion <= 0:
    parser.error("--episodes_per_motion must be positive")
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
                for name in TERMINATION_TERMS:
                    cause = raw_env.termination_manager.get_term(name).bool()
                    termination_counts[name] += (cause & done_mask).long()
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
        atomic_write_reports(output_dir, payload, rows)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        print(f"[EVAL] results={output_dir / 'results.json'}", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
