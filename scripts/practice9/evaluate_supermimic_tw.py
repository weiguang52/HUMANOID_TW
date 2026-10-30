"""Fixed-manifest TW evaluation, stopped at first failure; colored front/side replays."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from PIL import Image, ImageDraw, ImageFont
from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.trainer import SuperMimicTrainer
from supermimic.algo.state import encoder_bias_from_context
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import (
    LIMITS,
    disable_tw_velocity_limits,
    enable_tw_velocity_limits,
)


def labelled(frame, text):
    image = Image.fromarray(frame)
    ImageDraw.Draw(image).rectangle((0, 0, image.width, 34), fill=(15, 15, 15))
    ImageDraw.Draw(image).text(
        (8, 5), text, fill=(255, 255, 255), font=ImageFont.load_default(size=22)
    )
    return np.asarray(image)


@torch.no_grad()
def evaluate(args):
    assert args.max_steps > 0
    assert not (args.output / "results.json").exists(), (
        "Refusing to overwrite completed evaluation"
    )
    manifest = json.loads(args.manifest.read_text())
    assert manifest["joint_coordinate_contract"] == "tw123_yaw_waist_v3"
    rows = manifest["motions"]
    assert rows and len({r["id"] for r in rows}) == len(rows)
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
    extra = checkpoint["cfg"].get("extra", {})
    assert extra.get("tw_joint_coordinate_contract") == "tw123_yaw_waist_v3"
    assert extra.get("tw_joint_speed_limits") == LIMITS
    asset = Path(os.environ["TW_SUPERMIMIC_MJCF"])
    color_audit = json.loads(asset.with_suffix(".color_audit.json").read_text())
    assert color_audit["passed"]
    assert (
        color_audit["colored_mjcf_sha256"]
        == hashlib.sha256(asset.read_bytes()).hexdigest()
    )
    assert color_audit["source_mjcf_sha256"] == extra["tw_mjcf_sha256"]
    cfg = SuperMimicCfg(**checkpoint["cfg"])
    cfg.compile = False
    cfg.experiment_name = "evaluation_runtime"
    cfg.device = "cuda:0"
    env_cfg = tw_supermimic_env_cfg([r["file"] for r in rows], num_envs=1, play=True)
    env_cfg.seed = args.seed
    env_cfg.events = {}
    env_cfg.episode_length_s = 60.0
    env_cfg.auto_reset = False
    env_cfg.commands["motion"].sampling_mode = "start"
    env_cfg.commands["motion"].pose_range = {}
    env_cfg.commands["motion"].velocity_range = {}
    env_cfg.commands["motion"].joint_position_range = (0.0, 0.0)
    env_cfg.viewer.width = 960
    env_cfg.viewer.height = 640
    env_cfg.viewer.distance = 1.15
    env_cfg.viewer.elevation = -10.0
    env_cfg.viewer.fovy = 45.0
    args.output.mkdir(parents=True, exist_ok=True)
    env = ManagerBasedRlEnv(
        env_cfg, device="cuda:0", render_mode="rgb_array" if args.video else None
    )
    enable_tw_velocity_limits(env)
    trainer = SuperMimicTrainer(env, cfg, log_dir=args.output)
    trainer.load(args.checkpoint)
    trainer.initialize()
    trainer.actor_critic.eval()
    command = trainer.command
    reports = []
    try:
        for agent in args.agents:
            for row in rows:
                env.reset()
                clip = trainer.library.clip_names.index(Path(row["file"]).stem)
                command.reset_to_frame(torch.tensor([0], device="cuda:0"), clip, 0)
                env.sim.forward()
                trainer.last_action.zero_()
                state = trainer.extractor.state()
                context = trainer.extractor.context()
                obs = trainer.proprio(
                    state,
                    trainer.last_action,
                    encoder_bias_from_context(context, trainer.spec.num_joints),
                    add_noise=False,
                )
                trainer.history[:] = obs[:, None, :].expand_as(trainer.history)
                q = env.scene["robot"].data.root_link_quat_w[0].cpu().numpy()
                yaw = math.atan2(
                    2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)
                )
                planned_steps = min(
                    args.max_steps, int(trainer.library.clip_lengths_t[clip]) - 1
                )
                frames = {"front": [], "side": []}
                samples = []
                failure = False
                truncation = False
                reasons = []
                for step in range(planned_steps):
                    state = trainer.extractor.state()
                    context = trainer.extractor.context()
                    obs = trainer.proprio(
                        state,
                        trainer.last_action,
                        encoder_bias_from_context(context, trainer.spec.num_joints),
                        add_noise=False,
                    )
                    trainer.history = torch.cat(
                        [trainer.history[:, 1:], obs[:, None]], dim=1
                    )
                    action = (
                        trainer.actor_critic.act(
                            trainer.history.reshape(1, -1), command.goal()
                        )
                        if agent == "trained"
                        else torch.zeros((1, 30), device="cuda:0")
                    )
                    _, _, terminated, truncated, _ = env.step(action)
                    assert torch.isfinite(action).all()
                    assert torch.isfinite(command.robot_body_pos_w).all()
                    assert torch.isfinite(command.robot_joint_vel).all()
                    trainer.last_action = action
                    pos_error = torch.linalg.vector_norm(
                        command.robot_body_pos_w - command.body_pos_w, dim=-1
                    )[0]
                    joint_error = (command.robot_joint_pos - command.joint_pos).abs()[0]
                    dot = (
                        (command.robot_anchor_quat_w * command.anchor_quat_w)
                        .sum(-1)
                        .abs()
                        .clamp(0.0, 1.0)
                    )
                    samples.append(
                        {
                            "step": step + 1,
                            "reference_frame": int(command.time_steps[0]),
                            "body_error_m": pos_error.cpu().tolist(),
                            "joint_error_rad": joint_error.cpu().tolist(),
                            "anchor_error_m": float(
                                torch.linalg.vector_norm(
                                    command.robot_anchor_pos_w - command.anchor_pos_w
                                )
                            ),
                            "anchor_angle_rad": float(2 * torch.acos(dot)[0]),
                            "peak_speed_rad_s": float(
                                command.robot_joint_vel.abs().max()
                            ),
                        }
                    )
                    limits = torch.tensor(
                        [LIMITS[n] for n in env.scene["robot"].joint_names],
                        device="cuda:0",
                    )
                    assert torch.all(command.robot_joint_vel.abs() <= limits + 1e-5)
                    if args.video:
                        renderer = env._offline_renderer
                        cam = renderer._cam
                        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                        root = env.scene["robot"].data.root_link_pos_w[0].cpu().numpy()
                        cam.lookat[:] = [root[0], root[1], max(0.22, root[2] + 0.03)]
                        for view, offset in [("front", 180.0), ("side", 90.0)]:
                            cam.azimuth = math.degrees(yaw) + offset
                            frame = env.render()
                            assert frame is not None
                            frames[view].append(
                                labelled(
                                    frame,
                                    f"{agent} | {row['id']} | {view} | t={(step + 1) * env.step_dt:.2f}s",
                                )
                            )
                    failure = bool(terminated[0])
                    truncation = bool(truncated[0])
                    if failure or truncation:
                        manager = env.termination_manager
                        reasons = [
                            name
                            for name in manager.active_terms
                            if bool(manager.get_term(name)[0])
                        ]
                        break
                duration = len(samples) * env.step_dt
                result = {
                    "id": row["id"],
                    "agent": agent,
                    "steps": len(samples),
                    "duration_s": duration,
                    "planned_steps": planned_steps,
                    "planned_duration_s": planned_steps * env.step_dt,
                    "terminated": failure,
                    "truncated": truncation,
                    "termination_reasons": reasons,
                    "completed_requested_window": (
                        len(samples) == planned_steps and not failure
                    ),
                    "mean_body_error_m": float(
                        np.mean([s["body_error_m"] for s in samples])
                    ),
                    "mean_joint_error_rad": float(
                        np.mean([s["joint_error_rad"] for s in samples])
                    ),
                    "mean_anchor_error_m": float(
                        np.mean([s["anchor_error_m"] for s in samples])
                    ),
                    "mean_anchor_angle_rad": float(
                        np.mean([s["anchor_angle_rad"] for s in samples])
                    ),
                    "body_names": list(command.cfg.body_names),
                    "joint_names": list(env.scene["robot"].joint_names),
                    "camera": "Translation-following; front starts along initial robot +X; side along initial -Y. Both remain world-facing at initial heading during turns.",
                    "motion_file": row["file"],
                    "sample_file": f"{row['id']}_{agent}_samples.json",
                }
                (args.output / result["sample_file"]).write_text(
                    json.dumps(samples) + "\n"
                )
                if args.video:
                    import mediapy

                    for view, view_frames in frames.items():
                        path = args.output / f"{row['id']}_{agent}_{view}.mp4"
                        mediapy.write_video(str(path), view_frames, fps=50)
                        result[view + "_video"] = path.name
                reports.append(result)
                print(json.dumps(result), flush=True)
                (args.output / "results.partial.json").write_text(
                    json.dumps(reports, indent=2) + "\n"
                )
        report = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": hashlib.sha256(
                args.checkpoint.read_bytes()
            ).hexdigest(),
            "manifest": str(args.manifest),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "seed": args.seed,
            "max_steps": args.max_steps,
            "rows": reports,
            "note": "No restart after failure, no repeated best-of selection. Errors cover actual duration only; short episodes cannot be compared as 20-second drift. MuJoCo results are not direct PhysX policy comparisons.",
        }
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        trainer.close()
        env.close()
        disable_tw_velocity_limits(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--agents", nargs="+", choices=["trained", "zero"], default=["trained", "zero"]
    )
    parser.add_argument("--video", action="store_true")
    evaluate(parser.parse_args())
