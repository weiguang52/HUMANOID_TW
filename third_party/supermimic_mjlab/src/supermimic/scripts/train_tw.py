"""Train a SuperMimic motion-tracking policy in mjlab."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import tyro

from mjlab.envs import ManagerBasedRlEnv
from supermimic.config.tw.velocity_limits import (
    enable_tw_velocity_limits,
    disable_tw_velocity_limits,
    velocity_projection_stats,
)
from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.trainer import SuperMimicTrainer
from supermimic.config.tw.env_cfgs import (
    tw_supermimic_env_cfg,
)
from supermimic.config.tw.env_cfgs import (
    tw_supermimic_cfg,
)
from mjlab.utils.torch import configure_torch_backends


@dataclass
class TrainArgs:
    manifest: str | None = None
    motion_dir: str | None = None
    motion_files: tuple[str, ...] = ()
    num_envs: int = 1024
    episode_length_s: float = 30.0
    device: str = "cuda:0"
    seed: int = 0
    log_dir: str = "logs/supermimic"
    experiment_name: str = "tw_supermimic"
    resume: str | None = None
    resume_context_migration: bool = False
    resume_replay_migration: bool = False
    sm: SuperMimicCfg = field(default_factory=tw_supermimic_cfg)


def find_motion_files(
    motion_dir: str | None, motion_files: tuple[str, ...]
) -> tuple[str, ...]:
    if motion_files:
        return motion_files
    if motion_dir is None:
        raise ValueError("Provide --motion-dir or --motion-files.")
    root = Path(motion_dir)
    index = root / "index.json"
    if index.exists():
        import json

        with open(index) as f:
            manifest = json.load(f)
        paths = tuple(str(root / f"{clip['name']}.npz") for clip in manifest["clips"])
        return tuple(path for path in paths if Path(path).exists())
    paths = sorted(str(p) for p in root.glob("*.npz"))
    if not paths:
        raise ValueError(f"No .npz files found in {root}.")
    return tuple(paths)


def resolve_motion_files(args: TrainArgs) -> tuple[str, ...]:
    if args.manifest is not None:
        import json

        if args.motion_dir is not None or args.motion_files:
            raise ValueError("Use either --manifest or explicit motion inputs")
        data = json.loads(Path(args.manifest).read_text())
        if data.get("joint_coordinate_contract") != "tw123_yaw_waist_v3":
            raise ValueError("TW manifest has a stale coordinate contract")
        paths = tuple(row["file"] for row in data["motions"])
        if not paths or any(not Path(p).is_file() for p in paths):
            raise ValueError("TW manifest is empty or contains missing files")
        return paths
    return find_motion_files(args.motion_dir, args.motion_files)


def timestamped_name(name: str, when: datetime | None = None) -> str:
    """Append a run timestamp so each training run gets its own log directory."""
    stamp = (when or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{name}/{stamp}"


def main() -> None:
    args = tyro.cli(TrainArgs)
    configure_torch_backends()
    files = resolve_motion_files(args)
    if len(files) == 0:
        raise ValueError("No motion files resolved.")
    print(f"[INFO] Using {len(files)} motion clips.", flush=True)

    env_cfg = tw_supermimic_env_cfg(files, num_envs=args.num_envs, play=False)
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.seed = args.seed

    import hashlib
    import json
    import os
    from supermimic.config.tw.velocity_limits import LIMITS, PHYSICS_CONTRACT

    cfg = args.sm
    cfg.extra.update(
        {
            "tw_joint_coordinate_contract": "tw123_yaw_waist_v3",
            "tw_mjcf_sha256": hashlib.sha256(
                Path(os.environ["TW_SUPERMIMIC_MJCF"]).read_bytes()
            ).hexdigest(),
            "tw_motion_input_sha256": hashlib.sha256(
                json.dumps(files).encode()
            ).hexdigest(),
            "tw_joint_speed_limits": LIMITS,
            "tw_physics_contract": PHYSICS_CONTRACT,
            "tw_reward_scales": {
                key: value
                for key, value in vars(cfg).items()
                if key.startswith("reward_")
            },
            "tw_context_layout": "mean_friction_encoder_bias_com_v1",
            "tw_replay_window_contract": "episode_contiguous_pool_v2",
            "tw_history_contract": "past_only_shared_bias_reset_padding_v2",
            "tw_runtime": "mjlab1.6.0/mujoco3.11.0/warp1.18.0/torch2.7.0+cu128",
        }
    )
    if args.resume is not None:
        import torch

        previous = torch.load(args.resume, map_location="cpu", weights_only=False)
        previous_extra = previous["cfg"].get("extra", {})
        for key in (
            "tw_joint_coordinate_contract",
            "tw_mjcf_sha256",
            "tw_motion_input_sha256",
            "tw_joint_speed_limits",
            "tw_physics_contract",
            "tw_reward_scales",
        ):
            if previous_extra.get(key) != cfg.extra[key]:
                raise ValueError(f"TW resume contract mismatch: {key}")
        if previous_extra.get("tw_context_layout") != cfg.extra["tw_context_layout"]:
            if (
                not args.resume_context_migration
                or previous_extra.get("tw_context_layout") is not None
            ):
                raise ValueError(
                    "Context layout changed; explicit legacy context migration required"
                )
            cfg.extra["tw_context_migration"] = {
                "source_checkpoint_sha256": hashlib.sha256(
                    Path(args.resume).read_bytes()
                ).hexdigest(),
                "source_iteration": int(previous["iteration"]),
                "previous_bias_offset": 2,
                "corrected_bias_offset": 1,
                "note": "Preserve learned parameters/optimizers; correct legacy bias/COM slice. Replay and simulator rebuilt.",
            }
        if "tw_context_migration" in previous_extra:
            cfg.extra.setdefault(
                "tw_context_migration", previous_extra["tw_context_migration"]
            )
        if (
            previous_extra.get("tw_replay_window_contract")
            != cfg.extra["tw_replay_window_contract"]
        ):
            if (
                not args.resume_replay_migration
                or previous_extra.get("tw_replay_window_contract") is not None
            ):
                raise ValueError(
                    "Replay contract changed; explicit legacy replay migration required"
                )
            cfg.extra["tw_replay_migration"] = {
                "source_checkpoint_sha256": hashlib.sha256(
                    Path(args.resume).read_bytes()
                ).hexdigest(),
                "source_iteration": int(previous["iteration"]),
                "note": "Keep networks/optimizers; rebuild replay using full-length episode-contiguous windows only.",
            }
        elif "tw_replay_migration" in previous_extra:
            cfg.extra["tw_replay_migration"] = previous_extra["tw_replay_migration"]
        if (
            previous_extra.get("tw_history_contract")
            != cfg.extra["tw_history_contract"]
        ):
            if (
                not args.resume_replay_migration
                or previous_extra.get("tw_history_contract") is not None
            ):
                raise ValueError(
                    "Replay observation-history contract changed; explicit migration required"
                )
            cfg.extra["tw_history_migration"] = {
                "source_checkpoint_sha256": hashlib.sha256(
                    Path(args.resume).read_bytes()
                ).hexdigest(),
                "source_iteration": int(previous["iteration"]),
                "note": "Past-only actor history, shared encoder-bias accessor, reset/oldest-state padding; no duplicate current observation.",
            }
        elif "tw_history_migration" in previous_extra:
            cfg.extra["tw_history_migration"] = previous_extra["tw_history_migration"]
        del previous
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.log_dir = args.log_dir
    cfg.experiment_name = timestamped_name(args.experiment_name)
    print(f"[INFO] Logging to {Path(cfg.log_dir) / cfg.experiment_name}", flush=True)
    if cfg.lr_decay_iters > cfg.num_iterations:
        print(
            f"[INFO] lr_decay_iters={cfg.lr_decay_iters} > num_iterations="
            f"{cfg.num_iterations}; annealing over the run length instead.",
            flush=True,
        )
        cfg.lr_decay_iters = cfg.num_iterations

    env = ManagerBasedRlEnv(env_cfg, device=args.device)
    enable_tw_velocity_limits(env)
    cfg.extra["tw_action_joint_names"] = list(
        env.action_manager.get_term("joint_pos").target_names
    )
    cfg.extra["tw_state_joint_names"] = list(env.scene["robot"].joint_names)
    if args.resume is not None:
        for key in ("tw_action_joint_names", "tw_state_joint_names"):
            if previous_extra.get(key) != cfg.extra[key]:
                raise ValueError(f"TW resume ordering mismatch: {key}")
    trainer = SuperMimicTrainer(env, cfg, log_dir=cfg.log_dir)
    if args.resume is not None:
        trainer.load(args.resume)
    try:
        trainer.train()
        trainer.save("checkpoint_final.pt")
        print("TW_PROJECTION_STATS", velocity_projection_stats(env), flush=True)
    except KeyboardInterrupt:
        print("[INFO] Interrupted; saving checkpoint_interrupt.pt ...", flush=True)
        trainer.save("checkpoint_interrupt.pt")
    finally:
        trainer.close()
        env.close()
        disable_tw_velocity_limits(env)


if __name__ == "__main__":
    main()
