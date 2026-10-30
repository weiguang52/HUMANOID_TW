"""Check actual TW Warp stepping, joint-speed caps, and finite state; not policy quality."""

import argparse
import json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import (
    LIMITS,
    enable_tw_velocity_limits,
    disable_tw_velocity_limits,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert manifest["joint_coordinate_contract"] == "tw123_yaw_waist_v3"
    motion = manifest["motions"][0]
    cfg = tw_supermimic_env_cfg([motion["file"]], num_envs=args.num_envs, play=True)
    cfg.seed = 42
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    maximum, terminations = 0.0, 0
    try:
        enable_tw_velocity_limits(env)
        assert abs(env.step_dt - 0.02) < 1e-12
        assert abs(env.sim.mj_model.opt.timestep - 0.005) < 1e-12
        env.reset()
        robot = env.scene["robot"]
        limits = torch.tensor([LIMITS[n] for n in robot.joint_names], device="cuda:0")
        for _ in range(args.steps):
            result = env.step(torch.zeros((args.num_envs, 30), device="cuda:0"))
            assert torch.isfinite(robot.data.joint_pos).all()
            assert torch.isfinite(robot.data.joint_vel).all()
            assert torch.isfinite(robot.data.root_link_pos_w).all()
            maximum = max(maximum, float(robot.data.joint_vel.abs().max()))
            assert torch.all(robot.data.joint_vel.abs() <= limits + 1e-5)
            terminations += int(result[2].sum())
            done = result[2] | result[3]
            if done.any():
                env.reset(env_ids=done.nonzero().flatten())
        report = dict(
            passed=True,
            motion=motion["id"],
            num_envs=args.num_envs,
            control_steps=args.steps,
            physics_dt=0.005,
            control_dt=0.02,
            joint_count=30,
            joint_speed_caps=LIMITS,
            max_joint_velocity_rad_s=maximum,
            termination_events=terminations,
            torch=torch.__version__,
            scope="Zero-action runtime/constraint check; not policy-quality acceptance",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()
        disable_tw_velocity_limits(env)


if __name__ == "__main__":
    main()
