"""Continuous synthetic static/impulse diagnostics; not learned-policy validation."""

import argparse, json, hashlib, time, os
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import (
    LIMITS,
    enable_tw_velocity_limits,
    disable_tw_velocity_limits,
    velocity_projection_stats,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--initial-joint-speed", type=float, default=0.0)
    a = p.parse_args()
    assert not a.output.exists(), "Preserve existing diagnostic evidence"
    manifest = json.loads(a.manifest.read_text())
    assert manifest["synthetic_diagnostic"]
    cfg = tw_supermimic_env_cfg([manifest["motions"][0]["file"]], num_envs=1, play=True)
    cfg.events = {}
    cfg.seed = 42
    cfg.commands["motion"].pose_range = {}
    cfg.commands["motion"].velocity_range = {}
    cfg.commands["motion"].joint_position_range = (0.0, 0.0)
    torch.backends.cuda.matmul.allow_tf32 = True
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    enable_tw_velocity_limits(env)
    roots = []
    velocities = []
    done_reason = []
    terminated = False
    peak = 0.0
    started = time.monotonic()
    try:
        env.reset()
        env.command_manager.get_term("motion").reset_to_frame(
            torch.tensor([0], device="cuda:0"), 0, 0
        )
        env.sim.forward()
        robot = env.scene["robot"]
        torch.testing.assert_close(
            robot.data.joint_pos, robot.data.default_joint_pos, atol=1e-6, rtol=0.0
        )
        action = torch.zeros((1, 30), device="cuda:0")
        env.action_manager.process_action(action)
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.forward()
        initial_force = float(env.sim.data.actuator_force.abs().max())
        assert initial_force < 1e-4, initial_force
        if a.initial_joint_speed:
            speed = torch.tensor(
                [[a.initial_joint_speed * (-1 if j % 2 else 1) for j in range(30)]],
                device="cuda:0",
            )
            robot.write_joint_state_to_sim(robot.data.joint_pos.clone(), speed)
            env.sim.forward()
        roots.append(env.sim.data.qpos[0, :7].cpu().numpy().copy())
        caps = torch.tensor([LIMITS[n] for n in robot.joint_names], device="cuda:0")
        for step in range(a.steps):
            result = env.step(action)
            roots.append(env.sim.data.qpos[0, :7].cpu().numpy().copy())
            velocities.append(env.sim.data.qvel[0].cpu().numpy().copy())
            assert np.isfinite(roots[-1]).all() and np.isfinite(velocities[-1]).all()
            peak = max(peak, float(robot.data.joint_vel.abs().max()))
            assert torch.all(robot.data.joint_vel.abs() <= caps + 1e-5)
            terminated = bool(result[2][0])
            if bool((result[2] | result[3])[0]):
                done_reason = [
                    n
                    for n in env.termination_manager.active_terms
                    if bool(env.termination_manager.get_term(n)[0])
                ]
                break
            if (step + 1) % 250 == 0:
                print("CALIBRATION_STEP", step + 1, flush=True)
        roots = np.stack(roots)
        velocities = np.stack(velocities)
        actual = len(velocities)
        height = float(np.abs(roots[:, 2] - roots[0, 2]).max())
        horizontal = float(np.linalg.norm(roots[:, :2] - roots[0, :2], axis=-1).max())
        dot = np.abs(np.sum(roots[:, 3:7] * roots[0, 3:7], axis=-1)).clip(0, 1)
        angle = float((2 * np.arccos(dot)).max())
        stats = velocity_projection_stats(env)
        passed = (
            (
                not terminated
                and actual == a.steps
                and height < 0.02
                and horizontal < 0.02
                and angle < 0.1
            )
            if not a.initial_joint_speed
            else (stats["active_steps"] > 0)
        )
        report = dict(
            passed=passed,
            probe="forced_velocity_impulse"
            if a.initial_joint_speed
            else "synthetic_default_static",
            steps=actual,
            requested_steps=a.steps,
            duration_s=actual * env.step_dt,
            terminated=terminated,
            termination_reasons=done_reason,
            initial_hold_actuator_force_Nm=initial_force,
            max_height_change_m=height,
            max_horizontal_drift_m=horizontal,
            max_orientation_change_rad=angle,
            max_joint_speed_rad_s=peak,
            projection=stats,
            cuda_graph=env.sim.use_cuda_graph,
            wall_seconds=time.monotonic() - started,
            asset_sha256=hashlib.sha256(
                Path(os.environ["TW_SUPERMIMIC_MJCF"]).read_bytes()
            ).hexdigest(),
            note="Synthetic calibration only. No HumanML3D source, no learned policy, never counts as test-motion success.",
        )
        a.output.mkdir(parents=True)
        np.savez_compressed(a.output / "trace.npz", root_qpos=roots, qvel=velocities)
        (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if not passed:
            raise RuntimeError(
                "Physics calibration failed; preserve evidence and diagnose before training"
            )
    finally:
        env.close()
        disable_tw_velocity_limits(env)


if __name__ == "__main__":
    main()
