"""Frozen-policy first-episode reset/sensor/action factorial diagnostic."""

import argparse
import hashlib
import itertools
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.state import encoder_bias_from_context
from supermimic.algo.trainer import SuperMimicTrainer
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import (
    LIMITS,
    PHYSICS_CONTRACT,
    disable_tw_velocity_limits,
    enable_tw_velocity_limits,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def run_case(
    args, source, motion, profile, agent, action_noise, sensor_noise, initials
):
    name, perturb, sensors, actions = profile
    cfg = SuperMimicCfg(**source["cfg"])
    cfg.compile = False
    cfg.wm_anchor_collection = cfg.policy_anchor_collection = None
    cfg.experiment_name = f"{name}_{agent}"
    ec = tw_supermimic_env_cfg([motion], num_envs=args.num_envs, play=False)
    ec.seed = 42
    ec.events = {}
    ec.auto_reset = False
    ec.episode_length_s = 30.0
    cc = ec.commands["motion"]
    cc.sampling_mode = "start"
    ranges = {
        "pose": cc.pose_range,
        "velocity": cc.velocity_range,
        "joint": cc.joint_position_range,
    }
    if not perturb:
        cc.pose_range = {}
        cc.velocity_range = {}
        cc.joint_position_range = (0.0, 0.0)
    env = ManagerBasedRlEnv(ec, device="cuda:0")
    enable_tw_velocity_limits(env)
    t = None
    records = {
        key: [] for key in ["state", "next_state", "action", "context", "foot_tilt_deg"]
    }
    lengths = np.zeros(args.num_envs, dtype=int)
    reasons = [[] for _ in range(args.num_envs)]
    parking_errors = []
    hook = None
    try:
        t = SuperMimicTrainer(env, cfg, log_dir=args.output / "runtime")
        t.load(args.checkpoint, evaluation_only=True)
        t.initialize()
        t.actor_critic.eval()
        env.sim.forward()
        assert abs(env.step_dt - 0.02) < 1e-9
        assert list(t.spec.joint_names) == cfg.extra["tw_state_joint_names"]
        assert (
            list(env.action_manager.get_term("joint_pos").target_names)
            == cfg.extra["tw_action_joint_names"]
        )
        initial_clock = t.command.time_steps.cpu().clone()
        t.command.time_steps.zero_()
        t.command.update_relative_body_poses()
        assert bool((t.command.time_steps == 0).all())
        s0 = t.extractor.state().clone()
        ctx0 = t.extractor.context().clone()
        t.history[:] = t.proprio(
            s0, t.last_action, encoder_bias_from_context(ctx0, 30), add_noise=False
        )[:, None]
        if perturb in initials:
            torch.testing.assert_close(
                s0.cpu(), initials[perturb][0], rtol=0, atol=1e-6
            )
            torch.testing.assert_close(
                ctx0.cpu(), initials[perturb][1], rtol=0, atol=1e-7
            )
        else:
            initials[perturb] = (s0.cpu().clone(), ctx0.cpu().clone())
        np.savez_compressed(
            args.output / f"{name}_{agent}_initial.npz",
            state=s0.cpu().numpy(),
            context=ctx0.cpu().numpy(),
            clock_before_standardization=initial_clock.numpy(),
        )
        foot_ids = [
            list(t.command.cfg.body_names).index(n) for n in ["left_foot", "right_foot"]
        ]
        frozen = {k: v.clone() for k, v in t.actor_critic.state_dict().items()}
        wm_calls = [0]
        hook = t.world_model.register_forward_hook(
            lambda *unused: wm_calls.__setitem__(0, wm_calls[0] + 1)
        )
        caps = torch.tensor([LIMITS[n] for n in t.spec.joint_names], device="cuda:0")
        live = torch.ones(args.num_envs, device="cuda:0", dtype=torch.bool)
        for step in range(args.max_steps):
            s = t.extractor.state().clone()
            ctx = t.extractor.context().clone()
            obs = t.proprio(
                s, t.last_action, encoder_bias_from_context(ctx, 30), add_noise=False
            )
            if sensors:
                obs += torch.as_tensor(sensor_noise[step], device="cuda:0")
            t.history = torch.cat([t.history[:, 1:], obs[:, None]], dim=1)
            mean = t.actor_critic.act(
                t.history.reshape(args.num_envs, -1), t.command.goal()
            )
            u = mean if agent == "trained" else torch.zeros_like(mean)
            if actions:
                u = u + torch.as_tensor(action_noise[step], device="cuda:0")
            u[~live] = 0.0
            assert torch.isfinite(u).all()
            result = env.step(u)
            n = t.extractor.state().clone()
            q = t.command.robot_body_quat_w[:, foot_ids]
            torch.testing.assert_close(
                q.norm(dim=-1), torch.ones_like(q[..., 0]), rtol=0, atol=1e-5
            )
            tilt = torch.rad2deg(
                torch.acos(
                    (1 - 2 * (q[..., 1].square() + q[..., 2].square())).clamp(-1, 1)
                )
            )
            assert bool((n[:, 43:73].abs() <= caps + 1e-5).all())
            for key, value in [
                ("state", s),
                ("next_state", n),
                ("action", u),
                ("context", ctx),
                ("foot_tilt_deg", tilt),
            ]:
                assert torch.isfinite(value).all(), key
                records[key].append(value.cpu().numpy().copy())
            t.last_action = u.clone()
            done = (result[2] | result[3]) & live
            for i in done.nonzero().flatten().cpu().tolist():
                lengths[i] = step + 1
                reasons[i] = [
                    key
                    for key in env.termination_manager.active_terms
                    if bool(env.termination_manager.get_term(key)[i])
                ]
            live &= ~done
            if not bool(live.any()):
                break
            reset_ids = (result[2] | result[3]).nonzero().flatten()
            if len(reset_ids):
                active = n[live].clone()
                context = t.extractor.context()[live].clone()
                clock = t.command.time_steps[live].clone()
                episode_clock = env.episode_length_buf[live].clone()
                env.reset(env_ids=reset_ids)
                t.command.reset_to_frame(reset_ids, 0, 0)
                after = t.extractor.state().clone()
                torch.testing.assert_close(after[live], active, rtol=0, atol=1e-6)
                torch.testing.assert_close(
                    t.extractor.context()[live], context, rtol=0, atol=1e-7
                )
                assert torch.equal(t.command.time_steps[live], clock)
                assert torch.equal(env.episode_length_buf[live], episode_clock)
                parking_errors.append(float((after[live] - active).abs().max()))
                records["next_state"][-1][live.cpu().numpy()] = (
                    after[live].cpu().numpy()
                )
                t.last_action[~live] = 0.0
        assert wm_calls[0] == 0
        for key, value in frozen.items():
            assert torch.equal(t.actor_critic.state_dict()[key], value), key
        rows = []
        for i in range(args.num_envs):
            length = int(lengths[i]) if lengths[i] else len(records["state"])
            trace = {k: np.stack(v)[:length, i] for k, v in records.items()}
            assert np.array_equal(trace["next_state"][:-1], trace["state"][1:])
            filename = f"{name}_{agent}_env{i:02d}.npz"
            np.savez_compressed(args.output / filename, **trace)
            xy = np.linalg.norm(
                trace["next_state"][:, :2] - trace["state"][0, :2], axis=-1
            )
            row = {
                "profile": name,
                "agent": agent,
                "env_id": i,
                "steps": length,
                "seconds": length * 0.02,
                "termination_reasons": reasons[i],
                "trace": filename,
            }
            row.update(
                completed_requested_window=length == args.max_steps and not reasons[i],
                completed20s=length == 1000 and not reasons[i],
                root_xy_max_mm=float(xy.max() * 1000),
                foot_tilt_median_deg=np.median(trace["foot_tilt_deg"], axis=0).tolist(),
            )
            rows.append(row)
        summary = {
            "profile": name,
            "agent": agent,
            "episodes": args.num_envs,
            "completed_requested_window": sum(
                x["completed_requested_window"] for x in rows
            ),
            "completed20s": sum(x["completed20s"] for x in rows),
            "median_seconds": float(np.median([x["seconds"] for x in rows])),
            "min_seconds": min(x["seconds"] for x in rows),
            "max_seconds": max(x["seconds"] for x in rows),
            "parking_max_active_state_error": max(parking_errors, default=0.0),
            "reset_ranges": ranges
            if perturb
            else {"pose": {}, "velocity": {}, "joint": [0.0, 0.0]},
            "weights_frozen": True,
            "world_model_calls": 0,
        }
        return rows, summary
    except BaseException:
        if records["state"]:
            np.savez_compressed(
                args.output / f"{name}_{agent}_failed_arrays.npz",
                **{k: np.stack(v) for k, v in records.items()},
            )
        raise
    finally:
        if hook is not None:
            hook.remove()
        if t is not None:
            t.close()
        env.close()
        disable_tw_velocity_limits(env)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["manifest", "checkpoint", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--pd-repeat", action="store_true")
    args = p.parse_args()
    assert not (args.smoke and args.pd_repeat)
    assert not args.output.exists(), "Preserve previous attempts"
    assert (args.smoke and args.num_envs == 4 and args.max_steps == 40) or (
        not args.smoke and args.num_envs == 32 and args.max_steps == 1000
    )
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert source["kind"] == "real_physics_ppo_diagnostic_policy_v1"
    assert source["ppo_update"] == source["plan"]["updates"] == 1000
    extra = source["cfg"]["extra"]
    assert extra["tw_physics_contract"] == PHYSICS_CONTRACT
    assert digest(os.environ["TW_SUPERMIMIC_MJCF"]) == extra["tw_mjcf_sha256"]
    manifest = json.loads(args.manifest.read_text())
    assert manifest["synthetic_diagnostic"] and len(manifest["motions"]) == 1
    motion = manifest["motions"][0]["file"]
    with np.load(motion, allow_pickle=False) as reference:
        for key in ["joint_pos", "body_pos_w", "body_quat_w"]:
            assert np.allclose(reference[key], reference[key][0], rtol=0, atol=1e-7), (
                key
            )
        for key in ["joint_vel", "body_lin_vel_w", "body_ang_vel_w"]:
            assert np.max(np.abs(reference[key])) < 1e-7, key
    assert digest(args.manifest) == source["plan"]["manifest_sha256"]
    assert digest(motion) == source["plan"]["motion_sha256"][motion]
    tree = ET.parse(os.environ["TW_SUPERMIMIC_MJCF"]).getroot()
    for foot in ["left", "right"]:
        geom = tree.find(
            f".//body[@name='{foot}_foot']/geom[@name='{foot}_foot_collision']"
        )
        assert geom is not None and geom.get("type") == "box"
        assert geom.get("quat", "1 0 0 0") == "1 0 0 0"
    c = source["cfg"]
    sensor_scale = np.array(
        [c["sensor_noise_joint_pos"]] * 30
        + [c["sensor_noise_joint_vel"]] * 30
        + [c["sensor_noise_base_ang_vel"]] * 3
        + [c["sensor_noise_projected_gravity"]] * 3
        + [0.0] * 30,
        dtype=np.float32,
    )
    shape = (args.max_steps, args.num_envs)
    action_noise = (
        np.random.default_rng(10042).normal(size=(*shape, 30)).astype(np.float32)
        * c["action_noise"]
    )
    sensor_noise = (
        np.random.default_rng(20042)
        .uniform(-1, 1, size=(*shape, 96))
        .astype(np.float32)
        * sensor_scale
    )
    assert not sensor_noise[:, :, 66:].any()
    profiles = [
        (f"r{int(r)}_s{int(s)}_a{int(a)}", r, s, a)
        for r, s, a in itertools.product([False, True], repeat=3)
    ]
    if args.smoke:
        profiles = [profiles[0], profiles[-1]]
    if args.pd_repeat:
        profiles = [x for x in profiles if not x[2]]
    agents = ["zero"] if args.pd_repeat else ["zero", "trained"]
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "fixed_noise.npz",
        action_noise=action_noise,
        sensor_noise=sensor_noise,
        sensor_scale=sensor_scale,
    )
    plan = {
        "mode": "interface_smoke_not_acceptance"
        if args.smoke
        else "first20s_factorial_development_diagnostic",
        "seed": 42,
        "action_noise_seed": 10042,
        "sensor_noise_seed": 20042,
        "sensor_noise_distribution": "Independent uniform, same source amplitudes; fixed NumPy draws, no simulator RNG consumption",
        "profiles": profiles,
        "agents": agents,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "checkpoint_sha256": digest(args.checkpoint),
        "manifest_sha256": digest(args.manifest),
        "motion_sha256": digest(motion),
        "source_sha256": digest(__file__),
        "physics_contract": PHYSICS_CONTRACT,
        "events_disabled": True,
        "policy_frozen": True,
        "scope": "One training seed42, synthetic static reference, not independent motions or three training seeds; first episode only; original termination thresholds; dead worlds parked with live-state assertions",
    }
    if args.pd_repeat:
        plan["mode"] = "pd_numerical_repeat_control"
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    (args.output / "source_config.json").write_text(
        json.dumps(asdict(SuperMimicCfg(**source["cfg"])), indent=2)
    )
    initials = {}
    rows = []
    summaries = []
    for profile in profiles:
        for agent in agents:
            group, summary = run_case(
                args,
                source,
                motion,
                profile,
                agent,
                action_noise,
                sensor_noise,
                initials,
            )
            rows.extend(group)
            summaries.append(summary)
            (args.output / "partial.json").write_text(
                json.dumps({"rows": rows, "summary": summaries}, indent=2)
            )
            print(json.dumps(summary), flush=True)
    (args.output / "results.json").write_text(
        json.dumps({"plan": plan, "rows": rows, "summary": summaries}, indent=2)
    )
    print("FACTORIAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
