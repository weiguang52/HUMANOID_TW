"""Fixed static PD root-pose state branches, explicitly not trajectory windows."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_mul
from supermimic.algo.state import StateExtractor, SuperMimicStateSpec
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import (
    LIMITS,
    PHYSICS_CONTRACT,
    disable_tw_velocity_limits,
    enable_tw_velocity_limits,
)

p = argparse.ArgumentParser(description=__doc__)
for key in ["manifest", "source-checkpoint", "output"]:
    p.add_argument("--" + key, type=Path, required=True)
p.add_argument("--smoke", action="store_true")
a = p.parse_args()
assert not a.output.exists()
a.output.mkdir(parents=True)
m = json.loads(a.manifest.read_text())
assert m["synthetic_diagnostic"] and len(m["motions"]) == 1
c = torch.load(a.source_checkpoint, map_location="cpu", weights_only=False)
extra = c["cfg"]["extra"]
assert (
    extra["tw_physics_contract"] == PHYSICS_CONTRACT
    and extra["tw_mjcf_sha256"]
    == hashlib.sha256(Path(os.environ["TW_SUPERMIMIC_MJCF"]).read_bytes()).hexdigest()
)
plan = {
    "contract": "root_pose_state_branches_v1",
    "fit_warmups": [5] if a.smoke else [5, 25, 75, 150],
    "validation_warmups": [] if a.smoke else [15, 35, 125, 175],
    "levels": [1.0, 2.0],
    "base_delta": [0.0005, 0.0005, 0.0005, 0.001, 0.001, 0.001],
    "units": ["m"] * 3 + ["rad"] * 3,
    "num_envs": 12,
    "seed": 42,
    "replay_atol": 1e-5,
    "sign_assignments": 2,
    "steps_per_branch": 1,
    "source_checkpoint_sha256": hashlib.sha256(
        a.source_checkpoint.read_bytes()
    ).hexdigest(),
    "manifest_sha256": hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
    "scope": "World-frame root-link position and rotation perturbations, with world velocities held fixed, under zero-action PD. Contact mode may change. Not action branches. Static temporal split only, not independent motion-source validation. Individual branches are not episodes. No policy changes.",
}
(a.output / "plan.json").write_text(json.dumps(plan, indent=2))
e = tw_supermimic_env_cfg([m["motions"][0]["file"]], num_envs=12, play=True)
e.seed = 42
e.events = {}
e.auto_reset = False
e.commands["motion"].pose_range = {}
e.commands["motion"].velocity_range = {}
e.commands["motion"].joint_position_range = (0.0, 0.0)
env = ManagerBasedRlEnv(e, device="cuda:0")
enable_tw_velocity_limits(env)
robot = env.scene["robot"]
cmd = env.command_manager.get_term("motion")
term = env.action_manager.get_term("joint_pos")
spec = SuperMimicStateSpec.from_env(env, cmd.cfg.body_names)
extract = StateExtractor(
    env,
    spec,
    torso_body_name="chest",
    left_foot_geom_pattern=r"^left_foot_collision(_\d+)?$",
    right_foot_geom_pattern=r"^right_foot_collision(_\d+)?$",
)
assert (
    list(robot.joint_names) == extra["tw_state_joint_names"]
    and list(term.target_names) == extra["tw_action_joint_names"]
)
(a.output / "control_mapping.json").write_text(
    json.dumps(
        {
            "state_joint_names": list(robot.joint_names),
            "action_joint_names": list(term.target_names),
            "offset": term._offset.cpu().tolist(),
            "scale": term._scale.cpu().tolist(),
        },
        indent=2,
    )
)
zero = torch.zeros((12, 30), device="cuda:0")
ids = torch.arange(12, device="cuda:0")
caps = torch.tensor([LIMITS[n] for n in robot.joint_names], device="cuda:0")
reports = []


def reset(warmup):
    env.reset()
    cmd.reset_to_frame(ids, 0, 0)
    env.sim.forward()
    for _ in range(warmup):
        out = env.step(zero)
        assert not bool((out[2] | out[3]).any()), (
            "Warmup failure; preserve logs, no retry"
        )
    return extract.state().clone(), extract.context().clone()


def step(action):
    out = env.step(action)
    n = extract.state().clone()
    assert bool((robot.data.joint_vel.abs() <= caps + 1e-5).all())
    return n, (out[2] | out[3]).clone()


def save(name, **arrays):
    values = {
        k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        for k, v in arrays.items()
    }
    assert all(np.isfinite(v).all() for v in values.values())
    np.savez_compressed(a.output / name, **values)


try:
    assert abs(env.step_dt - 0.02) < 1e-9
    for split, warmups in [
        ("fit", plan["fit_warmups"]),
        ("validation", plan["validation_warmups"]),
    ]:
        for warmup in warmups:
            s0, ctx0 = reset(warmup)
            n0, done0 = step(zero)
            assert not bool(done0.any()), "Nominal PD zero response terminated"
            basefile = f"{split}_warmup{warmup}_zero.npz"
            save(
                basefile,
                state=s0,
                context=ctx0,
                action=zero,
                next_state=n0,
                terminated=done0,
            )
            reports.append(
                {
                    "split": split,
                    "warmup": warmup,
                    "kind": "zero",
                    "file": basefile,
                    "count": 12,
                }
            )
            for eps in plan["levels"]:
                for assignment in range(2):
                    s, ctx = reset(warmup)
                    error = float((s - s0).abs().max())
                    torch.testing.assert_close(s, s0, rtol=0, atol=plan["replay_atol"])
                    torch.testing.assert_close(ctx, ctx0, rtol=0, atol=1e-7)
                    joint = np.repeat(np.arange(6), 2)
                    sign = np.tile([-1, 1], 6) * (1 if assignment == 0 else -1)
                    delta = torch.zeros((12, 6), device="cuda:0")
                    magnitudes = np.array(plan["base_delta"])[joint] * eps * sign
                    delta[ids, torch.tensor(joint, device="cuda:0")] = torch.tensor(
                        magnitudes, device="cuda:0", dtype=torch.float32
                    )
                    expected = s.clone()
                    expected[:, :3] += delta[:, :3]
                    angle = torch.linalg.vector_norm(delta[:, 3:], dim=1, keepdim=True)
                    dq = torch.cat(
                        [
                            torch.cos(angle / 2),
                            0.5 * torch.sinc(angle / (2 * torch.pi)) * delta[:, 3:],
                        ],
                        dim=1,
                    )
                    expected[:, 3:7] = torch.nn.functional.normalize(
                        quat_mul(dq, s[:, 3:7]), dim=1
                    )
                    world_position = robot.data.root_link_pos_w.clone() + delta[:, :3]
                    robot.write_root_link_pose_to_sim(
                        torch.cat([world_position, expected[:, 3:7]], dim=1)
                    )
                    robot.write_root_link_velocity_to_sim(s[:, 7:13])
                    env.sim.forward()
                    perturbed = extract.state().clone()
                    actual_context = extract.context().clone()
                    torch.testing.assert_close(perturbed, expected, rtol=0, atol=2e-6)
                    torch.testing.assert_close(actual_context, ctx, rtol=0, atol=1e-7)
                    u = zero.clone()
                    n, done = step(u)
                    name = f"{split}_warmup{warmup}_level{eps:.2f}_assignment{assignment}.npz"
                    save(
                        name,
                        state=perturbed,
                        context=actual_context,
                        action=u,
                        next_state=n,
                        axis=joint,
                        sign=sign,
                        requested_delta=delta,
                        terminated=done,
                        zero_state=s,
                        zero_next_state=n0,
                    )
                    r = {
                        "split": split,
                        "warmup": warmup,
                        "kind": "branch",
                        "level": eps,
                        "assignment": assignment,
                        "file": name,
                        "count": 12,
                        "max_replay_state_error": error,
                        "terminated_count": int(done.sum()),
                    }
                    reports.append(r)
                    print(json.dumps(r), flush=True)
                    (a.output / "partial.json").write_text(
                        json.dumps(reports, indent=2)
                    )
    (a.output / "results.json").write_text(
        json.dumps({"contract": plan["contract"], "rows": reports}, indent=2)
    )
    (a.output / "status").write_text("collected_pending_audit\n")
except BaseException:
    (a.output / "status").write_text("failed\n")
    raise
finally:
    env.close()
    disable_tw_velocity_limits(env)
