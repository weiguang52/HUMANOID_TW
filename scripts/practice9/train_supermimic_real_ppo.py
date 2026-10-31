"""Real-feedback learnability diagnostic, not SuperMimic algorithm acceptance."""

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic.algo.config import SuperMimicCfg
from supermimic.algo.trainer import SuperMimicTrainer
from supermimic.config.tw.env_cfgs import tw_supermimic_env_cfg
from supermimic.config.tw.velocity_limits import enable_tw_velocity_limits
from supermimic_actor_initialization import zero_readout
from supermimic_real_ppo import (
    KIND,
    RealFeedbackCollector,
    bootstrap_timeouts,
    construct_ppo,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--updates", type=int, default=4)
    p.add_argument("--resume", type=Path)
    p.add_argument("--sole-normal-penalty", type=float, default=0.0)
    p.add_argument(
        "--stop-after",
        type=int,
        help="Smoke-only interrupted budget; resume keeps the original total plan",
    )
    a = p.parse_args()
    assert math.isfinite(a.sole_normal_penalty) and 0 <= a.sole_normal_penalty <= 5
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "metrics.jsonl").exists() and not a.resume:
        raise ValueError("Existing run: resume explicitly, never overwrite")
    source = torch.load(a.parent, map_location="cuda:0", weights_only=False)
    assert source.get("kind") is None
    assert (
        sha(os.environ["TW_SUPERMIMIC_MJCF"])
        == source["cfg"]["extra"]["tw_mjcf_sha256"]
    )
    cfg = SuperMimicCfg(**source["cfg"])
    cfg.compile = False
    cfg.wm_anchor_collection = None
    cfg.policy_anchor_collection = None
    cfg.experiment_name = "real_physics_ppo_diagnostic"
    rows = json.loads(a.manifest.read_text())["motions"]
    files = [r["file"] for r in rows]
    if a.sole_normal_penalty:
        assert json.loads(a.manifest.read_text())["synthetic_diagnostic"]
        assert len(files) == 1
        with np.load(files[0], allow_pickle=False) as reference:
            for key in ["joint_pos", "body_pos_w", "body_quat_w"]:
                assert np.allclose(reference[key], reference[key][0], rtol=0, atol=1e-7)
            for key in ["joint_vel", "body_lin_vel_w", "body_ang_vel_w"]:
                assert np.max(np.abs(reference[key])) < 1e-7
    plan = {
        "kind": KIND,
        "parent_sha256": sha(a.parent),
        "manifest_sha256": sha(a.manifest),
        "motion_sha256": {f: sha(f) for f in files},
        "num_envs": a.num_envs,
        "seed": 42,
        "updates": a.updates,
        "rollout_steps": 32,
        "epochs": 4,
        "minibatches": 4,
        "learning_rate": 1e-4,
        "gamma": cfg.gamma,
        "lam": 0.95,
        "clip": 0.2,
        "noise_std": cfg.action_noise,
        "fixed_source_normalizers": True,
        "all_actions_trainable": True,
        "world_model_used_for_policy_update": False,
        "sole_normal_penalty": a.sole_normal_penalty,
        "sole_normal_contract": "reference_body_z_normal_static_v1",
        "actual_transitions": a.num_envs * 32 * a.updates,
        "scope": "Diagnostic; different real-sample budget; not a matched algorithm comparison or final acceptance",
    }
    if (a.output / "plan.json").exists():
        assert json.loads((a.output / "plan.json").read_text()) == plan
    else:
        (a.output / "plan.json").write_text(json.dumps(plan, indent=2))
    env_cfg = tw_supermimic_env_cfg(files, num_envs=a.num_envs, play=False)
    env_cfg.seed = 42
    env_cfg.auto_reset = False
    env = ManagerBasedRlEnv(env_cfg, device="cuda:0")
    enable_tw_velocity_limits(env)
    t = SuperMimicTrainer(env, cfg, log_dir=a.output / "runtime")
    t.load(a.parent)
    t.initialize()
    collector = RealFeedbackCollector(t, sole_normal_penalty=a.sole_normal_penalty)
    from supermimic_sole_normal import check_known_rotations

    sole_math_check = check_known_rotations()
    obs = collector.observe()
    alg = construct_ppo(t.actor_critic, obs, a.num_envs, cfg.gamma, cfg.action_noise)
    # Test nonzero parent outputs, before zeroing the readout.
    ac = t.actor_critic
    flat = t.history.reshape(a.num_envs, -1)
    goal = t.command.goal()
    with torch.no_grad():
        torch.testing.assert_close(alg.actor(obs), ac.act(flat, goal), rtol=0, atol=0)
        torch.testing.assert_close(
            alg.critic(obs).squeeze(-1),
            ac.value(t.extractor.state(), goal, t.extractor.context()),
            rtol=0,
            atol=0,
        )
    init = zero_readout(ac, alg.optimizer)
    frozen_wm = {k: v.clone() for k, v in t.world_model.state_dict().items()}
    frozen_norm = {
        k: v.clone()
        for k, v in ac.state_dict().items()
        if k.startswith(("obs_norm.", "goal_norm.", "critic_norm."))
    }
    wm_calls = [0]
    hook = t.world_model.register_forward_hook(
        lambda *unused: wm_calls.__setitem__(0, wm_calls[0] + 1)
    )
    start = 0
    if a.resume:
        checkpoint = torch.load(a.resume, map_location="cuda:0", weights_only=False)
        assert checkpoint["kind"] == KIND and checkpoint["plan"] == plan
        assert checkpoint["cfg"] == asdict(cfg)
        for key, value in frozen_wm.items():
            assert torch.equal(checkpoint["world_model"][key], value), key
        ac.load_state_dict(checkpoint["actor_critic"])
        alg.optimizer.load_state_dict(checkpoint["ppo_optimizer"])
        start = int(checkpoint["ppo_update"])
        t.env_steps = int(checkpoint["env_steps"])
        # Simulator/replay rebuilt, as in the original trainer; not bitwise physical replay.
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint["cuda_rng_states"]])
    else:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    # Exact terminal-value bootstrap semantics, including simultaneous termination.
    torch.testing.assert_close(
        bootstrap_timeouts(
            torch.ones(3),
            torch.tensor([False, True, False]),
            torch.tensor([True, True, False]),
            torch.tensor([2.0, 7.0, 9.0]),
            0.99,
        ),
        torch.tensor([2.98, 1.0, 1.0]),
    )

    def invariants():
        assert wm_calls[0] == 0
        for k, v in frozen_wm.items():
            assert torch.equal(t.world_model.state_dict()[k], v), k
        for k, v in frozen_norm.items():
            assert torch.equal(ac.state_dict()[k], v), k
        for v in ac.state_dict().values():
            assert torch.isfinite(v).all()
        for state in alg.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    assert torch.isfinite(value).all()
        assert all(
            math.isfinite(float(x))
            for x in alg.actor.distribution.std.detach().flatten()
        )
        torch.testing.assert_close(
            alg.actor.distribution.std,
            torch.full_like(alg.actor.distribution.std, cfg.action_noise),
            rtol=0,
            atol=0,
        )

    def save(update, name):
        invariants()
        path = a.output / name
        temp = path.with_suffix(".tmp")
        data = {
            "kind": KIND,
            "plan": plan,
            "cfg": asdict(cfg),
            "actor_critic": ac.state_dict(),
            "world_model": t.world_model.state_dict(),
            "ppo_optimizer": alg.optimizer.state_dict(),
            "ppo_update": update,
            "iteration": source["iteration"] + update,
            "env_steps": t.env_steps,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
            "initialization": init,
            "simulator_resume": "Rebuilt, not bitwise continuation",
        }
        torch.save(data, temp)
        temp.replace(path)
        # Restore the actual optimizer and verify every Adam moment after serialization.
        loaded = torch.load(path, map_location="cuda:0", weights_only=False)
        before = alg.optimizer.state_dict()
        restored = torch.optim.Adam(alg.optimizer.param_groups, lr=1e-4)
        restored.load_state_dict(loaded["ppo_optimizer"])
        after = restored.state_dict()
        assert (
            before["param_groups"] == after["param_groups"]
            and before["state"].keys() == after["state"].keys()
        )
        for key, value in before["state"].items():
            for field, tensor in value.items():
                torch.testing.assert_close(
                    tensor.cpu(), after["state"][key][field].cpu(), rtol=0, atol=0
                )
        try:
            t.load(path)
        except ValueError as error:
            assert "own optimizer resume" in str(error)
        else:
            raise AssertionError("PPO checkpoint accepted for SuperMimic continuation")
        t.load(path, evaluation_only=True)
        return path

    alg.train_mode()
    began = time.monotonic()
    update = start
    try:
        with (a.output / "metrics.jsonl").open("a") as log:
            for update in range(
                start + 1, min(a.updates, a.stop_after or a.updates) + 1
            ):
                offset = len(collector.rewards)
                for step in range(32):
                    with torch.no_grad():
                        action = alg.act(obs)
                        obs = collector.step(action, alg)
                with torch.no_grad():
                    alg.compute_returns(obs)
                losses = alg.update()
                assert all(math.isfinite(float(v)) for v in losses.values())
                record = {
                    "update": update,
                    "env_steps": t.env_steps,
                    "diagnostic_env_steps": t.env_steps - source["env_steps"],
                    "reward_mean": float(np.mean(collector.rewards[offset:])),
                    "base_reward_mean": float(np.mean(collector.base_rewards[offset:])),
                    "sole_penalty_mean": float(np.mean(collector.penalties[offset:])),
                    "sole_cost_mean": float(np.mean(collector.sole_costs[offset:])),
                    "done_count": collector.done_count,
                    "timeout_count": collector.timeout_count,
                    "completed_seconds_mean": float(
                        np.mean(collector.completed_seconds[-100:])
                    )
                    if collector.completed_seconds
                    else 0,
                    "reset_live_max_error": collector.reset_live_max_error,
                    "losses": losses,
                    "seconds": time.monotonic() - began,
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                if update % 10 == 0 or update == a.updates:
                    print(json.dumps(record), flush=True)
                if update % 100 == 0:
                    save(update, f"checkpoint_{update:04d}.pt")
        final = save(
            update,
            "checkpoint_final.pt"
            if update == a.updates
            else f"checkpoint_{update:04d}.pt",
        )
        audit = {
            "resumed_from_update": start,
            "completed_updates": update,
            "checkpoint_sha256": sha(final),
            "adapter_parent_equivalence": True,
            "sole_math_check": sole_math_check,
            "sole_native_max_error": collector.sole.max_native_error
            if collector.sole
            else 0.0,
            "sole_normal_penalty": a.sole_normal_penalty,
            "sole_native_checks_current_segment": collector.sole.calls
            if collector.sole
            else 0,
            "world_model_calls": wm_calls[0],
            "world_model_unchanged": True,
            "normalizers_frozen": True,
            "actual_ppo_optimizer_roundtrip": True,
            "timeout_bootstrap_unit_check": True,
            "done_count": collector.done_count,
            "timeout_count": collector.timeout_count,
            "reset_live_max_error": collector.reset_live_max_error,
            "plan": plan,
        }
        (a.output / "audit.json").write_text(json.dumps(audit, indent=2))
        (a.output / "status").write_text(
            "trained_pending_evaluation\n"
            if update == a.updates
            else "smoke_interrupted_budget_resume_required\n"
        )
    except BaseException:
        # A completed-update checkpoint is safe to resume; in-progress rollout is not.
        (a.output / "status").write_text(
            "failed_preserve_logs_and_completed_checkpoints\n"
        )
        raise
    finally:
        hook.remove()
        t.close()
        env.close()


if __name__ == "__main__":
    main()
