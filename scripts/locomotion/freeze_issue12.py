"""Finalize immutable pre-training evidence after all no-update preflight gates."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
import isaaclab
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO
from issue12_common import ROOT, REPO, BASE, sha


def main():
    assert (ROOT/"calibration_complete").exists() and (ROOT/"runtime_smoke_complete").exists()
    assert json.loads((ROOT/"runtime_smoke_audit.json").read_text())["passed"]
    text=(ROOT/"unit_tests.log").read_text()
    assert "Ran 10 tests" in text and "\nOK\n" in text
    assert not list((REPO/"logs/rsl_rl/tw_stage1").glob("*_issue12_20261110_control_s*"))
    assert not list((REPO/"logs/rsl_rl/tw_stage1").glob("*_issue12_20261110_treatment_s*"))
    sources=[Path(x) for x in [
        "scripts/locomotion/train.py","scripts/locomotion/cli_args.py",
        "scripts/locomotion/remote_env.sh","scripts/locomotion/issue12_monitor.py",
        "scripts/locomotion/issue12_common.py","scripts/locomotion/run_issue12.py",
        "scripts/locomotion/calibrate_issue12.py","scripts/locomotion/smoke_issue12.py",
        "scripts/locomotion/freeze_issue12.py","scripts/locomotion/eval_cmd.py",
        "scripts/locomotion/failure_traces.py","scripts/locomotion/trial_risk_records.py",
        "tests/locomotion/test_issue12_recovery.py",
        "source/unitree_rl_lab/unitree_rl_lab/assets/robots/custom_humanoid.py"]]
    sources+=sorted(Path("source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/tw").rglob("*.py"))
    for p in sources:
        assert p.exists(),p
        if p.suffix==".py":ast.parse(p.read_text(),filename=str(p))
    dependencies=[Path(inspect.getsourcefile(OnPolicyRunner)),Path(inspect.getsourcefile(PPO))]
    ip=Path(isaaclab.__file__).parent
    dependencies += [ip/"envs/manager_based_rl_env.py",ip/"managers/reward_manager.py",
                     ip/"managers/event_manager.py"]
    calibration=json.loads((ROOT/"calibration_state.json").read_text());assert len(calibration)==5
    import yaml
    schemas={}
    for seed in range(1,6):
        audit=Path(calibration[str(seed)]["audit_dir"])
        d=json.loads((audit/"loaded_state_audit.json").read_text())
        ref=json.loads((REPO/f"validation_artifacts/stage1_issue10_20261109/requested_s{seed}_u4500_eval1.json").read_text())
        assert d["passed"] and d["observation_dims"]==ref["observation_dims"]
        summary=json.loads((audit/"run_summary.json").read_text())
        assert summary["additional_updates"]==0 and not list(Path(summary["log_dir"]).glob("model_*.pt"))
        cfg=yaml.load((Path(summary["log_dir"])/"params/env.yaml").read_text(),Loader=yaml.BaseLoader)
        spawn=cfg["scene"]["robot"]["spawn"]
        asset=Path(spawn.get("usd_path",spawn.get("asset_path")));assert asset.exists()
        dependencies.append(asset)
        if "usd_file_name" in spawn:
            cached=Path(spawn["usd_dir"])/spawn["usd_file_name"];assert cached.exists();dependencies.append(cached)
        schemas[str(seed)]=d["observation_dims"]
    identity={str(p):sha(p) for p in sorted(set(sources+dependencies),key=str)}
    result={"base":BASE,"frozen_before_formal_training":True,"frozen_unix":time.time(),
            "sha256":identity,"actual_parent_observation_dimensions_match_all_five":schemas,
            "training_distribution":"same robust08 strengthened-push bounded requested recipe",
            "only_control_treatment_reward_weight_differs":True,
            "calibration_has_no_saved_models_or_optimizer_updates":True}
    path=ROOT/"frozen_training_sources.json";assert not path.exists()
    path.write_text(json.dumps(result,indent=2))
    semantics={"runner_load_source":inspect.getsource(OnPolicyRunner.load),
               "ppo_load_source":inspect.getsource(PPO.load),
               "reward_compute_source":(ip/"managers/reward_manager.py").read_text().splitlines()[128:159],
               "env_reward_reset_push_order_source":(ip/"envs/manager_based_rl_env.py").read_text().splitlines()[198:239]}
    (ROOT/"runtime_resume_timing_semantics.json").write_text(json.dumps(semantics,indent=2))
    subprocess.run(["git","diff","--check"],check=True)
    (ROOT/"formal_gates_complete").write_text("10unit tests,5calibrations,3runtime smoke arms; weight/source/config gates frozen.")
    print("FORMAL_GATES_FROZEN",len(identity),"sources/dependencies",flush=True)

if __name__=="__main__":
    main()
