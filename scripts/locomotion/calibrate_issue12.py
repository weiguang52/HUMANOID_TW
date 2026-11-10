"""Five fixed parent calibration rollouts; zero PPO updates and one frozen weight."""
import json
import os
from pathlib import Path
from issue12_common import ROOT, REPO, run, train_args, sha


def main():
    os.chdir(REPO)
    assert "\nOK\n" in (ROOT/"unit_tests.log").read_text()
    assert not list((REPO/"logs/rsl_rl/tw_stage1").glob("*_issue12_20261110_control_s*"))
    assert not list((REPO/"logs/rsl_rl/tw_stage1").glob("*_issue12_20261110_treatment_s*"))
    statepath=ROOT/"calibration_state.json"
    state=json.loads(statepath.read_text()) if statepath.exists() else {}
    (ROOT/"calibration_driver.json").write_text(json.dumps({"pid":os.getpid(),"state":"running"}))
    for seed in range(1,6):
        key=str(seed)
        if key in state:
            audit=Path(state[key]["audit_dir"])
            assert (audit/"calibration.json").exists() and (audit/"run_summary.json").exists()
            assert json.loads((audit/"loaded_state_audit.json").read_text())["passed"]
            continue
        attempt=1
        while (ROOT/"calibration"/f"seed{seed}_attempt{attempt}").exists():
            attempt+=1
        audit=ROOT/"calibration"/f"seed{seed}_attempt{attempt}"
        args=train_args(seed,"calibration",audit,0)+["--issue12_calibration_steps","1024"]
        run(args,ROOT/f"calibration_s{seed}_attempt{attempt}.log","calibration")
        row=json.loads((audit/"calibration.json").read_text())
        assert row["environment_steps"]==1024 and row["transitions"]==4096*1024
        assert row["no_optimizer_or_parameter_or_normalizer_updates"]
        state[key]={"audit_dir":str(audit),"row":row}
        statepath.write_text(json.dumps(state,indent=2))
    rows=[state[str(s)]["row"] for s in range(1,6)]
    old=sum(r["old_reward_per_step"] for r in rows)/5
    raw=sum(r["raw_cost_mean"] for r in rows)/5
    dts={r["step_dt"] for r in rows};assert dts=={.02}
    assert old>0 and raw>0
    weight=.05*old/(.02*raw)
    record={"weight":weight,"window_s":1.0,"pitch_rate_squared_coefficient":.1,
            "target_fraction_of_old_signed_positive_mean":.05,
            "old_mean_reward_per_control_step":old,"raw_gated_cost_mean":raw,
            "step_dt":.02,"expected_new_mean_reward_per_step":-weight*.02*raw,
            "formula":"w=.05*mean(old_reward_per_step)/(step_dt*mean(gated pitch^2+.1*pitch_dot^2))",
            "calibration_seeds":[1,2,3,4,5],"steps_per_seed":1024,"environments":4096,
            "total_transitions":5*4096*1024,"optimizer_updates":0,
            "source_state":state,"no_weight_scan":True,"frozen_before_formal_training":True}
    path=ROOT/"frozen_weight.json"
    if path.exists():
        assert json.loads(path.read_text())==record
    else:
        path.write_text(json.dumps(record,indent=2))
    (ROOT/"frozen_weight_sha256.json").write_text(json.dumps({"frozen_weight.json":sha(path)},indent=2))
    (ROOT/"calibration_complete").write_text("Five frozen-parent rollouts; single calculated weight.")
    (ROOT/"calibration_driver.json").write_text(json.dumps({"pid":os.getpid(),"state":"complete"}))
    print("CALIBRATION_COMPLETE_WEIGHT",weight,"OLD_MEAN",old,"RAW_MEAN",raw,flush=True)

if __name__=="__main__":
    main()
