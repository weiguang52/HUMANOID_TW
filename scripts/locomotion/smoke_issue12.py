"""Runtime gate: original/zero/treated parent rollouts must share dynamics and RNG."""
import json
import os
from pathlib import Path
import numpy as np
from issue12_common import ROOT, REPO, run, train_args, sha


def main():
    os.chdir(REPO)
    assert (ROOT/"calibration_complete").exists()
    assert "\nOK\n" in (ROOT/"unit_tests.log").read_text()
    weight=json.loads((ROOT/"frozen_weight.json").read_text())["weight"]
    frozen=json.loads((ROOT/"frozen_weight_sha256.json").read_text())
    assert sha(ROOT/"frozen_weight.json")==frozen["frozen_weight.json"]
    statepath=ROOT/"smoke_state.json"
    state=json.loads(statepath.read_text()) if statepath.exists() else {}
    for mode,w in [("original",None),("control",0),("treatment",weight)]:
        if mode in state:
            assert (Path(state[mode])/"smoke_complete").exists()
            continue
        attempt=1
        while (ROOT/"smoke"/f"{mode}_attempt{attempt}").exists():
            attempt+=1
        audit=ROOT/"smoke"/f"{mode}_attempt{attempt}"
        args=train_args(1,"smoke_"+mode,audit,w,num_envs=64)+["--issue12_smoke_steps","1000"]
        run(args,ROOT/f"smoke_{mode}_attempt{attempt}.log","smoke")
        assert (audit/"smoke_complete").exists()
        state[mode]=str(audit)
        statepath.write_text(json.dumps(state,indent=2))
    audits={k:json.loads((Path(v)/"loaded_state_audit.json").read_text()) for k,v in state.items()}
    for mode,d in audits.items():
        assert d["passed"] and d["exact_state"]==audits["original"]["exact_state"]
        for key in ["observation_dims","actor_parameter_shapes","critic_parameter_shapes"]:
            assert d[key]==audits["original"][key]
    checks={}
    for point in ["initial","final"]:
        values={k:np.load(Path(v)/(point+"_state")/(point+".npz")) for k,v in state.items()}
        ref=values["original"]
        for mode in ["control","treatment"]:
            assert set(values[mode].files)==set(ref.files)
            for key in ref.files:
                assert np.array_equal(ref[key],values[mode][key]),(point,mode,key)
            checks[point+":"+mode]={"root_joints_complete_rng_bitwise_equal":True,
                                  "arrays":list(ref.files)}
    records={k:np.load(Path(v)/"smoke_rewards.npz") for k,v in state.items()}
    assert records["control"]["rewards"].shape==(1000,64)
    assert np.array_equal(records["original"]["rewards"],records["control"]["rewards"])
    assert np.array_equal(records["control"]["costs"],records["treatment"]["costs"])
    cost=records["control"]["costs"];assert cost.sum()>0
    expected=-weight*.02*cost
    actual=records["treatment"]["rewards"]-records["control"]["rewards"]
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    result={"passed":True,"steps":1000,"num_envs":64,"weight":weight,
            "original_vs_zero_all_step_rewards_bitwise_equal":True,
            "control_vs_treatment_unweighted_cost_bitwise_equal":True,
            "reward_difference_matches_only_new_term_max_abs_error":float(abs(actual-expected).max()),
            "policy_critic_observation_dims_unchanged":audits["original"]["observation_dims"],
            "actor_critic_network_shapes_and_loaded_state_unchanged":True,"state_rng_checks":checks,
            "all_parents":state,
            "limit":"64-env deterministic no-update smoke; formal runs retain 4096 envs and stochastic PPO."}
    (ROOT/"runtime_smoke_audit.json").write_text(json.dumps(result,indent=2))
    (ROOT/"runtime_smoke_complete").write_text("Original/zero dynamics+RNG and new-term identity passed.")
    print("RUNTIME_SMOKE_PASS",result["reward_difference_matches_only_new_term_max_abs_error"],flush=True)

if __name__=="__main__":
    main()
