"""Issue12 fixed ten-run study. No adaptive budget, checkpoint choice or weight search."""
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import torch
import yaml
from issue12_common import ROOT, REPO, BASE, run, train_args, evaluate, sha


def difference(a,b,path=""):
    if isinstance(a,dict) and isinstance(b,dict):
        for k in sorted(set(a)|set(b)):
            if k not in a or k not in b:
                yield path+"/"+k,a.get(k),b.get(k)
            else:
                yield from difference(a[k],b[k],path+"/"+k)
    elif a != b:
        yield path,a,b


def load_yaml(path):
    return yaml.load(Path(path).read_text(),Loader=yaml.BaseLoader)


def verify_frozen():
    assert (ROOT/"formal_gates_complete").exists()
    assert (ROOT/"runtime_smoke_complete").exists()
    assert json.loads((ROOT/"runtime_smoke_audit.json").read_text())["passed"]
    assert sha(ROOT/"frozen_weight.json")==json.loads((ROOT/"frozen_weight_sha256.json").read_text())["frozen_weight.json"]
    assert sha(ROOT/"protocol_plan.md")==json.loads((ROOT/"protocol_plan_sha256.json").read_text())["protocol_plan.md"]
    frozen=json.loads((ROOT/"frozen_training_sources.json").read_text())
    for name,h in frozen["sha256"].items():
        assert sha(name)==h,("Frozen source changed",name)
    assert subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()==BASE
    assert shutil.disk_usage(ROOT).free>2*1024**3,"Insufficient disk; do not delete unrelated files"


def audit_one(name,record,parent):
    folder=Path(record["run_dir"]);audit=Path(record["audit_dir"])
    old=Path(parent["checkpoint"]).parent
    changes={}
    for kind in ["env","agent"]:
        a=load_yaml(old/f"params/{kind}.yaml");b=load_yaml(folder/f"params/{kind}.yaml")
        diffs=list(difference(a,b))
        allowed={"/log_dir","/events/push_robot/func","/events/reset_post_push_clock",
                 "/rewards/post_push_pitch_recovery"} if kind=="env" else {
                 "/run_name","/max_iterations","/load_run","/load_checkpoint"}
        assert all(x[0] in allowed for x in diffs),(name,kind,diffs)
        assert b["seed"]==str(record["training_seed"])
        if kind=="env":
            assert b["scene"]["num_envs"]=="4096"
            assert b["events"]["push_robot"]["params"]==a["events"]["push_robot"]["params"]
            assert float(b["rewards"]["post_push_pitch_recovery"]["weight"])==-record["weight"]
            assert float(b["rewards"]["post_push_pitch_recovery"]["params"]["window_s"])==1.
        else:
            assert b["resume"]=="true" and b["max_iterations"]=="1500"
            assert b["load_checkpoint"]==Path(parent["checkpoint"]).name
            assert b["load_run"]==old.name
        changes[kind]=diffs
    loaded=json.loads((audit/"loaded_state_audit.json").read_text())
    assert loaded["passed"] and loaded["checkpoint_sha256"]==parent["sha256"]
    summary=json.loads((audit/"run_summary.json").read_text())
    assert summary["additional_updates"]==1500 and summary["num_envs"]==4096
    rows=list(csv.DictReader((audit/"reward_contributions.csv").open()))
    assert len(rows)==1500
    agent=load_yaml(folder/"params/agent.yaml")
    nsteps=int(agent["num_steps_per_env"])
    assert summary["environment_steps"]==1500*nsteps
    assert all(int(float(x["environment_steps"]))==nsteps for x in rows)
    assert all(float(x["max_term_identity_error"])<=1e-6 for x in rows)
    text=re.sub(r"\x1b\[[0-9;?]*[A-Za-z]","",(ROOT/f"{name}_train.log").read_text())
    iters=re.findall(r"Learning iteration\s+(\d+)/(\d+)",text)
    assert len(iters)==1500,(name,len(iters))
    assert int(iters[0][0])==loaded["iteration"] and int(iters[-1][0])==record["iteration"]
    previous=torch.load(parent["checkpoint"],map_location="cpu",weights_only=False)
    final=torch.load(record["checkpoint"],map_location="cpu",weights_only=False)
    a=previous["optimizer_state_dict"]["state"];b=final["optimizer_state_dict"]["state"]
    assert set(a)==set(b)
    increments=[float(b[k]["step"])-float(a[k]["step"]) for k in a if "step" in a[k]]
    expected=1500*int(agent["algorithm"]["num_learning_epochs"])*int(agent["algorithm"]["num_mini_batches"])
    assert increments and min(increments)==max(increments)==expected,(name,increments,expected)
    assert final["iter"]==record["iteration"]
    return {"passed":True,"config_changes_vs_own_parent":changes,"logged_updates":len(iters),
            "optimizer_step_increase_min":min(increments),"optimizer_step_increase_max":max(increments),
            "expected_optimizer_increase":expected,"monitor_rows":len(rows),
            "exact_actor_critic_optimizer_load":True,"std_reset":False}


def audit_pair(seed,models):
    a=models[f"control_s{seed}"];b=models[f"treatment_s{seed}"]
    changes={}
    for kind in ["env","agent"]:
        x=load_yaml(Path(a["run_dir"])/f"params/{kind}.yaml")
        y=load_yaml(Path(b["run_dir"])/f"params/{kind}.yaml")
        diffs=list(difference(x,y))
        allowed={"/log_dir","/rewards/post_push_pitch_recovery/weight"} if kind=="env" else {"/run_name"}
        assert all(z[0] in allowed for z in diffs),(seed,kind,diffs)
        changes[kind]=diffs
    import numpy as np
    x=np.load(Path(a["audit_dir"])/"initial_state/initial.npz")
    y=np.load(Path(b["audit_dir"])/"initial_state/initial.npz")
    assert set(x.files)==set(y.files)
    for key in x.files:
        assert np.array_equal(x[key],y[key]),("Initial pair state/RNG mismatch",seed,key)
    return {"passed":True,"only_one_reward_weight_plus_run_metadata_differs":changes,
            "all_initial_root_joint_complete_rng_arrays_bitwise_equal":list(x.files)}


def main():
    os.chdir(REPO);verify_frozen()
    (ROOT/"formal_driver_process.json").write_text(json.dumps({"pid":os.getpid(),"state":"running"}))
    parents=json.loads((ROOT/"input_models.json").read_text())
    weight=json.loads((ROOT/"frozen_weight.json").read_text())["weight"]
    statepath=ROOT/"formal_training_state.json"
    models=json.loads(statepath.read_text()) if statepath.exists() else {}
    audits=json.loads((ROOT/"training_audits.json").read_text()) if (ROOT/"training_audits.json").exists() else {}
    pairs={}
    for seed in range(1,6):
        parent=parents[f"requested_s{seed}_u4500"]
        for group,w in [("control",0.),("treatment",weight)]:
            name=f"{group}_s{seed}";verify_frozen()
            if name not in models:
                audit=ROOT/"training"/name
                suffix=f"issue12_20261110_{group}_s{seed}"
                if audit.exists() or list((REPO/"logs/rsl_rl/tw_stage1").glob("*_"+suffix)):
                    raise RuntimeError("Incomplete run requires explicit budget-preserving recovery: "+name)
                run(train_args(seed,group,audit,w),ROOT/f"{name}_train.log","training_"+name)
                summary=json.loads((audit/"run_summary.json").read_text())
                folder=Path(summary["log_dir"]);iteration=summary["final_iteration"]
                checkpoint=folder/f"model_{iteration}.pt";assert checkpoint.exists()
                record={"group":group,"training_seed":seed,"weight":w,"run_dir":str(folder),
                        "audit_dir":str(audit),"parent_checkpoint":parent["checkpoint"],
                        "parent_sha256":parent["sha256"],"additional_updates":1500,
                        "cumulative_requested_updates":6000,"iteration":iteration,
                        "checkpoint":str(checkpoint),"sha256":sha(checkpoint),
                        "elapsed_seconds":json.loads((ROOT/"progress.json").read_text())["finished_unix"]-json.loads((ROOT/"progress.json").read_text())["started_unix"]}
                audits[name]=audit_one(name,record,parent)
                models[name]=record
                (ROOT/"training_audits.json").write_text(json.dumps(audits,indent=2))
                statepath.write_text(json.dumps(models,indent=2))
            else:
                assert sha(models[name]["checkpoint"])==models[name]["sha256"]
                audits[name]=audit_one(name,models[name],parent)
        pairs[str(seed)]=audit_pair(seed,models)
        (ROOT/"matched_pair_config_rng_audit.json").write_text(json.dumps(pairs,indent=2))
    assert len(models)==10 and len(pairs)==5
    (ROOT/"training_complete").write_text("Ten matched full-state +1500-update continuations.")
    (ROOT/"final_models.json").write_text(json.dumps(models,indent=2))
    for name,m in models.items():
        for eval_seed in [1,2,3]:
            verify_frozen()
            output=ROOT/f"{name}_eval{eval_seed}.json"
            if output.exists():
                d=json.loads(output.read_text())
                assert d["checkpoint"]==m["checkpoint"] and len(d["results"])==3
                for command in ["forward_03","forward_05","forward_08"]:
                    assert (ROOT/"traces"/output.stem/(command+"_trials.npz")).exists()
                continue
            run(evaluate(m["checkpoint"],eval_seed,output),output.with_suffix(".log"),"evaluation_"+name)
    (ROOT/"evaluation_complete").write_text("30jobs/90commands/92160initial trials; all final models.")
    (ROOT/"formal_driver_process.json").write_text(json.dumps({"pid":os.getpid(),"state":"complete"}))
    print("ISSUE12_TRAINING_AND_EVALUATION_COMPLETE",flush=True)

if __name__=="__main__":
    main()
