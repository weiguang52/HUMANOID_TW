"""Full-denominator issue12 analysis after ALL frozen training/evaluation jobs."""
import csv
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path
import numpy as np
from issue12_common import ROOT, COMMANDS, sha
from issue12_dataset import load_dataset
from issue12_statistics import summary, paired, tracking, paired_delay_rr_change
from issue12_statistics import decision_counts, fall_summary, paired_falls
from risk_analysis_utils import risk, seed_effect


def write_csv(name, rows):
    rows=list(rows)
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="") as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
            w.writeheader();w.writerows(rows)


def save_json(name, value):
    (ROOT/name).write_text(json.dumps(value,indent=2,allow_nan=False))


def partition_records(kind,rows):
    result={}
    for group in ["control","treatment"]:
        for command in COMMANDS:
            selected=[r for r in rows if r["group"]==group and r["command"]==command]
            assert selected
            p=ROOT/"records"/f"{kind}_{group}_{command}.csv.gz"
            p.parent.mkdir(exist_ok=True)
            with p.open("wb") as raw:
                with gzip.GzipFile(filename="",fileobj=raw,mode="wb",mtime=0) as stream:
                    with io.TextIOWrapper(stream,newline="") as text:
                        w=csv.DictWriter(text,fieldnames=list(selected[0]),lineterminator="\n")
                        w.writeheader();w.writerows(selected)
            b=gzip.decompress(p.read_bytes())
            assert sum(1 for _ in csv.DictReader(io.StringIO(b.decode())))==len(selected)
            result[str(p.relative_to(ROOT))]={"rows":len(selected),"gzip_sha256":sha(p),
                         "decompressed_csv_sha256":hashlib.sha256(b).hexdigest(),
                         "format":"lossless gzip, partitioned only to keep transport files small"}
    return result


def monitor_summary(models):
    rows=[];series=[]
    for name,m in models.items():
        data=list(csv.DictReader((Path(m["audit_dir"])/"reward_contributions.csv").open()))
        assert len(data)==1500
        keys=["raw_cost_mean","active_fraction","new_reward_per_step","old_reward_per_step",
              "total_reward_per_step","xy_tracking_reward_per_step","yaw_tracking_reward_per_step"]
        means={k:float(np.mean([float(x[k]) for x in data])) for k in keys}
        n=abs(means["new_reward_per_step"]);old=abs(means["old_reward_per_step"])
        track=means["xy_tracking_reward_per_step"]+means["yaw_tracking_reward_per_step"]
        ratios=np.asarray([float(x["abs_new_over_tracking"]) if x["abs_new_over_tracking"] else np.nan for x in data])
        valid=ratios[np.isfinite(ratios)]
        total_ratios=np.asarray([abs(float(x["new_reward_per_step"]))/abs(float(x["total_reward_per_step"])) if float(x["total_reward_per_step"])!=0 else np.nan for x in data])
        total_valid=total_ratios[np.isfinite(total_ratios)]
        row={"model":name,"group":m["group"],"training_seed":m["training_seed"],"weight":m["weight"],
             **means,"overall_abs_new_over_abs_old":None if old==0 else n/old,
             "overall_abs_new_over_tracking":None if track==0 else n/abs(track),
             "overall_abs_new_over_abs_total":None if means["total_reward_per_step"]==0 else n/abs(means["total_reward_per_step"]),
             "iteration_total_ratio_p95":None if not len(total_valid) else float(np.quantile(total_valid,.95)),
             "iteration_total_ratio_max":None if not len(total_valid) else float(total_valid.max()),
             "iterations_missing_total_ratio":int((~np.isfinite(total_ratios)).sum()),
             "iteration_ratio_p95":None if not len(valid) else float(np.quantile(valid,.95)),
             "iteration_ratio_max":None if not len(valid) else float(valid.max()),
             "iterations_new_abs_exceeds_tracking_sum":int((valid>1).sum()),
             "iterations_missing_ratio":int((~np.isfinite(ratios)).sum()),
             "max_term_identity_error":max(float(x["max_term_identity_error"]) for x in data)}
        rows.append(row)
        for i,x in enumerate(data):
            series.append({"model":name,"group":m["group"],"training_seed":m["training_seed"],
                           "additional_update":i+1,**x,"abs_new_over_abs_total":None if not np.isfinite(total_ratios[i]) else float(total_ratios[i])})
    write_csv("reward_monitor_summary.csv",rows)
    write_csv("reward_monitor_timeseries.csv",series)
    return rows


def main():
    assert (ROOT/"training_complete").exists() and (ROOT/"evaluation_complete").exists()
    assert "Ran 8 tests" in (ROOT/"statistics_tests_final.log").read_text() and "\nOK\n" in (ROOT/"statistics_tests_final.log").read_text()
    models=json.loads((ROOT/"final_models.json").read_text());assert len(models)==10
    trials,pushes,cases,cuts=load_dataset()
    assert len(trials)==92160
    manifest=partition_records("trials",trials)
    manifest.update(partition_records("pushes",pushes))
    save_json("partition_records_manifest.json",manifest)
    save_json("failure_case_records.json",cases)

    records=[];policies=[];grouped={};xy={};yaw={};effects={}
    for group in ["control","treatment"]:
        for seed in range(1,6):
            name=f"{group}_s{seed}";outputs=[]
            for es in [1,2,3]:
                d=json.loads((ROOT/f"{name}_eval{es}.json").read_text());outputs.append(d)
                assert d["checkpoint"]==models[name]["checkpoint"]
                assert d["observation_dims"]=={"policy":[101],"critic":[134]}
                for r in d["results"]:
                    assert r["num_envs"]==1024 and r["seconds"]==20
                    records.append({"model":name,"group":group,"training_seed":seed,
                        "eval_seed":es,"command":r["command"],
                        **risk(r["failed_trials"],r["num_envs"]),
                        "xy_error":r["metrics"]["xy_error"],"yaw_error":r["metrics"]["yaw_error"]})
            for command in COMMANDS:
                rr=[next(x for x in d["results"] if x["command"]==command) for d in outputs]
                policies.append({"model":name,"group":group,"training_seed":seed,"command":command,
                    **risk(sum(x["failed_trials"] for x in rr),3072),
                    "xy_error":float(np.mean([x["metrics"]["xy_error"] for x in rr])),
                    "yaw_error":float(np.mean([x["metrics"]["yaw_error"] for x in rr]))})
    write_csv("evaluation_rates.csv",records);write_csv("policy_rates.csv",policies)
    def policy_vector(group,command,key):
        return [next(x[key] for x in policies if x["group"]==group and x["command"]==command and x["training_seed"]==s) for s in range(1,6)]
    for command in COMMANDS:
        for group in ["control","treatment"]:
            selected=[p for p in policies if p["group"]==group and p["command"]==command]
            grouped[group+":"+command]={"seed_summary":fall_summary(policy_vector(group,command,"falls"),3072),
                                       "pooled_descriptive":risk(sum(x["falls"] for x in selected),15360)}
        effects[command]=paired_falls(policy_vector("treatment",command,"falls"),policy_vector("control",command,"falls"),3072)
        xy[command]=tracking(policy_vector("treatment",command,"xy_error"),policy_vector("control",command,"xy_error"))
        yaw[command]=tracking(policy_vector("treatment",command,"yaw_error"),policy_vector("control",command,"yaw_error"))
    save_json("group_results.json",grouped);save_json("paired_effects.json",effects)
    save_json("tracking_comparison.json",{"xy":xy,"yaw":yaw,
              "definition":"Each training seed: mean of three #7 non-terminal-frame eval summaries, including later auto-reset episodes. Group means equally weight five training seeds. Only xy has the frozen10% gate."})
    accept=decision_counts(policy_vector("control","forward_03","falls"),policy_vector("treatment","forward_03","falls"),
                    policy_vector("control","forward_05","falls"),policy_vector("treatment","forward_05","falls"),xy)
    save_json("acceptance.json",accept)

    # Full-denominator marginal/joint tables; no failure-only support-phase denominator.
    trial_tables=[];push_tables=[];joint_tables=[];q4rows=[];delay={};delay_counts={}
    factor_levels={"mass_q":[1,2,3,4],"static_q":[1,2,3,4],"dynamic_q":[1,2,3,4],"delay_ms":[0,20]}
    push_levels={**factor_levels,"amplitude_q":[1,2,3,4],"direction":["forward","back","side"],
                 "support_code":[0,1,2,3],"phase_bin":[0,1,2,3,4]}
    for group in ["control","treatment"]:
        for command in COMMANDS:
            ts=[t for t in trials if t["group"]==group and t["command"]==command]
            ps=[p for p in pushes if p["group"]==group and p["command"]==command and p["eligible"]]
            assert len(ts)==15360
            for factor,levels in factor_levels.items():
                for level in levels:
                    selected=[t for t in ts if t[factor]==level]
                    k=sum(t["failed"] for t in selected)
                    trial_tables.append({"group":group,"command":command,"factor":factor,"level":level,
                                         **risk(k,len(selected)),"sparse_n_lt30":len(selected)<30})
            for factor,levels in push_levels.items():
                for level in levels:
                    selected=[p for p in ps if p[factor]==level]
                    k=sum(p["failure_within1s"] for p in selected)
                    push_tables.append({"group":group,"command":command,"factor":factor,"level":level,
                                        **risk(k,len(selected)),"sparse_n_lt30":len(selected)<30})
            for static,mass,lag in itertools.product([1,2,3,4],[1,2,3,4],[0,20]):
                selected=[t for t in ts if (t["static_q"],t["mass_q"],t["delay_ms"])==(static,mass,lag)]
                joint_tables.append({"group":group,"command":command,"static_q":static,"mass_q":mass,
                                     "delay_ms":lag,**risk(sum(t["failed"] for t in selected),len(selected)),
                                     "sparse_n_lt30":len(selected)<30})
            counts={}
            for lag in [0,20]:
                counts[lag]=[]
                for seed in range(1,6):
                    selected=[t for t in ts if t["training_seed"]==seed and t["delay_ms"]==lag]
                    counts[lag].append([sum(t["failed"] for t in selected),len(selected)])
            delay[group+":"+command]=seed_effect(counts[20],counts[0])
            delay_counts[group+":"+command]=counts
            for seed in range(1,6):
                selected=[p for p in ps if p["training_seed"]==seed and p["amplitude_q"]==4 and p["direction"]=="forward"]
                events=[p for p in selected if p["failure_within1s"]]
                assert len(events)==len({p["trial_id"] for p in events})
                all_falls=sum(t["failed"] for t in ts if t["training_seed"]==seed)
                q4rows.append({"group":group,"command":command,"training_seed":seed,
                     **risk(len(events),len(selected)),"exposed_initial_trials":len({p["trial_id"] for p in selected}),
                     "all_initial_trials":3072,"all_first_failures":all_falls,
                     "captured_first_failure_fraction":None if not all_falls else len(events)/all_falls})
    write_csv("trial_stratified_risks.csv",trial_tables);write_csv("push_stratified_risks.csv",push_tables)
    write_csv("trial_joint_strata.csv",joint_tables);write_csv("q4_forward_seed_windows.csv",q4rows)
    rrchanges={}
    for command in COMMANDS:
        t=delay_counts["treatment:"+command];c=delay_counts["control:"+command]
        rrchanges[command]=paired_delay_rr_change(t[20],t[0],c[20],c[0])
    save_json("delay_effects.json",delay);save_json("paired_delay_rr_changes.json",rrchanges)
    q4summary={}
    for command in COMMANDS:
        result={}
        for group in ["control","treatment"]:
            rows=[x for x in q4rows if x["group"]==group and x["command"]==command]
            result[group]={"pooled_window_risk":risk(sum(x["falls"] for x in rows),sum(x["n"] for x in rows)),
                           "exposed_initial_trials":sum(x["exposed_initial_trials"] for x in rows),
                           "all_initial_trials":15360,"all_first_failures":sum(x["all_first_failures"] for x in rows),
                           "per_seed_events":[x["falls"] for x in rows]}
        t=[x for x in q4rows if x["group"]=="treatment" and x["command"]==command]
        c=[x for x in q4rows if x["group"]=="control" and x["command"]==command]
        result["paired_event_count_change"]=paired([x["falls"] for x in t],[x["falls"] for x in c])
        result["window_risk_effect"]=seed_effect([[x["falls"],x["n"]] for x in t],[[x["falls"],x["n"]] for x in c])
        q4summary[command]=result
    save_json("q4_forward_comparison.json",q4summary)

    modes=[]
    for group in ["control","treatment"]:
        for command in COMMANDS:
            selected=[c for c in cases if c["group"]==group and c["command"]==command]
            recent=sum(c["recent_push"] for c in selected)
            sagittal=sum(abs(c["terminal_pitch"])>=abs(c["terminal_roll"]) for c in selected)
            both=sum(c["recent_push"] and abs(c["terminal_pitch"])>=abs(c["terminal_roll"]) for c in selected)
            modes.append({"group":group,"command":command,"first_failures":len(selected),
                          "recent_push_within1s":recent,"terminal_sagittal":sagittal,"both":both})
    write_csv("failure_mechanism_summary.csv",modes)
    monitor_summary(models)
    worst={}
    for name,m in models.items():
        candidates=[p for p in policies if p["model"]==name]
        row=max(candidates,key=lambda x:(x["rate"],COMMANDS.index(x["command"])))
        worst[name]={"command":row["command"],"falls":row["falls"],"n":row["n"],
                     "pooled_fall_rate":row["rate"],"illustrative_video_seed":1,
                     "tie_rule":"prefer forward_08,forward_05,forward_03"}
    save_json("worst_command_selection.json",worst)
    save_json("analysis_integrity_audit.json",{"passed":True,"models":10,"evaluation_jobs":30,
        "command_records":90,"all_initial_trials":len(trials),"all_pushes":len(pushes),
        "all_eligible_1s_pushes":sum(p["eligible"] for p in pushes),"first_failures":len(cases),
        "raw_world_body_delta_contact_delay_mass_checks_passed":True,
        "all_partition_trial_rows":sum(x["rows"] for p,x in manifest.items() if Path(p).name.startswith("trials_")),
        "all_partition_push_rows":sum(x["rows"] for p,x in manifest.items() if Path(p).name.startswith("pushes_")),
        "seed_unit_inference":5,"fixed_behavior_sources_and_weight":True,"audit_only_precision_amendment_recorded":True})
    (ROOT/"analysis_complete").write_text("All ten models/92160 trials; frozen five-pair inference.")
    print("ISSUE12_ANALYSIS_COMPLETE",accept["decision"],"TRIALS",len(trials),flush=True)

if __name__=="__main__":
    main()
