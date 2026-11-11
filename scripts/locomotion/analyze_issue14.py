"""Three-group paired analysis with frozen issue14 gates and descriptive achieved-speed bins."""
import csv,json,hashlib,math
from pathlib import Path
import numpy as np
from issue14_common import ROOT,REPO,sha
from run_issue14 import verify_frozen,CLEAN,FORWARD
from issue13_statistics import compare,summary
from issue13_tracking import SCOPES,PARTS
from issue14_analysis_data import load_command,pooled_statistics,env_statistics,TrialCsvs
OLD=Path("/root/gpufree-data/tw_stage1/issue13_20261110")
OLD_ARCHIVE=REPO/"validation_artifacts/stage1_issue13_20261110"
GROUPS=["control","treatment","clock"]
PAIRS=[("control","treatment"),("control","clock"),("treatment","clock")]

def save(name,data):(ROOT/name).write_text(json.dumps(data,indent=2,allow_nan=False))
def csvwrite(name,rows):
 rows=list(rows)
 with (ROOT/name).open("w",newline="") as f:
  if rows:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def main():
 assert (ROOT/"formal_evaluation_complete").exists();verify_frozen()
 jobs=json.loads((ROOT/"formal_job_state.json").read_text());assert len(jobs)==30 and all(v["passed"] for v in jobs.values())
 models={**json.loads((ROOT/"historical_models.json").read_text()),**json.loads((ROOT/"final_models.json").read_text())};assert len(models)==15
 oldmanifest=json.loads((OLD_ARCHIVE/"sha256.json").read_text());rows=[];falls=[];audits=[];candidates={g:[] for g in GROUPS};exports=TrialCsvs();bin_trials=[]
 for condition in ["clean","combined"]:
  for name,m in models.items():
   for es in [1,2,3]:
    origin=ROOT if m["group"]=="clock" else OLD;out=origin/condition/f"{name}_eval{es}.json";d=json.loads(out.read_text())
    assert d["checkpoint"]==m["checkpoint"] and sha(m["checkpoint"])==m["sha256"]
    assert d["issue13_tracking"]["model_optimizer_state_unchanged"] and d["issue13_tracking"]["training_updates"]==0
    folder=Path(d["issue13_tracking"]["output"])
    for result in d["results"]:
     cmd=result["command"];z,meta=load_command(folder,cmd,result)
     base={"model":name,"group":m["group"],"training_seed":m["training_seed"],"eval_seed":es}
     exports.write(condition,cmd,base,z,result["target"])
     falls.append({"condition":condition,"command":cmd,**base,"falls":result["failed_trials"],"n":1024})
     for si,scope in enumerate(SCOPES):
      for pi in ([0] if condition=="clean" else range(5)):
       stat=pooled_statistics(z,si,pi,result["target"]);rows.append({"condition":condition,"command":cmd,**base,"scope":scope,"partition":PARTS[pi],**stat})
     files=[folder/(cmd+".npz"),folder/(cmd+".json"),out]
     if m["group"]!="clock":
      for p in files:
       rel=str(p.relative_to(OLD));assert sha(p)==oldmanifest[rel]==sha(OLD_ARCHIVE/rel),(name,cmd,rel)
     audits.append({"condition":condition,"model":name,"group":m["group"],"eval_seed":es,"command":cmd,"initial_trials":1024,
      "source_files_sha256":{str(p):sha(p) for p in files},"original_metrics_packed_masks_counts_moments_reset_and_reference_audited":True,"historical_bytes_match_committed_issue13_bundle":m["group"]!="clock"})
     if condition=="combined":
      counts,means,_=env_statistics(z,0,0,result["target"]);assert (counts>0).all()
      for i in range(1024):
       vx=float(means[i,2]);b=int(np.searchsorted(np.arange(0.,1.000001,.05),vx,side="right"))
       bin_trials.append({"command":cmd,**base,"env_id":i,"actual_mean_vx":vx,"bin_id":b,"fell":int(z["first_failure_step"][i]>=0)})
      if cmd=="forward_08":
       candidates[m["group"]].extend({"model":name,"group":m["group"],"training_seed":m["training_seed"],"eval_seed":es,"env_id":i,
        "xy_error":float(means[i,0]),"source_output":str(out),"source_moments":str(folder/(cmd+".npz")),"first_failure_step":int(z["first_failure_step"][i])} for i in range(1024))
 save("per_trial_csv_manifest.json",exports.close());assert len(audits)==360 and sum(x["initial_trials"] for x in audits)==368640
 csvwrite("evaluation_metric_rows.csv",rows);csvwrite("evaluation_fall_rows.csv",falls)
 metrics=[k for k in rows[0] if k not in ["condition","command","model","group","training_seed","eval_seed","scope","partition"]];policy=[]
 for condition in ["clean","combined"]:
  for name,m in models.items():
   for cmd in (CLEAN if condition=="clean" else FORWARD):
    for scope in SCOPES:
     for part in (["all"] if condition=="clean" else PARTS):
      selected=[x for x in rows if x["condition"]==condition and x["model"]==name and x["command"]==cmd and x["scope"]==scope and x["partition"]==part];assert len(selected)==3
      policy.append({"condition":condition,"command":cmd,"model":name,"group":m["group"],"training_seed":m["training_seed"],"scope":scope,"partition":part,
       **{k:float(np.mean([x[k] for x in selected])) for k in metrics}})
 csvwrite("policy_metric_rows.csv",policy)
 # Recomputed historical seed statistics must agree exactly with the previously delivered analysis.
 prior=list(csv.DictReader((OLD/"policy_metric_rows.csv").open()))
 for x in policy:
  if x["group"]=="clock":continue
  y=next(v for v in prior if all(str(x[k])==v[k] for k in ["condition","command","model","group","training_seed","scope","partition"]))
  for k in metrics:assert x[k]==float(y[k]),(x["model"],k)
 def values(condition,cmd,scope,part,metric,g):
  return [next(x[metric] for x in policy if x["condition"]==condition and x["command"]==cmd and x["scope"]==scope and x["partition"]==part and x["group"]==g and x["training_seed"]==s) for s in range(1,6)]
 comparisons={};flat=[]
 signed={"vx_mean","vy_mean","wz_mean","vx_bias"}
 for condition in ["clean","combined"]:
  for cmd in (CLEAN if condition=="clean" else FORWARD):
   for scope in SCOPES:
    for part in (["all"] if condition=="clean" else PARTS):
     for metric in metrics:
      for reference,candidate in PAIRS:
       d=compare(values(condition,cmd,scope,part,metric,reference),values(condition,cmd,scope,part,metric,candidate))
       if metric in signed:
        for field in ["ratio_of_group_means","relative_change","relative_change_bootstrap95","point_not_worse_than5percent","point_worse_than5percent_and_positive95"]:d[field]=None
        d["relative_change_not_interpreted_for_signed_metric"]=True
       d["reference_group"]=reference;d["candidate_group"]=candidate
       key=":".join([candidate+"_vs_"+reference,condition,cmd,scope,part,metric]);comparisons[key]=d
       ci=d["paired_difference"]["bootstrap95"]
       flat.append({"comparison":candidate+"_vs_"+reference,"condition":condition,"command":cmd,"scope":scope,"partition":part,"metric":metric,
        "reference_mean":d["control"]["mean"],"candidate_mean":d["treatment"]["mean"],"paired_difference":d["paired_difference"]["mean"],"paired95_low":ci[0],"paired95_high":ci[1],
        "relative_change":d["relative_change"],"multiple_comparisons_corrected":False})
 save("paired_comparisons.json",comparisons);csvwrite("paired_metric_comparisons.csv",flat)
 fall_results={}
 for condition in ["clean","combined"]:
  for cmd in (CLEAN if condition=="clean" else FORWARD):
   groups={}
   for g in GROUPS:
    counts=[sum(x["falls"] for x in falls if x["condition"]==condition and x["command"]==cmd and x["group"]==g and x["training_seed"]==s) for s in range(1,6)]
    groups[g]={"per_seed_falls":counts,"per_seed_n":3072,"pooled_falls":sum(counts),"pooled_n":15360,"seed_summary":summary(np.asarray(counts)/3072)}
   paired={candidate+"_vs_"+reference:compare(np.asarray(groups[reference]["per_seed_falls"])/3072,np.asarray(groups[candidate]["per_seed_falls"])/3072) for reference,candidate in PAIRS}
   fall_results[condition+":"+cmd]={"groups":groups,"paired":paired}
 save("fall_results.json",fall_results)
 def get(pair,cond,cmd,part,metric,scope=SCOPES[0]):return comparisons[":".join([pair,cond,cmd,scope,part,metric])]
 def gates_for(scope=SCOPES[0]):
  primary=get("clock_vs_treatment","clean","forward_08","all","xy_error",scope)
  speed=get("clock_vs_control","clean","forward_08","all","vx_mean",scope)
  c03=fall_results["combined:forward_03"]["groups"]["clock"]["seed_summary"]
  c05=fall_results["combined:forward_05"]["groups"]
  gates={"primary_clean08_xy_vs_treatment_upper95_below_zero":primary["paired_difference"]["bootstrap95"][1]<0,
   "clean05_xy_vs_control_mean_degradation_at_most5percent":bool(get("clock_vs_control","clean","forward_05","all","xy_error",scope)["point_not_worse_than5percent"]),
   "clean08_xy_vs_control_mean_degradation_at_most5percent":bool(get("clock_vs_control","clean","forward_08","all","xy_error",scope)["point_not_worse_than5percent"]),
   "clean08_mean_vx_vs_control95_interval_not_entirely_negative":speed["paired_difference"]["bootstrap95"][1]>=0,
   "combined03_clock_seed_level_fall95_upper_below1percent":c03["bootstrap95"][1]<.01,
   "combined05_clock_mean_falls_not_above_control":c05["clock"]["seed_summary"]["mean"]<=c05["control"]["seed_summary"]["mean"]}
  return {"gates":gates,"primary_mitigation_supported":gates["primary_clean08_xy_vs_treatment_upper95_below_zero"],"all_gates_pass":all(gates.values()),"primary":primary,"clean08_vx":speed,"scope":scope}
 decision=gates_for();decision.update({"formal_updates_per_seed":1500,"formal_seeds":5,"total_formal_updates":7500,"multiple_comparisons_corrected":False,
 "paired_resamples":3125,"default_promoted":False,"no_extra_budget_or_encoding_search":True,"stop_clock_hypothesis":not decision["primary_mitigation_supported"],"original_issue12_rejection_unchanged":True})
 save("decision.json",decision);save("classification_sensitivity.json",{"primary_not_reselected":True,"initial_episode":gates_for(SCOPES[1]),"note":"Fall outcomes use original initial-trial first-failure definitions in both analyses; clock training boundary is unchanged."})
 selection={}
 for group,choices in candidates.items():
  assert len(choices)==15360;median=float(np.median([x["xy_error"] for x in choices]));pick=min(choices,key=lambda x:(abs(x["xy_error"]-median),x["training_seed"],x["eval_seed"],x["env_id"]))
  selection[group]={**pick,"command":"forward_08","condition":"combined","group_median_xy":median,"distance_from_group_median":abs(pick["xy_error"]-median),
   "candidate_trials":15360,"survival_filter_used":False,"checkpoint_sha256":models[pick["model"]]["sha256"]}
 for group,old in json.loads((OLD/"representative_selection.json").read_text()).items():
  for k in ["model","training_seed","eval_seed","env_id","xy_error","group_median_xy"]:assert selection[group][k]==old[k],(group,k)
 save("representative_selection.json",selection);save("representative_selection_sha256.json",{"sha256":sha(ROOT/"representative_selection.json")})
 # Achieved-speed fall comparisons are explicitly descriptive and post-treatment conditioned.
 csvwrite("actual_speed_trial_rows.csv",bin_trials);binrows=[];binpairs=[]
 for cmd in FORWARD:
  bins=[]
  for b in range(22):
   pergroup={}
   label="<0" if b==0 else (">=1.0" if b==21 else f"[{(b-1)*.05:.2f},{b*.05:.2f})")
   for g in GROUPS:
    selected=[x for x in bin_trials if x["command"]==cmd and x["group"]==g and x["bin_id"]==b]
    ns=[sum(x["training_seed"]==s for x in selected) for s in range(1,6)];fs=[sum(x["fell"] for x in selected if x["training_seed"]==s) for s in range(1,6)]
    rates=[fs[i]/ns[i] if ns[i] else None for i in range(5)];allfive=all(n>0 for n in ns)
    pergroup[g]={"n":len(selected),"falls":sum(fs),"per_seed_n":ns,"per_seed_falls":fs,"per_seed_rates":rates,"all_five_seeds_supported":allfive,
     "seed_summary":summary(rates) if allfive else None,"pooled_rate":sum(fs)/len(selected) if selected else None,
     "pooled_mean_vx":float(np.mean([x["actual_mean_vx"] for x in selected])) if selected else None}
   common=all(v["n"]>0 for v in pergroup.values())
   for g,v in pergroup.items():binrows.append({"command":cmd,"bin_id":b,"actual_vx_bin":label,"group":g,"common_all3_groups":common,
    "n":v["n"],"falls":v["falls"],"pooled_rate":v["pooled_rate"],"pooled_mean_vx":v["pooled_mean_vx"],"all_five_seeds_supported":v["all_five_seeds_supported"],
    "seed_mean":None if v["seed_summary"] is None else v["seed_summary"]["mean"],"seed95_low":None if v["seed_summary"] is None else v["seed_summary"]["bootstrap95"][0],
    "seed95_high":None if v["seed_summary"] is None else v["seed_summary"]["bootstrap95"][1]})
   for reference,candidate in PAIRS:
    supported=pergroup[reference]["all_five_seeds_supported"] and pergroup[candidate]["all_five_seeds_supported"]
    d=compare(pergroup[reference]["per_seed_rates"],pergroup[candidate]["per_seed_rates"]) if supported else None
    binpairs.append({"command":cmd,"bin_id":b,"actual_vx_bin":label,"reference_group":reference,"candidate_group":candidate,"all_five_paired_seeds_supported":supported,"comparison":d})
   bins.append({"bin_id":b,"actual_vx_bin":label,"common_all3_groups":common,"groups":pergroup})
  save("actual_speed_bins_"+cmd+".json",{"fixed_bin_width_m_s":.05,"bins":bins,"post_treatment_conditioning":True,"causal_recovery_effect_identified":False})
 csvwrite("actual_speed_fall_bins.csv",binrows);save("actual_speed_bin_paired_comparisons.json",binpairs)
 speedfall=[]
 for condition in ["clean","combined"]:
  for cmd in FORWARD:
   for g in GROUPS:
    vx=summary(values(condition,cmd,SCOPES[0],"all","vx_mean",g));sd=summary(values(condition,cmd,SCOPES[0],"all","vx_temporal_sd_mean",g));fall=fall_results[condition+":"+cmd]["groups"][g]
    speedfall.append({"condition":condition,"command":cmd,"group":g,"mean_actual_vx":vx["mean"],"actual_vx95_low":vx["bootstrap95"][0],"actual_vx95_high":vx["bootstrap95"][1],
     "mean_temporal_sd_vx":sd["mean"],"falls":fall["pooled_falls"],"n":fall["pooled_n"],"fall_seed_mean":fall["seed_summary"]["mean"],
     "fall95_low":fall["seed_summary"]["bootstrap95"][0],"fall95_high":fall["seed_summary"]["bootstrap95"][1]})
 csvwrite("actual_speed_fall_table.csv",speedfall)
 context=[]
 for cmd in FORWARD:
  cases={}
  for reference,candidate in PAIRS:
   pair=candidate+"_vs_"+reference
   vx=get(pair,"combined",cmd,"all","vx_mean")["paired_difference"]
   fall=fall_results["combined:"+cmd]["paired"][pair]["paired_difference"]
   speed_status="slower" if vx["bootstrap95"][1]<0 else ("faster" if vx["bootstrap95"][0]>0 else "direction_unresolved_not_equivalence")
   fall_status="fewer_falls" if fall["bootstrap95"][1]<0 else ("more_falls" if fall["bootstrap95"][0]>0 else "fall_difference_unresolved")
   interpretation=("Lower achieved speed remains a plausible contributor to any observed lower fall rate." if speed_status=="slower" else
    ("This candidate is faster on the marginal mean; a lower marginal mean speed does not explain any observed fall reduction in these summaries. This is not proof of a recovery mechanism." if speed_status=="faster" else
     "The achieved-speed direction is unresolved; inclusion of zero is not evidence of speed equivalence, so a slowing contribution cannot be excluded."))
   if fall_status!="fewer_falls":interpretation="This contrast does not establish a fall-rate reduction, so it cannot demonstrate improved recovery. "+interpretation
   cases[pair]={"actual_mean_vx_difference":vx,"fall_rate_difference":fall,"speed_direction":speed_status,"fall_direction":fall_status,"interpretation":interpretation}
  bins=json.loads((ROOT/("actual_speed_bins_"+cmd+".json")).read_text())["bins"]
  common=[b for b in bins if b["common_all3_groups"]]
  coverage={g:sum(b["groups"][g]["n"] for b in common)/15360 for g in GROUPS}
  context.append({"command":cmd,"comparisons":cases,"common_three_group_bins":len(common),"fraction_trials_in_common_bins":coverage,
   "post_treatment_speed_bins_identify_causal_recovery":False})
 save("actual_speed_fall_interpretation.json",context)
 save("analysis_integrity_audit.json",{"passed":True,"new_formal_jobs":30,"historical_reused_jobs":60,"commands":360,"all_initial_trials":368640,"new_initial_trials":122880,
 "raw_command_audits":audits,"historical_policy_metric_rows_recomputed_exactly":True,"per_trial_csv_rows":sum(v["rows"] for v in json.loads((ROOT/"per_trial_csv_manifest.json").read_text()).values()),
 "all_three_group_representatives_unfiltered_true_medians":True,"secondary_speed_bins_prespecified_and_no_causal_claim":True})
 outputs=["evaluation_metric_rows.csv","evaluation_fall_rows.csv","policy_metric_rows.csv","paired_comparisons.json","paired_metric_comparisons.csv","fall_results.json","decision.json",
 "classification_sensitivity.json","representative_selection.json","per_trial_csv_manifest.json","analysis_integrity_audit.json","actual_speed_trial_rows.csv","actual_speed_fall_bins.csv",
 "actual_speed_bin_paired_comparisons.json","actual_speed_fall_table.csv","actual_speed_fall_interpretation.json"]+["actual_speed_bins_"+c+".json" for c in FORWARD]
 source=Path(__file__);reader=source.with_name("issue14_analysis_data.py");oldreader=source.with_name("issue13_analysis_data.py")
 assert reader.read_text()==oldreader.read_text().replace("from issue13_common import ROOT","from issue14_common import ROOT")
 save("analysis_outputs_sha256.json",{"outputs":{n:sha(ROOT/n) for n in outputs},"source_sha256":{str(p):sha(p) for p in [source,reader,oldreader,source.with_name("issue13_statistics.py")]},
 "reader_only_changes_output_root":True,"derived_code_not_claimed_prospectively_frozen":True})
 (ROOT/"analysis_complete").write_text(json.dumps({"primary_mitigation_supported":decision["primary_mitigation_supported"],"all_gates_pass":decision["all_gates_pass"]})+"\n")
 print("ISSUE14_ANALYSIS_COMPLETE",decision["primary_mitigation_supported"],decision["all_gates_pass"],flush=True)
if __name__=="__main__":main()
