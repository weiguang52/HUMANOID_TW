"""Full-denominator issue13 analysis after all60 frozen evaluation jobs."""
import csv,json,math,statistics,hashlib
from pathlib import Path
import numpy as np
from issue13_common import ROOT,CLEAN,FORWARD,sha
from issue13_statistics import compare,summary,classify
from issue13_tracking import SCOPES,PARTS
from issue13_analysis_data import load_command,pooled_statistics,env_statistics,TrialCsvs
from run_issue13 import verify_frozen
def save(name,value):(ROOT/name).write_text(json.dumps(value,indent=2,allow_nan=False))
def csvwrite(name,rows):
 rows=list(rows)
 with (ROOT/name).open("w",newline="") as f:
  if rows:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def main():
 assert (ROOT/"formal_evaluation_complete").exists();verify_frozen()
 state=json.loads((ROOT/"formal_job_state.json").read_text());assert len(state)==60 and all(x["passed"] for x in state.values())
 models=json.loads((ROOT/"input_models.json").read_text());evalrows=[];fallrows=[];audits=[];candidates={"control":[],"treatment":[]};exports=TrialCsvs()
 for condition in ["clean","combined"]:
  for name,m in models.items():
   for es in [1,2,3]:
    output=ROOT/condition/f"{name}_eval{es}.json";d=json.loads(output.read_text());folder=Path(d["issue13_tracking"]["output"])
    for result in d["results"]:
     command=result["command"];z,meta=load_command(folder,command,result)
     base={"model":name,"group":m["group"],"training_seed":m["training_seed"],"eval_seed":es}
     exports.write(condition,command,base,z,result["target"])
     fallrows.append({"condition":condition,"command":command,**base,"falls":result["failed_trials"],"n":1024})
     for si,scope in enumerate(SCOPES):
      for pi in ([0] if condition=="clean" else range(5)):
       record=pooled_statistics(z,si,pi,result["target"])
       evalrows.append({"condition":condition,"command":command,**base,"scope":scope,"partition":PARTS[pi],**record})
     if condition=="combined" and command=="forward_08":
      counts,means,_=env_statistics(z,0,0,result["target"]);assert (counts>0).all()
      candidates[m["group"]].extend({"model":name,"group":m["group"],"training_seed":m["training_seed"],"eval_seed":es,
       "env_id":i,"xy_error":float(means[i,0]),"source_output":str(output),"source_moments":str(folder/(command+".npz")),
       "first_failure_step":int(z["first_failure_step"][i])} for i in range(1024))
     audits.append({"condition":condition,"model":name,"eval_seed":es,"command":command,"trials":1024,
      "moments_sha256":sha(folder/(command+".npz")),"metadata_sha256":sha(folder/(command+".json")),"source_eval_sha256":sha(output),"source_moments":str(folder/(command+".npz")),"source_metadata":str(folder/(command+".json")),"source_eval":str(output),
      "all_packed_windows_reset_counts_independently_reconstructed":True,"all_sum_square_and_mixture_identities_passed":True,
      "sampled_reference_env_values_reconstruct_moments":True,"original_metrics_reconstructed":True})
 save("per_trial_csv_manifest.json",exports.close())
 assert len(audits)==240 and sum(x["trials"] for x in audits)==245760
 csvwrite("evaluation_metric_rows.csv",evalrows);csvwrite("evaluation_fall_rows.csv",fallrows)
 metrics=[k for k in evalrows[0] if k not in ["condition","command","model","group","training_seed","eval_seed","scope","partition"]]
 policyrows=[]
 for condition in ["clean","combined"]:
  for name,m in models.items():
   for command in (CLEAN if condition=="clean" else FORWARD):
    for scope in SCOPES:
     for partition in (["all"] if condition=="clean" else PARTS):
      selected=[x for x in evalrows if x["condition"]==condition and x["model"]==name and x["command"]==command and x["scope"]==scope and x["partition"]==partition]
      assert len(selected)==3
      policyrows.append({"condition":condition,"model":name,"group":m["group"],"training_seed":m["training_seed"],"command":command,"scope":scope,"partition":partition,
       **{key:float(np.mean([x[key] for x in selected])) for key in metrics}})
 csvwrite("policy_metric_rows.csv",policyrows)
 comparisons={};flat=[]
 signed={"vx_mean","vy_mean","wz_mean","vx_bias"}
 def key(condition,command,scope,partition,metric):return ":".join([condition,command,scope,partition,metric])
 for condition in ["clean","combined"]:
  for command in (CLEAN if condition=="clean" else FORWARD):
   for scope in SCOPES:
    for partition in (["all"] if condition=="clean" else PARTS):
     for metric in metrics:
      groups={}
      for group in ["control","treatment"]:
       groups[group]=[next(x[metric] for x in policyrows if x["condition"]==condition and x["command"]==command and x["scope"]==scope and x["partition"]==partition and x["group"]==group and x["training_seed"]==seed) for seed in range(1,6)]
      result=compare(groups["control"],groups["treatment"]);k=key(condition,command,scope,partition,metric)
      if metric in signed:
       result["relative_change_not_interpreted_for_signed_metric"]=True
       for field in ["ratio_of_group_means","relative_change","relative_change_bootstrap95","point_not_worse_than5percent","point_worse_than5percent_and_positive95"]:result[field]=None
      comparisons[k]=result
      ci=result["paired_difference"]["bootstrap95"]
      flat.append({"condition":condition,"command":command,"scope":scope,"partition":partition,"metric":metric,
       "control_mean":result["control"]["mean"],"control_seed_sd":result["control"]["sample_sd"],
       "treatment_mean":result["treatment"]["mean"],"treatment_seed_sd":result["treatment"]["sample_sd"],
       "paired_difference":result["paired_difference"]["mean"],"paired95_low":ci[0],"paired95_high":ci[1],
       "relative_change":result["relative_change"],"relative95_low":None if result["relative_change_bootstrap95"] is None else result["relative_change_bootstrap95"][0],
       "relative95_high":None if result["relative_change_bootstrap95"] is None else result["relative_change_bootstrap95"][1],"multiple_comparisons_corrected":False})
 save("paired_comparisons.json",comparisons);csvwrite("paired_metric_comparisons.csv",flat)
 def gate_set(scope=SCOPES[0],outside="outside"):
  selected={}
  for command in CLEAN:
   for metric in ["xy_error","yaw_error"]:
    k=key("clean",command,scope,"all",metric);selected[k]=comparisons[k]
  for command in FORWARD:
   k=key("combined",command,scope,outside,"xy_error");selected[k]=comparisons[k]
  assert len(selected)==13
  return selected
 primary=gate_set();decision=classify(primary);save("primary_gate_comparisons.json",primary)
 sensitivity={"reward_aligned_boundary":classify(gate_set(outside="reward_aligned_outside")),
  "initial_episode_only":classify(gate_set(scope=SCOPES[1])),
  "primary_not_reselected":True}
 save("classification_sensitivity.json",sensitivity)
 decision["primary_scope"]=SCOPES[0];decision["primary_window"]="post-step [push,push+1s), ages0..49"
 if decision["decision"].startswith("A_"):
  recommendation={"one_recommendation":"Ask the user whether future tracking acceptance should be evaluated under clean/outside-window conditions; do not change the criterion here.","implemented":False,"basis":"All13prospective point degradation bounds are <=5%; statistical uncertainty and boundary/initial-episode sensitivities remain reported."}
 elif decision["decision"].startswith("B_"):
  recommendation={"one_recommendation":"In a future user-authorized matched study, add the push-recovery clock to the critic only to test phase-aware value/credit assignment; do not implement it in this issue.","implemented":False,"basis":"At least one frozen clean/outside tracking contrast worsens >5% with positive paired95% absolute difference. Reward timing is privileged but currently absent from critic observations; this is a structural hypothesis, not a proven remedy."}
 else:
  recommendation={"one_recommendation":"Retain the issue12 rejection and seek user authorization for additional independent training-seed evidence before choosing a structural change.","implemented":False,"basis":"Neither all13point bounds nor any confirmed >5% degradation criterion establishes the frozen alternatives."}
 decision["only_next_step_recommendation"]=recommendation
 save("decision.json",decision)
 fall_groups={}
 for condition in ["clean","combined"]:
  for command in (CLEAN if condition=="clean" else FORWARD):
   groups={}
   for group in ["control","treatment"]:
    counts=[sum(x["falls"] for x in fallrows if x["condition"]==condition and x["command"]==command and x["group"]==group and x["training_seed"]==seed) for seed in range(1,6)]
    groups[group]={"per_seed_falls":counts,"per_seed_n":3072,"pooled_falls":sum(counts),"pooled_n":15360,"seed_summary":summary(np.asarray(counts)/3072)}
   c=np.asarray(groups["control"]["per_seed_falls"]);t=np.asarray(groups["treatment"]["per_seed_falls"])
   groups["paired_difference"]=summary((t-c)/3072);fall_groups[condition+":"+command]=groups
 save("fall_results.json",fall_groups)
 selection={}
 for group,rows in candidates.items():
  assert len(rows)==15360;median=float(np.median([x["xy_error"] for x in rows]))
  chosen=min(rows,key=lambda x:(abs(x["xy_error"]-median),x["training_seed"],x["eval_seed"],x["env_id"]))
  selection[group]={**chosen,"command":"forward_08","condition":"combined","group_median_xy":median,
   "distance_from_group_median":abs(chosen["xy_error"]-median),"candidate_trials":15360,"survival_filter_used":False,
   "selection":"Nearest full-horizon nonterminal trial-mean xy error to group median; ties training seed,eval seed,env id. Illustrative median, not an acceptance sample.",
   "checkpoint_sha256":models[chosen["model"]]["sha256"]}
 save("representative_selection.json",selection);save("representative_selection_sha256.json",{"sha256":sha(ROOT/"representative_selection.json")})
 save("analysis_integrity_audit.json",{"passed":True,"formal_jobs":60,"commands":240,"initial_trials":245760,
  "all_original_combined_issue12_outputs_reproduced":all(x["every_original_issue12_field_and_all_trial_failure_arrays_exactly_reproduced"] for x in state.values() if x["condition"]=="combined"),
  "raw_command_audits":audits,"per_trial_primary_csv_rows":sum(v["rows"] for v in json.loads((ROOT/"per_trial_csv_manifest.json").read_text()).values()),
  "primary_and_two_prespecified_sensitivities_reported":True,"training_updates":0,"default_promoted":False})
 outputs=["evaluation_metric_rows.csv","evaluation_fall_rows.csv","policy_metric_rows.csv","paired_comparisons.json","paired_metric_comparisons.csv","primary_gate_comparisons.json","classification_sensitivity.json","decision.json","fall_results.json","representative_selection.json","per_trial_csv_manifest.json","analysis_integrity_audit.json"]
 save("analysis_outputs_sha256.json",{"outputs":{name:sha(ROOT/name) for name in outputs},"analysis_source_sha256":sha(Path(__file__)),"independent_reader_source_sha256":sha(Path(__file__).with_name("issue13_analysis_data.py")),"statistics_source_sha256":sha(Path(__file__).with_name("issue13_statistics.py"))})
 (ROOT/"analysis_complete").write_text(decision["decision"]+"\n")
 print("ISSUE13_ANALYSIS_COMPLETE",decision["decision"],"COMPARISONS",len(comparisons),flush=True)
if __name__=="__main__":main()
