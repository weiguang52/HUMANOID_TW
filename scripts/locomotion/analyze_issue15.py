"""Analyze the frozen complete issue15 lineage, retaining training-seed inference and raw audits."""
import csv,gzip,itertools,json,math,os,time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from issue15_common import ROOT,REPO,COMMANDS,sha
from issue15_analysis_data import load_yaw,job_statistics
from issue15_math import paired_summary,spectrum,contact_frequencies
from issue15_historical_consistency import compare_job,summarize as summarize_historical
from run_issue15 import verify_frozen,audit_eval
FORWARD=["forward_03","forward_05","forward_08"]
def save(n,x):(ROOT/n).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def aggregate(rows):
 keys=sorted(set.intersection(*(set(x) for x in rows)))
 return {k:None if any(x[k] is None for x in rows) else float(np.mean([x[k] for x in rows])) for k in keys}
def main():
 os.chdir(REPO);verify_frozen();assert (ROOT/"formal_evaluation_complete").exists()
 cohort=json.loads((ROOT/"cohort.json").read_text());jobs={};audits=[];raw_hashes={};spectra={};historical=[];representatives={"ft5":[],"control6000":[]}
 grid=np.arange(.2,10.00001,.05);folder=ROOT/"per_trial";folder.mkdir(exist_ok=True)
 fields=["model","node","training_seed","eval_seed","command","env_id","samples","first_failure_step","yaw_mae","xy_mae","yaw_mean","yaw_sd","heading_all_segments_degrees","initial_prefix_heading_degrees","initial_prefix_seconds","full20s_survivor_heading_degrees","yaw_peak_hz","frequency_resolution_hz","left_stride_hz","right_stride_hz","step_hz","peak_over_stride","peak_over_step","spectrum_supported","fullband_yaw_peak_hz","fullband_peak_over_stride","fullband_peak_over_step","fullband_spectrum_supported"]
 pertrial_paths={cmd:folder/(cmd+".csv.gz") for cmd in COMMANDS};pertrial_rows={cmd:0 for cmd in COMMANDS};rowcount=0
 with ExitStack() as stack:
  writers={}
  for cmd,path in pertrial_paths.items():
   stream=stack.enter_context(gzip.open(path,"wt",newline=""))
   writers[cmd]=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writers[cmd].writeheader()
  for name,m in cohort["unique_evaluation_models"].items():
   for es in [1,2,3]:
    path=ROOT/"clean"/f"{name}_eval{es}.json";d=json.loads(path.read_text());audit_eval(path,m,es)
    raw_hashes[str(path)]=sha(path);out=Path(d["issue15_tracking"]["output"])
    if m["node"] in ["control6000","treatment","clock"]:historical.append(compare_job(m,es,path))
    for row in d["results"]:
     cmd=row["command"];z,y,masks,meta=load_yaw(out,cmd,row);stat=job_statistics(z,y,meta,row)
     for p in [out/(cmd+".npz"),out/(cmd+".json"),out/(cmd+"_yaw.npz"),out/(cmd+"_yaw.json")]:raw_hashes[str(p)]=sha(p)
     n=meta["num_envs"];counts=z["counts"][0,0];means=np.divide(z["sums"][0,0],counts[:,None],out=np.full_like(z["sums"][0,0],np.nan),where=counts[:,None]>0)
     var=np.maximum(np.divide(z["sumsq"][0,0],counts[:,None],out=np.full_like(z["sums"][0,0],np.nan),where=counts[:,None]>0)-means**2,0)
     contacts=np.unpackbits(y["packed_foot_contacts"],axis=1,count=n,bitorder="little").astype(bool) if cmd in FORWARD else None
     peaks=[];steps=[];ratios_stride=[];ratios_step=[];powers=[];support=0;fullpeaks=[];full_stride=[];full_step=[]
     for env in range(n):
      peak=None;freqres=None;gait=None;ratio_stride=None;ratio_step=None;fullpeak=None;full_ratio_stride=None;full_ratio_step=None
      if cmd in FORWARD:
       L=int(z["counts"][1,0,env])
       if L>=250:
        freqres=1/(L*.02)
        assert masks[:L,1,0,env].all() and not masks[L:,1,0,env].any()
        spec=spectrum(y["wz"][:L,env]);fullspec=spectrum(y["wz"][:L,env],lo=1e-12,hi=25.);gait=contact_frequencies(contacts[:L,env])
        if fullspec is not None:
         fullpeak=fullspec["peak_hz"];fullpeaks.append(fullpeak)
         stride_full=(gait["left_stride_hz"]+gait["right_stride_hz"])/2
         if stride_full>0:full_ratio_stride=fullpeak/stride_full;full_stride.append(full_ratio_stride)
         if gait["step_hz"]>0:full_ratio_step=fullpeak/gait["step_hz"];full_step.append(full_ratio_step)
        if spec is not None:
         peak=spec["peak_hz"];freqres=spec["resolution_hz"];peaks.append(peak);support+=1
         normal=spec["power"]/max(float(spec["power"].sum()),1e-30)
         powers.append(np.interp(grid,spec["frequencies"],normal))
         stride=(gait["left_stride_hz"]+gait["right_stride_hz"])/2
         if stride>0:ratio_stride=peak/stride;ratios_stride.append(ratio_stride)
         if gait["step_hz"]>0:ratio_step=peak/gait["step_hz"];ratios_step.append(ratio_step)
        steps.append(gait["step_hz"])
      survived=y["first_failure_step"][env]<0
      record={"model":name,"node":m["node"],"training_seed":m["training_seed"],"eval_seed":es,"command":cmd,"env_id":env,"samples":int(counts[env]),"first_failure_step":int(y["first_failure_step"][env]),
       "yaw_mae":float(means[env,1]),"xy_mae":float(means[env,0]),"yaw_mean":float(means[env,4]),"yaw_sd":float(np.sqrt(var[env,4])),
       "heading_all_segments_degrees":float(y["heading_delta_sum"][0,env]*180/np.pi),"initial_prefix_heading_degrees":float(y["heading_delta_sum"][1,env]*180/np.pi),
       "initial_prefix_seconds":float(y["heading_count"][1,env]*meta["dt"]),"full20s_survivor_heading_degrees":float(y["heading_delta_sum"][0,env]*180/np.pi) if survived else None,
       "yaw_peak_hz":peak,"frequency_resolution_hz":freqres,"left_stride_hz":gait["left_stride_hz"] if gait else None,"right_stride_hz":gait["right_stride_hz"] if gait else None,"step_hz":gait["step_hz"] if gait else None,
       "peak_over_stride":ratio_stride,"peak_over_step":ratio_step,"spectrum_supported":peak is not None,"fullband_yaw_peak_hz":fullpeak,"fullband_peak_over_stride":full_ratio_stride,"fullband_peak_over_step":full_ratio_step,"fullband_spectrum_supported":fullpeak is not None}
      writers[cmd].writerow(record);pertrial_rows[cmd]+=1;rowcount+=1
      if cmd=="forward_03" and m["node"] in representatives:
       representatives[m["node"]].append({"model":name,"training_seed":m["training_seed"],"eval_seed":es,"env_id":env,"yaw_error":float(means[env,1]),"first_failure_step":int(y["first_failure_step"][env]),"checkpoint":m["checkpoint"],"checkpoint_sha256":m["sha256"],"output":str(path)})
     if cmd in FORWARD:
      stat.update(spectrum_supported_trials=support,spectrum_total_trials=n,fullband_spectrum_supported_trials=len(fullpeaks),
       fullband_yaw_peak_hz_mean=float(np.mean(fullpeaks)) if fullpeaks else None,
       fullband_peak_over_stride_mean=float(np.mean(full_stride)) if full_stride else None,fullband_peak_over_step_mean=float(np.mean(full_step)) if full_step else None,
       fullband_peak_outside_planned_band_fraction=float(np.mean([(x<.2 or x>10) for x in fullpeaks])) if fullpeaks else None,
       yaw_peak_hz_mean=float(np.mean(peaks)) if peaks else None,yaw_peak_hz_median=float(np.median(peaks)) if peaks else None,
       footstrike_step_hz_mean=float(np.mean(steps)) if steps else None,
       peak_over_stride_mean=float(np.mean(ratios_stride)) if ratios_stride else None,peak_over_step_mean=float(np.mean(ratios_step)) if ratios_step else None,
       fraction_near_stride1x_or2x=float(np.mean([min(abs(x-1),abs(x/2-1))<=.2 for x in ratios_stride])) if ratios_stride else None)
      spectra[f"{name}:eval{es}:{cmd}"]={"supported":support,"total":n,"frequency_grid_hz":grid.tolist(),"mean_normalized_power":np.mean(powers,axis=0).tolist() if powers else None,"fullband_supported":len(fullpeaks),"fullband_definition":"All strictly positive resolvable FFT frequencies through Nyquist25Hz; supplementary coverage, not a changed frozen decision.","scope":"Initial uninterrupted episode >=5s; no reset concatenation or missing-sample fill."}
     jobs[f"{name}:eval{es}:{cmd}"]=stat;audits.append({"model":name,"eval_seed":es,"command":cmd,"passed":True})
   print("ANALYZED",name,flush=True)
 assert rowcount==645120 and all(n==107520 for n in pertrial_rows.values())
 save("historical_clean_reproduction.json",summarize_historical(historical))
 pertrial_manifest={"files":{str(path.relative_to(ROOT)):{"rows":pertrial_rows[cmd],"sha256":sha(path)} for cmd,path in pertrial_paths.items()},"total_rows":rowcount,"schema":fields,"scope":"Six per-command CSV gzip files, all35 unique policies x3eval seeds x1024trials. No alias double-counting."}
 save("per_trial_manifest.json",pertrial_manifest)
 nodes={};seed_values={}
 for node,names in cohort["nodes"].items():
  nodes[node]={};seed_values[node]={}
  for cmd in COMMANDS:
   per={}
   for name in names:
    m=cohort["models"][name];canonical=m["evaluation_model"];seed=m["training_seed"]
    per[str(seed)]=aggregate([jobs[f"{canonical}:eval{es}:{cmd}"] for es in [1,2,3]])
   seed_values[node][cmd]=per;keys=set.intersection(*(set(x) for x in per.values()));summary={}
   for k in sorted(keys):
    a=[x[k] for x in per.values()];summary[k]=None if any(x is None for x in a) else paired_summary(a)
   nodes[node][cmd]={"training_seeds":len(per),"seed_ids":list(map(int,per)),"metrics":summary}
 comparisons=[]
 for previous,current in cohort["comparison_edges"]:
  for cmd in COMMANDS:
   a=seed_values[previous][cmd];b=seed_values[current][cmd]
   key="full_yaw_error";am=float(np.mean([x[key] for x in a.values()]));bm=float(np.mean([x[key] for x in b.values()]));increase=bm/am-1
   common=sorted(set(a)&set(b),key=int)
   if len(a)==1 and len(b)>1:
    baseline=next(iter(a.values()))[key];changes=[x[key]-baseline for x in b.values()];method="later training seeds versus FIXED single-seed baseline; conditional only";used=list(b)
   else:
    changes=[b[s][key]-a[s][key] for s in common];method="paired shared training seeds";used=common
   assert changes
   summary=paired_summary(changes);ci=summary["bootstrap95"]
   alias=previous=="bounded" and current=="requested1500"
   if alias:assert all(x==0 for x in changes)
   ci_above=ci is not None and ci[0]>0
   qualified=increase>.20 and (len(b)==1 or ci_above) and not alias
   comparisons.append({"previous":previous,"current":current,"command":cmd,"prospectively_frozen_straight_comparison":cmd in FORWARD,"previous_all_node_mean":am,"current_all_node_mean":bm,"relative_increase":increase,"increase_strictly_above20percent":increase>.20,"difference":summary,"comparison_method":method,"paired_or_resampled_seed_ids":used,
    "paired_subset_differs_from_full_node_composition":len(a)!=len(b) and len(a)>1,"interval_excludes_zero_above":ci_above,"meets_issue15_descriptive_or_multiseed_criterion":qualified,"single_seed_descriptive_only":len(b)==1,"alias_cohort_expansion_not_configuration_effect":alias,
    "causal_attribution":"Requires actual config/parent audit; diagnostic edge alone cannot identify unique cause."})
 classifications={}
 for node in cohort["nodes"]:
  classifications[node]={}
  for cmd in FORWARD:
   s=nodes[node][cmd]["metrics"]["full_yaw_within_energy_fraction"]
   if s is None:label="unsupported_zero_energy"
   elif s["bootstrap95"] is None:label="single_seed_descriptive_oscillation" if s["mean"]>.5 else ("single_seed_descriptive_curve" if s["mean"]<.5 else "dominance_uncertain")
   elif s["bootstrap95"][0]>.5:label="oscillation_dominant"
   elif s["bootstrap95"][1]<.5:label="curve_dominant"
   else:label="dominance_uncertain"
   classifications[node][cmd]={"classification":label,"within_energy_fraction":s,"caveat":"Body-z rate decomposition, not a causal gait diagnosis; full20s includes ordinary post-failure episodes."}
 selected=[]
 for group,rows in representatives.items():
  median=float(np.median([x["yaw_error"] for x in rows]));best=min(rows,key=lambda x:(abs(x["yaw_error"]-median),x["training_seed"],x["eval_seed"],x["env_id"]))
  selected.append(dict(best,group=group,group_median_yaw_error=median,distance_from_group_median=abs(best["yaw_error"]-median),selection_population=len(rows),survival_filter=False))
 legacy=[]
 for es in [1,2,3]:
  path=ROOT/"legacy"/f"ft5_eval{es}.json";new=json.loads(path.read_text());audit_eval(path,cohort["models"]["ft5_s1"],es,n=128,legacy=True)
  old=json.loads((REPO/"validation_artifacts/stage1_issues_20261108"/f"ft5_seed{es}.json").read_text());differences=[];rows=[]
  assert [x["command"] for x in new["results"]]==[x["command"] for x in old["results"]]
  for a,b in zip(old["results"],new["results"]):
   assert a["num_envs"]==b["num_envs"] and a["seconds"]==b["seconds"]
   md={k:float(b["metrics"][k]-a["metrics"][k]) for k in a["metrics"]};differences+=list(md.values())
   rows.append({"command":a["command"],"original_yaw":a["metrics"]["yaw_error"],"rerun_yaw":b["metrics"]["yaw_error"],"all_original_metric_differences":md,"original_falls":a["failed_trials"],"rerun_falls":b["failed_trials"],"original_minimum_height":a["minimum_base_height"],"rerun_minimum_height":b["minimum_base_height"],"all_original_result_fields_exact":all(b[k]==v for k,v in a.items() if k!="metrics") and all(v==0 for v in md.values())})
  legacy.append({"eval_seed":es,"original_formula_unchanged":True,"max_absolute_original_metric_difference":max(map(abs,differences)),"exact_original_metrics_reproduced":all(x==0 for x in differences),"exact_original_result_fields_reproduced":all(x["all_original_result_fields_exact"] for x in rows),"rows":rows,"original_checkpoint_sha256":sha(REPO/"validation_artifacts/tw_stage1_ft5_20261107/model_5997.pt")})
 save("job_statistics.json",jobs);save("seed_statistics.json",seed_values);save("node_statistics.json",nodes);save("lineage_comparisons.json",comparisons);save("yaw_dominance.json",classifications);save("spectra.json",spectra);save("representative_selection.json",selected);save("original_ft5_protocol_comparison.json",legacy)
 fields2=["node","command","metric","training_seeds","mean","sd","ci95_low","ci95_high"]
 with (ROOT/"lineage_metrics.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields2,lineterminator="\n");w.writeheader()
  for node in cohort["display_order"]:
   for cmd,d in nodes[node].items():
    for k,s in d["metrics"].items():
     if s is not None:w.writerow({"node":node,"command":cmd,"metric":k,"training_seeds":s["training_seeds"],"mean":s["mean"],"sd":s["sd"],"ci95_low":s["bootstrap95"][0] if s["bootstrap95"] else None,"ci95_high":s["bootstrap95"][1] if s["bootstrap95"] else None})
 save("analysis_integrity_audit.json",{"passed":True,"unique_clean_jobs":105,"commands":len(audits),"initial_trials":rowcount,"all_moment_mask_metric_and_heading_reconstructions":True,"checks":audits,"historical_FT5_protocols":3,"training_updates":0,"per_trial_manifest_sha256":sha(ROOT/"per_trial_manifest.json"),"per_trial_files":pertrial_manifest["files"],"per_trial_rows":rowcount,"raw_input_sha256":raw_hashes})
 outputs=["job_statistics.json","seed_statistics.json","node_statistics.json","lineage_comparisons.json","yaw_dominance.json","spectra.json","representative_selection.json","original_ft5_protocol_comparison.json","lineage_metrics.csv","per_trial_manifest.json","historical_clean_reproduction.json","analysis_integrity_audit.json"]
 sources=[Path(__file__),Path(__file__).with_name("issue15_analysis_data.py"),Path(__file__).with_name("issue15_math.py"),Path(__file__).with_name("issue13_analysis_data.py"),Path(__file__).with_name("issue15_historical_consistency.py")]
 save("analysis_outputs_sha256.json",{"sources":{str(p):sha(p) for p in sources},"outputs":{n:sha(ROOT/n) for n in outputs},"per_trial":{str(p):sha(p) for p in pertrial_paths.values()},"derived_after_evaluation_freeze":True})
 (ROOT/"analysis_complete").write_text("All frozen lineage tables, decompositions, comparisons and raw evidence independently reconstructed.\n")
 print("ISSUE15_ANALYSIS_COMPLETE",len(audits),rowcount,flush=True)
if __name__=="__main__":main()
