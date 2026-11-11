"""Validate every issue15 deliverable before archive/publication; manual review cannot be inferred."""
import csv,gzip,itertools,json,os
from pathlib import Path
import numpy as np
from issue15_common import ROOT,REPO,sha,COMMANDS
from run_issue15 import verify_frozen,audit_eval
from issue15_math import paired_summary
from issue15_analysis_data import load_yaw
def load(n):return json.loads((ROOT/n).read_text())
def main():
 os.chdir(REPO);verify_frozen()
 assert (ROOT/"formal_evaluation_complete").exists() and (ROOT/"automated_postprocessing_complete").exists()
 cohort=load("cohort.json");assert len(cohort["models"])==37 and len(cohort["unique_evaluation_models"])==35
 total=0;checks=0;spectral_reference_expected={}
 for name,m in cohort["unique_evaluation_models"].items():
  for es in [1,2,3]:
   p=ROOT/"clean"/f"{name}_eval{es}.json";a=audit_eval(p,m,es);total+=a["initial_trials"];d=json.loads(p.read_text())
   for row in d["results"]:
    z,y,masks,meta=load_yaw(d["issue15_tracking"]["output"],row["command"],row);checks+=1
    if row["command"].startswith("forward"):
     contact=np.unpackbits(y["packed_foot_contacts"],axis=1,count=1024,bitorder="little").astype(bool)
     for env in [0,1,17,1023]:
      L=int(z["counts"][1,0,env]);expected={"fullband_yaw_peak_hz":None,"yaw_peak_hz":None,"frequency_resolution_hz":None,"step_hz":None,"left_stride_hz":None,"right_stride_hz":None,"fullband_peak_over_step":None,"fullband_peak_over_stride":None}
      if L>=250:
       x=y["wz"][:L,env].astype(np.float64);power=abs(np.fft.rfft((x-x.mean())*np.hanning(L)))**2;freq=np.fft.rfftfreq(L,.02)
       for field,mask in [("fullband_yaw_peak_hz",freq>0),("yaw_peak_hz",(freq>=.2)&(freq<=10))]:
        indices=np.flatnonzero(mask)
        if power[mask].sum()>1e-20:expected[field]=float(freq[indices[np.argmax(power[mask])]])
       expected["frequency_resolution_hz"]=1/(L*.02);counts=[]
       for foot in [0,1]:
        a=contact[:L,env,foot];rises=np.flatnonzero(a[1:]&~a[:-1])+1;last=-10**9;nstrike=0
        for t in rises:
         if t-last>=5:nstrike+=1;last=t
        counts.append(nstrike)
       expected["left_stride_hz"]=counts[0]/((L-1)*.02);expected["right_stride_hz"]=counts[1]/((L-1)*.02);expected["step_hz"]=sum(counts)/((L-1)*.02)
       if expected["fullband_yaw_peak_hz"] is not None:
        if expected["step_hz"]>0:expected["fullband_peak_over_step"]=expected["fullband_yaw_peak_hz"]/expected["step_hz"]
        stride=sum(counts)/(2*(L-1)*.02)
        if stride>0:expected["fullband_peak_over_stride"]=expected["fullband_yaw_peak_hz"]/stride
      spectral_reference_expected[(name,es,row["command"],env)]=expected
 for es in [1,2,3]:total+=audit_eval(ROOT/"legacy"/f"ft5_eval{es}.json",cohort["models"]["ft5_s1"],es,n=128,legacy=True)["initial_trials"]
 assert total==647424 and checks==630 and len(load("formal_jobs.json"))==108
 analysis=load("analysis_integrity_audit.json");assert analysis["passed"] and analysis["commands"]==630 and analysis["per_trial_rows"]==645120
 for p,h in analysis["raw_input_sha256"].items():assert sha(p)==h,p
 manifest=load("per_trial_manifest.json");assert manifest["total_rows"]==645120 and len(manifest["files"])==6
 assert sha(ROOT/"per_trial_manifest.json")==analysis["per_trial_manifest_sha256"] and manifest["files"]==analysis["per_trial_files"]
 seen=set();rows=0
 for name,info in manifest["files"].items():
  pertrial=ROOT/name;assert sha(pertrial)==info["sha256"];file_rows=0
  with gzip.open(pertrial,"rt") as f:
   for row in csv.DictReader(f):
    rows+=1;file_rows+=1;key=(row["model"],int(row["eval_seed"]),row["command"],int(row["env_id"]))
    if key in spectral_reference_expected:
     assert key not in seen;seen.add(key)
     for field,value in spectral_reference_expected[key].items():
      if value is None:assert row[field]=="",(key,field)
      else:assert abs(float(row[field])-value)<1e-10,(key,field)
  assert file_rows==info["rows"]==107520
 assert rows==645120 and len(seen)==len(spectral_reference_expected)==1260
 testlog=(ROOT/"supplementary_spectral_tests.log").read_text();assert "Ran 2 tests" in testlog and testlog.rstrip().endswith("OK")
 for name in ["analysis_outputs_sha256.json","figure_outputs_sha256.json","trajectory_figure_sha256.json"]:
  x=load(name)
  for k in ["sources","input_trajectories","per_trial"]:
   for p,h in x.get(k,{}).items():assert sha(p)==h,p
  for k in ["outputs","inputs"]:
   for p,h in x.get(k,{}).items():assert sha(ROOT/p)==h,p
 seed=load("seed_statistics.json");nodes=load("node_statistics.json")
 for node,commands in seed.items():
  for cmd,rows in commands.items():
   assert len(rows)==len(cohort["nodes"][node])
   for key,s in nodes[node][cmd]["metrics"].items():
    if s is not None:assert paired_summary([r[key] for r in rows.values()])==s,(node,cmd,key)
 for comparison in load("lineage_comparisons.json"):
  a=seed[comparison["previous"]][comparison["command"]];b=seed[comparison["current"]][comparison["command"]];key="full_yaw_error"
  ratio=np.mean([r[key] for r in b.values()])/np.mean([r[key] for r in a.values()])-1
  assert abs(ratio-comparison["relative_increase"])<1e-14
  if len(a)==1 and len(b)>1:delta=[r[key]-next(iter(a.values()))[key] for r in b.values()]
  else:delta=[b[k][key]-a[k][key] for k in sorted(set(a)&set(b),key=int)]
  assert paired_summary(delta)==comparison["difference"]
  ci=comparison["difference"]["bootstrap95"]
  expected=ratio>.2 and (len(b)==1 or ci is not None and ci[0]>0) and not comparison["alias_cohort_expansion_not_configuration_effect"]
  assert expected==comparison["meets_issue15_descriptive_or_multiseed_criterion"]
 selected=load("representative_selection.json");assert len(selected)==2
 for s in selected:
  candidates=[]
  for name in cohort["nodes"][s["group"]]:
   m=cohort["models"][name]
   for es in [1,2,3]:
    d=json.loads((ROOT/"clean"/f"{m['evaluation_model']}_eval{es}.json").read_text());folder=Path(d["issue15_tracking"]["output"])
    with np.load(folder/"forward_03.npz") as z:values=z["sums"][0,0,:,1]/z["counts"][0,0]
    candidates += [(float(v),m["training_seed"],es,i,name) for i,v in enumerate(values)]
  median=float(np.median([x[0] for x in candidates]));best=min(candidates,key=lambda x:(abs(x[0]-median),*x[1:4]))
  assert (s["yaw_error"],s["training_seed"],s["eval_seed"],s["env_id"],s["model"])==best and s["group_median_yaw_error"]==median and not s["survival_filter"]
 reconstruction=load("reconstruction_state.json")
 for group,s in reconstruction.items():
  assert s["all_original_metrics_raw_records_and_moments_exactly_reproduced"] and s["formal_denominators_unchanged"] and sha(s["trajectory"])==s["trajectory_sha256"]
 media=load("media_source_sha256.json")
 for key in ["sources","assets"]:
  for p,h in media[key].items():assert sha(p)==h
 assert sha(ROOT/"video_plan.json")==media["video_plan_sha256"]
 plan=load("video_plan.json");assert len(plan)==2
 for entry in plan:
  for e in entry["views"].values():
   assert sha(e["path"])==e["sha256"]
   metadata=json.loads(Path(e["metadata"]).read_text())
   assert metadata["replay_source_sha256"]==sha("scripts/locomotion/replay_issue15.py") and metadata["render_adapter_sha256"]==sha("scripts/locomotion/render_eval_issue15.py")
   assert metadata["source_trajectory_sha256"]==reconstruction[entry["group"]]["trajectory_sha256"] and metadata["command"]=="forward_03" and not metadata["physics_integrated"]
 video=load("video_quality_audit.json");assert video["passed"] and len(video["videos"])==4 and video["sampled_frames"]==28 and video["contact_sheets"]==2
 manual=load("video_physics_manual_audit.json")
 assert manual["passed"] and manual["reviewed_plot_count"]==6 and manual["reviewed_contact_sheets"]==2
 assert manual["front_and_side_colors_fullbody_and_attachment_samples_pass"] and manual["world_XY_heading_inset_readable"]
 assert len(manual["full_resolution_motion_frames_reviewed"])==8
 assert len(manual["reviewed_plot_sha256"])==6 and len(manual["reviewed_frame_and_sheet_sha256"])==10
 for key in ["reviewed_plot_sha256","reviewed_frame_and_sheet_sha256"]:
  for p,h in manual[key].items():assert sha(ROOT/p)==h
 history=load("historical_clean_reproduction.json");assert history["comparison_completed"] and history["job_count"]==45 and history["command_count"]==225
 from issue15_historical_consistency import committed_manifest
 for job in history["jobs"]:
  assert job["comparison_completed"] and len(job["commands"])==5
  for path,h in job["committed_old_input_sha256"].items():
   assert sha(path)==h
   p=Path(path);parts=p.relative_to(REPO/"validation_artifacts").parts;assert committed_manifest(parts[0])["/".join(parts[1:])]==h
  current=json.loads(Path(job["new_job"]).read_text());old=json.loads(Path(job["original_job"]).read_text());rows={x["command"]:x for x in current["results"]}
  for before,reported in zip(old["results"],job["commands"]):
   now=rows[before["command"]];assert reported["original_row_exactly_reproduced"]==(before==now)
   assert reported["metric_differences"]=={k:now["metrics"][k]-v for k,v in before["metrics"].items()}
 ancestry=load("actual_resume_lineage.json");assert ancestry["passed"]
 assert sha("scripts/locomotion/audit_issue15_ancestry.py")==ancestry["source_sha256"]
 for h,r in ancestry["records_by_checkpoint_sha256"].items():
  assert sha(r["checkpoint"])==h
  for kind in ["agent","env"]:
   if kind+"_config_sha256" in r:assert sha(Path(r["run_dir"])/"params"/(kind+".yaml"))==r[kind+"_config_sha256"]
 config=load("configuration_lineage_audit.json")
 for r in config["models"].values():
  for c in r["config_files"].values():assert sha(c["path"])==c["sha256"]
 original_comparisons=load("original_ft5_protocol_comparison.json")
 assert len(original_comparisons)==3
 for result in original_comparisons:
  es=result["eval_seed"]
  oldpath=REPO/"validation_artifacts/stage1_issues_20261108"/f"ft5_seed{es}.json"
  import subprocess
  assert oldpath.read_bytes()==subprocess.check_output(["git","show",load("cohort.json")["base_commit"]+":"+str(oldpath.relative_to(REPO))])
  old=json.loads(oldpath.read_text());current=load(f"legacy/ft5_eval{es}.json")
  for a,b,r in zip(old["results"],current["results"],result["rows"]):
   assert a["command"]==b["command"]==r["command"]
   assert a["metrics"]["yaw_error"]==r["original_yaw"] and b["metrics"]["yaw_error"]==r["rerun_yaw"]
   assert a["failed_trials"]==r["original_falls"] and b["failed_trials"]==r["rerun_falls"]
   assert a["minimum_base_height"]==r["original_minimum_height"] and b["minimum_base_height"]==r["rerun_minimum_height"]
   assert r["all_original_result_fields_exact"]==(all(b[k]==v for k,v in a.items() if k!="metrics") and all(x==0 for x in r["all_original_metric_differences"].values()))
   assert {k:b["metrics"][k]-v for k,v in a["metrics"].items()}==r["all_original_metric_differences"]
 conclusions=load("conclusions.json");assert len(conclusions["next_recommendations"])==1 and not conclusions["recommendation_implemented"]
 assert conclusions["source_sha256"]==sha("scripts/locomotion/conclude_issue15.py")
 for name,h in conclusions["interpretation_basis_sha256"].items():assert sha(ROOT/name)==h
 assert conclusions["gait_phase_source_sha256"]==sha("source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/tw/velocity_env_cfg.py")
 requirements=dict(complete_requested37_entry35_unique_lineage_and_three_original_FT5_protocols=True,
 frozen_zero_training_budget_and_all221_sources_preserved=True,observer_original_metrics_physical_state_and_full_RNG_neutrality=True,
 all108_jobs_and647424_initial_trials_complete=True,all630_command_moments_masks_yaw_and_heading_independently_reconstructed=True,
 all_three_forward_yaw_bias_temporal_SD_heading_drift_and_spectral_gait_relation_reported=True,
 training_seed_bootstrap_single_seed_no_CI_shared_ancestry_and_censoring_caveats=True,
 fixed20percent_rule_and_all_paired_or_conditional_intervals_recomputed=True,
 actual_parent_graph_config_differences_and_no_forced_unique_causal_claim=True,
 all225_previous_clean_records_compared_and_reproduction_or_differences_explicitly_reported=True,
 all_three_eval_seeds_original_FT5_protocol_formula_and_same_checkpoint_comparison=True,
 two_true_unfiltered_median_yaw_trials_and_exact_state_reconstructions=True,
 four_colored_front_side20s_clips_world_heading_inset_and28_frame_automatic_QA=True,
 six_figures_two_sheets_eight_extreme_motion_heading_frames_actual_manual_review=True,
 exactly_one_next_recommendation_not_implemented_and_no_default_promotion=True)
 result={"passed":True,"requirements":requirements,"formal_training_updates":0,"formal_evaluation_jobs":108,"formal_initial_trials":total,"unique_command_audits":checks,"independent_forward_spectral_contact_reference_trials":len(seen),"supplementary_full_observable_band_and_original_planned_band_both_checked":True}
 (ROOT/"delivery_integrity_audit.json").write_text(json.dumps(result,indent=2)+"\n");print("ISSUE15_DELIVERY_AUDIT_PASS",flush=True)
if __name__=="__main__":main()
