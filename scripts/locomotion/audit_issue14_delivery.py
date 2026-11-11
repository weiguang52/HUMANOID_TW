"""Requirement-level scientific and sampled-media audit before issue14 publication."""
import csv,gzip,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from issue14_common import ROOT,REPO,sha
from run_issue14 import verify_frozen,audit_training,audit_eval,CLEAN,FORWARD
from issue13_statistics import compare,summary
from issue13_tracking import SCOPES
OLD=Path("/root/gpufree-data/tw_stage1/issue13_20261110")
def load(n):return json.loads((ROOT/n).read_text())
def main():
 for n in ["formal_gates_complete","training_complete","formal_evaluation_complete","analysis_complete","plots_complete","reconstruction_complete","videos_complete","automated_postprocessing_complete"]:assert (ROOT/n).exists(),n
 verify_frozen()
 assert "Ran 10 tests" in (ROOT/"unit_tests.log").read_text() and "\nOK\n" in (ROOT/"unit_tests.log").read_text()
 assert all(load(n)["passed"] for n in ["runtime_smoke_audit.json","eval_adapter_smoke_audit.json","default_off_source_audit.json","historical_source_equivalence_audit.json","first_formal_start_audit.json"])
 parents=load("input_parents.json");new=load("final_models.json");historical=load("historical_models.json");assert len(new)==5 and len(historical)==10
 fresh={}
 for name,m in new.items():
  seed=m["training_seed"];assert m["additional_updates"]==1500 and m["parent_sha256"]==parents[f"requested_s{seed}_u4500"]["sha256"]
  fresh[name]=audit_training(seed,m,parents[f"requested_s{seed}_u4500"]);assert fresh[name]["passed"]
 jobs=load("formal_job_state.json");assert len(jobs)==30 and sum(v["initial_trials"] for v in jobs.values())==122880
 for cond in ["clean","combined"]:
  for name,m in new.items():
   for es in [1,2,3]:assert audit_eval(ROOT/cond/f"{name}_eval{es}.json",cond,m)["passed"]
 bound=load("analysis_outputs_sha256.json")
 for name,h in bound["outputs"].items():assert sha(ROOT/name)==h,name
 for name,h in bound["source_sha256"].items():assert sha(name)==h,name
 figures=load("figure_outputs_sha256.json");assert len(figures["figures_sha256"])==12
 for name,h in figures["source_sha256"].items():assert sha(name)==h
 for name,h in {**figures["input_sha256"],**figures["figures_sha256"]}.items():assert sha(ROOT/name)==h
 media=load("media_source_sha256.json")
 for name,h in {**media["source_sha256"],**media["asset_sha256"]}.items():assert sha(name)==h
 assert sha(ROOT/"video_plan.json")==media["video_plan_sha256"]
 data=load("analysis_integrity_audit.json");assert data["passed"] and data["commands"]==360 and data["all_initial_trials"]==368640 and data["new_initial_trials"]==122880
 assert data["per_trial_csv_rows"]==645120 and len(data["raw_command_audits"])==360 and data["historical_policy_metric_rows_recomputed_exactly"]
 oldmanifest=json.loads((REPO/"validation_artifacts/stage1_issue13_20261110/sha256.json").read_text())
 for row in data["raw_command_audits"]:
  assert row["original_metrics_packed_masks_counts_moments_reset_and_reference_audited"]
  for p,h in row["source_files_sha256"].items():
   assert sha(p)==h,p
   if row["group"]!="clock":
    rel=str(Path(p).relative_to(OLD));assert h==oldmanifest[rel]==sha(REPO/"validation_artifacts/stage1_issue13_20261110"/rel)
 for n,e in load("per_trial_csv_manifest.json").items():
  raw=(ROOT/n).read_bytes();assert hashlib.sha256(raw).hexdigest()==e["gzip_sha256"] and hashlib.sha256(gzip.decompress(raw)).hexdigest()==e["decompressed_sha256"]
 comp=load("paired_comparisons.json")
 for key,d in comp.items():
  repeated=compare(d["control"]["per_seed"],d["treatment"]["per_seed"])
  for field in ["control","treatment","paired_difference"]:assert repeated[field]==d[field],(key,field)
  assert d["multiple_comparisons_corrected"] is False
 def get(pair,cond,cmd,part,metric):return comp[":".join([pair,cond,cmd,SCOPES[0],part,metric])]
 falls=load("fall_results.json")
 for row in falls.values():
  for g,v in row["groups"].items():assert summary(np.array(v["per_seed_falls"])/3072)==v["seed_summary"] and sum(v["per_seed_falls"])==v["pooled_falls"]
 primary=get("clock_vs_treatment","clean","forward_08","all","xy_error")
 vx=get("clock_vs_control","clean","forward_08","all","vx_mean")
 d=load("decision.json");g=d["gates"]
 expected={
 "primary_clean08_xy_vs_treatment_upper95_below_zero":primary["paired_difference"]["bootstrap95"][1]<0,
 "clean05_xy_vs_control_mean_degradation_at_most5percent":get("clock_vs_control","clean","forward_05","all","xy_error")["treatment"]["mean"]<=1.05*get("clock_vs_control","clean","forward_05","all","xy_error")["control"]["mean"],
 "clean08_xy_vs_control_mean_degradation_at_most5percent":get("clock_vs_control","clean","forward_08","all","xy_error")["treatment"]["mean"]<=1.05*get("clock_vs_control","clean","forward_08","all","xy_error")["control"]["mean"],
 "clean08_mean_vx_vs_control95_interval_not_entirely_negative":vx["paired_difference"]["bootstrap95"][1]>=0,
 "combined03_clock_seed_level_fall95_upper_below1percent":falls["combined:forward_03"]["groups"]["clock"]["seed_summary"]["bootstrap95"][1]<.01,
 "combined05_clock_mean_falls_not_above_control":falls["combined:forward_05"]["groups"]["clock"]["seed_summary"]["mean"]<=falls["combined:forward_05"]["groups"]["control"]["seed_summary"]["mean"]}
 snapshot=load("primary_clean_complete_snapshot.json");assert snapshot["passed"] and snapshot["primary_clean_dataset_complete"]
 for field in ["control","treatment","paired_difference"]:assert snapshot["primary"][field]==primary[field],field
 assert snapshot["primary_mitigation_supported"]==expected["primary_clean08_xy_vs_treatment_upper95_below_zero"]
 for name,h in snapshot["source_sha256"].items():assert sha(name)==h
 assert g==expected and d["primary_mitigation_supported"]==expected["primary_clean08_xy_vs_treatment_upper95_below_zero"] and d["all_gates_pass"]==all(expected.values())
 assert d["stop_clock_hypothesis"]==(not d["primary_mitigation_supported"]) and not d["default_promoted"] and d["total_formal_updates"]==7500
 for cmd in FORWARD:
  bins=load("actual_speed_bins_"+cmd+".json");assert bins["fixed_bin_width_m_s"]==.05 and bins["post_treatment_conditioning"] and not bins["causal_recovery_effect_identified"]
  assert len(bins["bins"])==22
  for group in ["control","treatment","clock"]:
   assert sum(b["groups"][group]["n"] for b in bins["bins"])==15360
   assert sum(b["groups"][group]["falls"] for b in bins["bins"])==falls["combined:"+cmd]["groups"][group]["pooled_falls"]
 context=load("actual_speed_fall_interpretation.json");assert {x["command"] for x in context}==set(FORWARD)
 for x in context:
  assert not x["post_treatment_speed_bins_identify_causal_recovery"]
  assert all(0<=v<=1 for v in x["fraction_trials_in_common_bins"].values())
  for pair,direction in x["comparisons"].items():
   current=get(pair,"combined",x["command"],"all","vx_mean")["paired_difference"]
   assert direction["actual_mean_vx_difference"]==current
   expected="slower" if current["bootstrap95"][1]<0 else ("faster" if current["bootstrap95"][0]>0 else "direction_unresolved_not_equivalence")
   assert direction["speed_direction"]==expected
 assert sum(1 for _ in csv.DictReader((ROOT/"actual_speed_trial_rows.csv").open()))==138240
 models={**historical,**new};selected=load("representative_selection.json");assert len(selected)==3 and sha(ROOT/"representative_selection.json")==load("representative_selection_sha256.json")["sha256"]
 for group,s in selected.items():
  rows=[]
  for name,m in models.items():
   if m["group"]!=group:continue
   origin=ROOT if group=="clock" else OLD
   for es in [1,2,3]:
    with np.load(origin/"combined/moments"/f"{name}_eval{es}"/"forward_08.npz") as z:
     means=z["sums"][0,0,:,0]/z["counts"][0,0];rows.extend((float(v),m["training_seed"],es,i,name) for i,v in enumerate(means))
  assert len(rows)==15360;median=float(np.median([r[0] for r in rows]));pick=min(rows,key=lambda r:(abs(r[0]-median),r[1],r[2],r[3]))
  assert (s["xy_error"],s["training_seed"],s["eval_seed"],s["env_id"],s["model"])==pick and s["group_median_xy"]==median and not s["survival_filter_used"]
 reconstruction=load("reconstruction_state.json");assert len(reconstruction)==3
 for group,s in reconstruction.items():assert sha(s["trajectory"])==s["trajectory_sha256"] and s["all_original_metrics_raw_records_and_moments_exactly_reproduced"] and s["formal_denominators_unchanged"]
 plan=load("video_plan.json");assert len(plan)==3
 for entry in plan:
  group=entry["group"];source=reconstruction[group];assert entry["push_seconds"]==source["push_seconds"]
  assert entry["views"]["front"]["source_trajectory_sha256"]==entry["views"]["side"]["source_trajectory_sha256"]==source["trajectory_sha256"]
  for view,e in entry["views"].items():
   assert sha(e["path"])==e["sha256"];meta=json.loads(Path(e["metadata"]).read_text())
   assert meta["frames"]==1000 and meta["fps"]==50 and not meta["physics_integrated"] and meta["humanml_source_npy"]=="N/A: Stage1 velocity commands"
   assert meta["push_seconds"]==source["push_seconds"]
   assert "Colored USD integrity: 1876 original attributes unchanged" in (ROOT/(group+"_"+view+".log")).read_text()
   if entry["reused_issue13"]:assert e["bytes_reused_exactly"] and sha(e["original_video"])==e["sha256"]
 video=load("video_quality_audit.json");assert video["passed"] and len(video["videos"])==6 and video["sampled_frames"]==36
 manual=load("video_physics_manual_audit.json");assert manual["passed"] and manual["reviewed_contact_sheets"]==3 and manual["reviewed_plot_count"]==6
 assert len(manual["full_resolution_motion_frames_reviewed"])==6 and manual["front_and_side_colors_fullbody_and_attachment_samples_pass"] and manual["humanml_source_npy"]=="N/A: Stage1 velocity commands"
 for name,h in manual["reviewed_frame_and_sheet_sha256"].items():assert sha(ROOT/name)==h,name
 assert len(manual["reviewed_plot_sha256"])==6
 for name,h in manual["reviewed_plot_sha256"].items():assert sha(ROOT/name)==h,name
 assert len(list((ROOT/"figures").glob("*.png")))==6 and len(list((ROOT/"figures").glob("*.svg")))==6
 result={"passed":True,"primary_mitigation_supported":d["primary_mitigation_supported"],"all_gates_pass":d["all_gates_pass"],"formal_training_updates":7500,"smoke_updates":6,
 "requirements":{"only_critic_two_clock_inputs_reward_boundary_reset_policy101_hidden_layers_and_distributions_unchanged":True,
 "strict_original_five_parent_actor_critic_Adam_std_load_and_zero_new_columns_no_extra_rng":True,
 "ten_unit_tests_off_switch_bitwise_PPO_and_on1000step_dynamics_rng_regression":True,
 "five_4096env_1500update_runs_no_extra_budget_or_smoke_model_selection":True,
 "historical_control_treatment_code_config_models_and_evaluation_reused_exactly":True,
 "thirty_new_evaluation_jobs_both_original_protocols_1024x3x20s":True,
 "three_group_full_trial_falls_tracking_speed_sd_and_paired_seed95_unadjusted":True,
 "inside_outside_moments_raw_masks_counts_and_variance_identities_independently_audited":True,
 "achieved_speed_fall_table_fixed_common_bins_support_and_post_treatment_caveat":True,
 "frozen_primary_and_all_acceptance_gates_recomputed_no_relaxation_or_promotion":True,
 "all_three_true_unfiltered_median_trials_and_exact_state_source_reconstruction":True,
 "six_precolored_front_side20s_clips_actual_pushes_no_physics_reintegration":True,
 "six_plot_three_sheet_six_full_resolution_frame_actual_manual_review_and36sample_video_QA":True}}
 (ROOT/"delivery_integrity_audit.json").write_text(json.dumps(result,indent=2));print("ISSUE14_DELIVERY_AUDIT_PASS",d["primary_mitigation_supported"],d["all_gates_pass"],flush=True)
if __name__=="__main__":main()
