"""Requirement-level issue13 audit after actual manual media review, before commit."""
import csv,gzip,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from issue13_common import ROOT,REPO,BASE,CLEAN,FORWARD,sha
from issue13_statistics import compare,classify
from issue13_tracking import SCOPES
from run_issue13 import verify_frozen
def load(name):return json.loads((ROOT/name).read_text())
def main():
 for n in ["runtime_smoke_complete","formal_evaluation_complete","analysis_complete","plots_complete","reconstruction_complete","videos_complete","automated_postprocessing_complete"]:assert (ROOT/n).exists(),n
 verify_frozen()
 assert "Ran 10 tests" in (ROOT/"unit_tests.log").read_text() and "\nOK\n" in (ROOT/"unit_tests.log").read_text()
 assert set(load("runtime_smoke_audit.json"))=={"clean","combined"} and all(v["passed"] for v in load("runtime_smoke_audit.json").values())
 assert load("runtime_strict_byte_equivalence_audit.json")["passed"]
 models=load("input_models.json")
 assert models==json.loads((REPO/"validation_artifacts/stage1_issue12_20261110/final_models.json").read_text())
 for m in models.values():assert sha(m["checkpoint"])==m["sha256"]
 state=load("formal_job_state.json");assert len(state)==60 and sum(v["initial_trials"] for v in state.values())==245760
 assert all(v["passed"] and v["training_updates"]==0 and v["model_optimizer_unchanged"] for v in state.values())
 assert sum(v["condition"]=="clean" and v["actual_nominal_no_perturbations_verified"] for v in state.values() if v["condition"]=="clean")==30
 assert sum(v["every_original_issue12_field_and_all_trial_failure_arrays_exactly_reproduced"] for v in state.values() if v["condition"]=="combined")==30
 hashes=load("analysis_outputs_sha256.json")
 for name,h in hashes["outputs"].items():assert sha(ROOT/name)==h,name
 for file,key in [("analyze_issue13.py","analysis_source_sha256"),("issue13_analysis_data.py","independent_reader_source_sha256"),("issue13_statistics.py","statistics_source_sha256")]:assert sha(Path("scripts/locomotion")/file)==hashes[key]
 data=load("analysis_integrity_audit.json");assert data["passed"] and len(data["raw_command_audits"])==240 and data["initial_trials"]==245760
 for a in data["raw_command_audits"]:
  assert sha(a["source_moments"])==a["moments_sha256"] and sha(a["source_metadata"])==a["metadata_sha256"] and sha(a["source_eval"])==a["source_eval_sha256"]
  assert a["all_packed_windows_reset_counts_independently_reconstructed"] and a["all_sum_square_and_mixture_identities_passed"] and a["sampled_reference_env_values_reconstruct_moments"] and a["original_metrics_reconstructed"]
 csvs=load("per_trial_csv_manifest.json");assert len(csvs)==8 and sum(x["rows"] for x in csvs.values())==430080
 for name,e in csvs.items():
  b=(ROOT/name).read_bytes();assert hashlib.sha256(b).hexdigest()==e["gzip_sha256"];assert hashlib.sha256(gzip.decompress(b)).hexdigest()==e["decompressed_sha256"]
 comparisons=load("paired_comparisons.json")
 for key,c in comparisons.items():
  repeated=compare(c["control"]["per_seed"],c["treatment"]["per_seed"])
  assert repeated["control"]==c["control"] and repeated["treatment"]==c["treatment"] and repeated["paired_difference"]==c["paired_difference"]
  assert c["multiple_comparisons_corrected"] is False
 primary=load("primary_gate_comparisons.json");expected={f"clean:{cmd}:{SCOPES[0]}:all:{metric}" for cmd in CLEAN for metric in ["xy_error","yaw_error"]}|{f"combined:{cmd}:{SCOPES[0]}:outside:xy_error" for cmd in FORWARD}
 assert set(primary)==expected
 fresh={key:compare(c["control"]["per_seed"],c["treatment"]["per_seed"]) for key,c in primary.items()}
 for key in fresh:assert fresh[key]==primary[key]
 actual=classify(fresh);decision=load("decision.json")
 for key,value in actual.items():assert decision[key]==value
 assert not decision["only_next_step_recommendation"]["implemented"] and isinstance(decision["only_next_step_recommendation"]["one_recommendation"],str)
 assert not decision["default_promoted"] and decision["training_updates"]==0 and decision["issue12_rejection_unchanged"]
 selected=load("representative_selection.json");assert sha(ROOT/"representative_selection.json")==load("representative_selection_sha256.json")["sha256"]
 for group in ["control","treatment"]:
  rows=[]
  for name,m in models.items():
   if m["group"]!=group:continue
   for es in [1,2,3]:
    with np.load(ROOT/"combined/moments"/f"{name}_eval{es}"/"forward_08.npz") as z:
     means=z["sums"][0,0,:,0]/z["counts"][0,0]
     rows.extend((float(v),m["training_seed"],es,i,name) for i,v in enumerate(means))
  assert len(rows)==15360;median=float(np.median([r[0] for r in rows]));pick=min(rows,key=lambda r:(abs(r[0]-median),r[1],r[2],r[3]));s=selected[group]
  assert (s["xy_error"],s["training_seed"],s["eval_seed"],s["env_id"],s["model"])==pick and s["group_median_xy"]==median and not s["survival_filter_used"]
 reconstruction=load("reconstruction_state.json");assert len(reconstruction)==2
 for group,s in reconstruction.items():
  assert s["all_original_metrics_raw_records_and_moments_exactly_reproduced"] and s["formal_denominators_unchanged"]
  assert sha(s["trajectory"])==s["trajectory_sha256"] and s["env_id"]==selected[group]["env_id"]
 video=load("video_quality_audit.json");assert video["passed"] and len(video["videos"])==4 and video["sampled_frames"]==24
 manual=load("video_physics_manual_audit.json")
 assert manual["passed"] and manual["reviewed_contact_sheets"]==2 and manual["reviewed_plot_count"]==6 and len(manual["full_resolution_motion_frames_reviewed"])==4
 assert manual["front_and_side_colors_fullbody_and_attachment_samples_pass"] and manual["humanml_source_npy"]=="N/A: Stage1 velocity commands"
 for name,h in manual["reviewed_frame_and_sheet_sha256"].items():assert sha(ROOT/name)==h,name
 assert len(manual["reviewed_plot_sha256"])==6
 for name,h in manual["reviewed_plot_sha256"].items():assert sha(ROOT/name)==h,name
 assert len(list((ROOT/"figures").glob("*.png")))==6 and len(list((ROOT/"figures").glob("*.svg")))==6
 for entry in load("video_plan.json"):
  group=entry["group"]
  assert entry["push_seconds"]==reconstruction[group]["push_seconds"]
  assert entry["views"]["front"]["source_trajectory_sha256"]==entry["views"]["side"]["source_trajectory_sha256"]==reconstruction[group]["trajectory_sha256"]
  for view in ["front","side"]:
   e=entry["views"][view];assert sha(e["path"])==e["sha256"]
   meta=json.loads(Path(e["metadata"]).read_text());assert not meta["physics_integrated"] and meta["push_seconds"]==entry["push_seconds"]
 for p in ["scripts/locomotion/train.py","scripts/locomotion/eval_cmd.py"]:assert Path(p).read_bytes()==subprocess.check_output(["git","show",BASE+":"+p])
 assert not subprocess.check_output(["git","diff","--name-only",BASE,"--","source/unitree_rl_lab"]).strip()
 result={"passed":True,"decision":decision["decision"],"requirements":{"analysis_only_existing10models_unchanged_no_training_or_config_promotion":True,
  "clean1024x3seedsx5commands_and_combined1024x3seedsx3speeds_all10models":True,
  "nominal_clean_physics_zero_delay_no_noise_no_push_verified":True,
  "all240commands245760trials_full_moments_windows_counts_and_original_reproduction_audited":True,
  "per_trial_bias_temporal_fluctuation_and_all_seed_paired95_intervals_no_multiple_correction":True,
  "frozen13comparison_A_B_rule_prespecified_boundary_and_initial_episode_sensitivities":True,
  "exactly_one_future_recommendation_no_implementation_or_issue12_acceptance_change":True,
  "two_true_median_representatives_four_colored_exact_state_front_side_videos_actual_push_times":True,
  "actual_manual_plot_and_motion_attachment_review_and_encoding24sample_audit":True},
  "raw_moment_commands_bound_by_sha256":240,"primary_per_trial_csv_rows":430080,"training_updates":0,"default_promoted":False}
 (ROOT/"delivery_integrity_audit.json").write_text(json.dumps(result,indent=2));print("ISSUE13_DELIVERY_AUDIT_PASS",decision["decision"],flush=True)
if __name__=="__main__":main()
