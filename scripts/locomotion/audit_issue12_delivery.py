"""Requirement-level delivery audit, after manual figures/video inspection."""
import csv,gzip,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from issue12_common import ROOT,REPO,BASE,COMMANDS,sha
from run_issue12 import audit_one,audit_pair,verify_frozen

def main():
    for name in ["training_complete","evaluation_complete","analysis_complete","plots_complete","videos_complete"]:
        assert (ROOT/name).exists(),name
    verify_frozen()
    assert "Ran 10 tests" in (ROOT/"unit_tests.log").read_text() and "\nOK\n" in (ROOT/"unit_tests.log").read_text()
    assert "Ran 11 tests" in (ROOT/"unit_tests_after_precision.log").read_text() and "\nOK\n" in (ROOT/"unit_tests_after_precision.log").read_text()
    assert "Ran 8 tests" in (ROOT/"statistics_tests_final.log").read_text() and "\nOK\n" in (ROOT/"statistics_tests_final.log").read_text()
    initial=json.loads((ROOT/"frozen_training_sources_initial.json").read_text())
    latest=json.loads((ROOT/"frozen_training_sources.json").read_text())
    changed={p for p in initial["sha256"] if initial["sha256"][p]!=latest["sha256"][p]}
    assert changed=={"scripts/locomotion/issue12_monitor.py","tests/locomotion/test_issue12_recovery.py"}
    notes=json.loads((ROOT/"monitor_precision_addendum.json").read_text())
    for p in changed:
        assert notes["changed_hashes"][p]=={"before":initial["sha256"][p],"after":latest["sha256"][p]}
    assert sha(ROOT/"frozen_source_snapshots/issue12_monitor_initial.py")==initial["sha256"]["scripts/locomotion/issue12_monitor.py"]
    assert sha(ROOT/"frozen_source_snapshots/test_issue12_recovery_initial.py")==initial["sha256"]["tests/locomotion/test_issue12_recovery.py"]
    weight=json.loads((ROOT/"frozen_weight.json").read_text())
    assert weight["optimizer_updates"]==0 and weight["no_weight_scan"]
    assert abs(weight["expected_new_mean_reward_per_step"]/weight["old_mean_reward_per_control_step"]+.05)<1e-12
    old=json.loads((REPO/"validation_artifacts/stage1_issue10_20261109/input_models.json").read_text())
    parents=json.loads((ROOT/"input_models.json").read_text())
    assert parents=={k:v for k,v in old.items() if k.endswith("_u4500")}
    for m in parents.values():assert sha(m["checkpoint"])==m["sha256"]
    models=json.loads((ROOT/"final_models.json").read_text())
    assert set(models)=={f"{g}_s{s}" for g in ["control","treatment"] for s in range(1,6)}
    audits={}
    for name,m in models.items():
        assert m["additional_updates"]==1500 and m["cumulative_requested_updates"]==6000
        assert m["weight"]==(0 if m["group"]=="control" else weight["weight"])
        assert sha(m["checkpoint"])==m["sha256"]
        parent=parents[f"requested_s{m['training_seed']}_u4500"]
        audits[name]=audit_one(name,m,parent)
    pairs={str(s):audit_pair(s,models) for s in range(1,6)}
    data=json.loads((ROOT/"analysis_integrity_audit.json").read_text())
    raw=json.loads((ROOT/"raw_record_integrity_audit.json").read_text())
    assert data["passed"] and raw["passed"] and raw["all_trials"]==92160 and len(raw["command_records"])==90
    assert data["all_partition_trial_rows"]==92160 and data["all_partition_push_rows"]==raw["all_pushes"]
    manifest=json.loads((ROOT/"partition_records_manifest.json").read_text())
    for name,e in manifest.items():
        assert sha(ROOT/name)==e["gzip_sha256"]
        b=gzip.decompress((ROOT/name).read_bytes())
        assert hashlib.sha256(b).hexdigest()==e["decompressed_csv_sha256"]
    assert json.loads((ROOT/"randomization_pairing_audit.json").read_text())["exactly_match_issue11_startup_draws"]
    video=json.loads((ROOT/"video_quality_audit.json").read_text())
    assert video["passed"] and len(video["videos"])==20 and video["sampled_frames"]==100
    manual=json.loads((ROOT/"video_physics_manual_audit.json").read_text())
    assert manual["passed"] and manual["reviewed_contact_sheets"]==10 and manual["reviewed_plot_count"]==6
    assert manual["front_and_side_colors_fullbody_and_attachment_samples_pass"]
    assert manual["humanml_source_npy"]=="N/A: Stage1 velocity commands"
    reviewed=manual["full_resolution_motion_frames_reviewed"]
    assert len(reviewed)==20 and len(set(reviewed))==20
    frame_hashes={}
    expected_stems={Path(name).stem for name in video["videos"]}
    reviewed_stems=set()
    for name in reviewed:
        frame=Path(name)
        if not frame.is_absolute():frame=ROOT/frame
        assert frame.resolve().parent==(ROOT/"video_review_frames").resolve()
        assert frame.name.endswith(("_frame500.png","_frame750.png"))
        stem=frame.stem.rsplit("_frame",1)[0]
        assert stem in expected_stems
        reviewed_stems.add(stem)
        frame_hashes[str(frame.relative_to(ROOT))]=sha(frame)
    assert reviewed_stems==expected_stems
    assert len(list((ROOT/"figures").glob("*.png")))==6 and len(list((ROOT/"figures").glob("*.svg")))==6
    acceptance=json.loads((ROOT/"acceptance.json").read_text())
    assert not acceptance["default_promoted"] and not acceptance["weight_scan"]
    for p in ["source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/tw/velocity_env_cfg.py",
              "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/tw/experiments.py",
              "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/tw/action_observation.py",
              "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/tw/mass_randomization.py"]:
        assert Path(p).read_bytes()==subprocess.check_output(["git","show",BASE+":"+p])
    result={"passed":True,"requirements":{"one_opt_in_reward_fixed5percent_calibrated_weight":True,
            "11_timer_config_monitor_tests_and8_statistics_tests":True,
            "actual_obs101_134_and_network_state_unchanged":True,
            "ten_full_actor_critic_optimizer_resumes_1500_each_no_std_reset":audits,
            "five_matched_pairs_only_reward_weight_config_diff_and_initial_rng_equality":pairs,
            "30eval_jobs_90commands_92160_initial_trials_full_records":True,
            "xy_yaw_and_seed_bootstrap_primary_secondary_gates_reported":True,
            "complete_per_update_per_control_step_average_reward_monitoring":True,
            "20colored_worst_command_front_side_clips_and_manual_review":True,
            "all_behavior_configuration_defaults_unchanged_reward_opt_in":True,
            "audit_only_precision_fix_fully_versioned_and_disclosed":True},
            "manual_full_resolution_review_frame_sha256":frame_hashes,
            "acceptance_decision":acceptance["decision"],"default_promoted":False,
            "training_sources_and_weight_match_documented_versions":True}
    (ROOT/"delivery_integrity_audit.json").write_text(json.dumps(result,indent=2))
    print("ISSUE12_DELIVERY_AUDIT_PASS",acceptance["decision"],flush=True)
if __name__=="__main__":main()
