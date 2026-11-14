"""All requested scientific, presentation and source requirements before Git publication."""
import json,subprocess
from pathlib import Path
from issue21_common import ROOT,REPO,sha,save

def read(n):return json.loads((ROOT/n).read_text())
def main():
 assert (ROOT/'formal_evaluation_complete').exists();records=read('evaluation_records.json');assert len(records)==25
 assert read('independent_open_continuity_audit.json')['passed'];assert read('independent_protocol_audit.json')['passed'];assert read('independent_component_crossing_audit.json')['passed'];assert read('existing_issue20_input_objects.json')['passed'];assert read('existing_reference_bindings.json')['passed']
 for manifest in ['frozen_evaluation_sources.json','frozen_analysis_presentation_sources.json']:
  for p,d in read(manifest)['files'].items():assert sha(p)==d,p
 assert sha(ROOT/'classification_rules.json')=='5d017196e31b317135bbceda1fa5ebd46f7ec82b662030432c434044987c4192'
 conclusion=read('scientific_conclusion.json');assert conclusion['classification'] in ['O','N','H','other'] and conclusion['checks']['complete'];assert conclusion['training_updates']==0 and not conclusion['default_promoted'] and not conclusion['recommendation_implemented'] and not conclusion['parameter_scan']
 assert len(read('pair_analysis.json'))==60 and len(read('perturbation_analysis.json'))==30
 for row in read('pair_analysis.json'):
  assert max(row['initial_mirror_max'].values())<=1e-6
  if row['mode']=='air':assert row['max_contact_force']==0
 assert read('colored_asset_integrity.json')['passed'];qa=read('video_quality_audit.json');assert qa['passed'] and qa['total_clips']==8;assert len(read('video_plan.json'))==4;assert len(read('figure_manifest.json'))==11
 for row in read('video_plan.json'):
  assert set(row['views'])=={'front','side'}
  for v in row['views'].values():assert sha(v['path'])==v['sha256']
 manual=read('manual_visual_review.json');assert manual['actual_review_completed'] and manual['figures_reviewed']==11 and manual['contact_sheets_reviewed']==4 and manual['full_resolution_frames_reviewed']>=24
 for p,d in manual['reviewed_files_sha256'].items():assert sha(ROOT/p)==d,p
 for name in ['results.md','first_crossings.csv','first_component_crossings.csv','O_trigger_components.json','installed_torque_API_evidence.json','torque_instrumentation_correction.json','startup_correction.json','queue_orchestration_correction/ledger.json']:
  assert (ROOT/name).is_file() and (ROOT/name).stat().st_size>0,name
 tests=read('test_validation.json');assert tests['analysis_tests_passed']==3 and tests['codec_tests_passed']==4;syntax=read('source_syntax_validation.json');assert syntax['passed']
 for p,d in syntax['files'].items():assert sha(p)==d,p
 assert not subprocess.check_output(['git','diff','--name-only','b8ddc926','--','source','scripts/locomotion/eval_cmd.py','scripts/locomotion/train.py'])
 groups={'same_five_unchanged_final_policies_and_strict_USD':['cohort.json','existing_issue20_input_objects.json','existing_reference_bindings.json'],'all_closed_loop_commands_counts_phases_1000steps':['evaluation_records.json','independent_protocol_audit.json'],'every_observation_action_root_joint_and_contact_component':['pair_analysis.json','first_crossings.csv','first_component_crossings.csv'],'independent_component_crossings_and_step_alignment':['independent_component_crossing_audit.json'],'air_fixed_no_contact_and_ground200steps_exact_open_actions':['independent_protocol_audit.json','independent_open_continuity_audit.json','ground_open_lifecycle_correction.json'],'six_yaw_perturbations_five_seeds1024env_and_censoring':['perturbation_analysis.json'],'frozen_O_N_H_other_and_one_unimplemented_recommendation':['classification_rules.json','scientific_conclusion.json','O_trigger_components.json'],'measured_active_joint_force_and_separate_PD_estimate_semantics':['installed_torque_API_evidence.json','torque_instrumentation_correction.json'],'all_sources_models_defaults_unchanged_and_no_training':['frozen_evaluation_sources.json','independent_protocol_audit.json'],'eight_exact_pose_colored_front_side_videos':['colored_asset_integrity.json','video_plan.json','video_quality_audit.json'],'actual_eleven_plots_four_sheets_twentyfour_frames':['manual_visual_review.json','figure_manifest.json'],'complete_evidence_and_historical_failures_preserved':['results.md','torque_instrumentation_correction.json','startup_correction.json'],'meaningful_tests_and_syntax_validation':['test_validation.json','source_syntax_validation.json']}
 save('prepublication_requirement_audit.json',{'passed':True,'groups':{k:{'passed':True,'evidence':{n:sha(ROOT/n) for n in names}} for k,names in groups.items()},'jobs':25,'trace_pairs':60,'perturbation_cells':30,'classification':conclusion['classification'],'training_updates':0,'publication_separate':True});print('ISSUE21_REQUIREMENT_AUDIT_PASS',len(groups),flush=True)
if __name__=='__main__':main()
