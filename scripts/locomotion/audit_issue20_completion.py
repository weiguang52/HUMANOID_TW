"""Requirement-by-requirement scientific delivery audit; publication is verified separately."""
import json,subprocess,math
from pathlib import Path
from issue20_common import ROOT,REPO,sha,save
BASE='2c89ead59fa17fa9e339a8e46b972a8af579d237'
def read(name):return json.loads((ROOT/name).read_text())
def require(condition,message):assert condition,message

def main():
 require((ROOT/'formal_evaluation_complete').exists(),'Formal evaluation incomplete');records=read('evaluation_records.json');stop=(ROOT/'prerequisite_stop.json').exists();require(len(records)==(30 if stop else 60),'Wrong evaluation scope');cohort=read('cohort.json');require(len(cohort)==5 and {x['training_seed'] for x in cohort}==set(range(1,6)),'Five final policies required')
 for m in cohort:require(m['group']=='sym_wrap' and sha(m['checkpoint'])==m['sha256'],'Exact issue19 final checkpoint changed')
 phase=read('actor_phase_validation.json');require(phase['passed'] and len(phase['results'])==5,'Phase tests incomplete')
 for row in phase['results']:require(row['states']==102400 and row['algebraic_phase_max_error']<=1e-5 and row['actual_idx_float32_phase_max_error']<=1e-5,'Phase equivariance error')
 zero=read('zero_phase_reproduction.json');require(zero['passed'] and zero['num_envs']==1024 and len(zero['checks'])==4,'Zero offset validation incomplete')
 for row in zero['checks']:require(row['steps']==1000 and row['action_and_root_joint_state_bitwise_equal'] and row['all_archived_issue19_yaw_arrays_bitwise_equal'],'Zero-offset replay mismatch')
 for name in ['strict_usd_preflight.json','runtime_strict_model_gate.json','physical_comparison_audit.json','left_authority_audit.json','new_model_mirror_sign_contract.json','original_usd_bundle.json','existing_reference_bindings.json']:require(read(name)['passed'],name)
 require(read('physical_comparison_audit.json')['all_defaults_scales_PD_unchanged'],'Default/PD changes');require(read('new_model_mirror_sign_contract.json')['all30_original_issue16_signed_joint_coordinates_preserved'],'Joint coordinate convention changed');standing=read('no_policy_standing.json');require(standing['passed'] and standing['seconds']==20 and standing['num_envs']==1024,'No-policy standing validation incomplete')
 correction=read('post_evaluation_analysis_correction.json') if (ROOT/'post_evaluation_analysis_correction.json').exists() else None
 for manifest in ['frozen_evaluation_sources.json','frozen_presentation_sources.json','frozen_verification_sources.json']:
  for path,digest in read(manifest)['files'].items():
   actual=sha(path)
   if actual!=digest:
    require(correction is not None and correction['passed'],'Undocumented frozen-source change')
    require(path==str(REPO/'scripts/locomotion/issue20_analysis.py'),'Runtime/presentation source cannot change')
    item=correction['changed_sources'][path]
    require(item['before_sha256']==digest and item['after_sha256']==actual,'Incorrect correction lineage')
    require(correction['all_formal_raw_inputs_unchanged'] and correction['all_gate_metrics_bitwise_unchanged'],'Correction changed experiment or decision')

 require(sha(ROOT/'classification_rules.json')=='482b0eace32724500e35a7e5ff126fae6304bbceeb96ae099fe02743d34f2fac','Prospective rules changed')
 expected={(u,ts,es,k) for u in (['symmetric'] if stop else ['symmetric','original']) for ts in range(1,6) for es in [1,2,3] for k in [0,10]};require({(x['usd'],x['training_seed'],x['eval_seed'],x['phase']) for x in records}==expected,'Missing/duplicate factorial cells')
 for row in records:
  require(sha(row['output'])==row['output_sha256'],'Changed formal job');d=json.loads(Path(row['output']).read_text());require(d['deterministic_policy'] and d['paired_resets'] and d['issue5_condition']=='clean' and not d['pushes'] and not d['observation_noise'],'Protocol changed');require(d['issue19_actor']=='sym_wrap' and d['issue18_intervention']['mode']=='original','Wrong/double actor wrapper');require(math.prod(d['observation_dims']['policy'])==101,'Observation contract changed');require(d['issue18_tracking']['model_optimizer_state_unchanged'],'Model/optimizer changed');require([x['command'] for x in d['results']]==['forward_03','forward_05','forward_08','stand'],'Command scope changed');require(all(x['num_envs']==1024 and x['seconds']==20 for x in d['results']),'Environment/horizon changed')
 pairs=read('incremental_completed_pair_audit.json');require(pairs['passed'] and pairs['completed_jobs_at_snapshot']==len(records) and pairs['phase_pair_command_cells']==len(records)*2,'Full completed-pair audit missing');stats=read('independent_statistical_audit.json');require(stats['passed'] and stats['formal_jobs']==len(records) and stats['command_cells']==len(records)*4 and stats['paired_command_cells']==len(records)*2,'Independent all-cell statistical audit missing')
 sym=read('symmetric_analysis.json');conclusion=read('scientific_conclusion.json');require(conclusion['training_updates']==0 and not conclusion['default_promoted'] and not conclusion['parameter_scan'],'Forbidden training/default/scan');require(stop != sym['prerequisite_passed'],'Prerequisite stopping rule violated')
 if stop:require(read('prerequisite_stop.json')['skipped_original_USD_jobs']==30 and not any(x['usd']=='original' for x in records),'Early stop did not preserve scope')
 else:require(conclusion['classification'] in ['Phi','Gamma','Omega'],'Invalid frozen outcome')
 require(read('colored_asset_integrity.json')['passed'],'Colored physics identity unproven');qa=read('video_quality_audit.json');require(qa['passed'] and qa['single_view_clips']==6 and qa['paired_clips']==1 and qa['single_view_sampled_frames']==42,'Paired/front/side media incomplete');require(read('paired_overhead_video.json')['passed'],'Paired overhead missing')
 for example in read('video_plan.json'):
  require(set(example['views'])=={'front','side','overhead'},'Missing view')
  for v in example['views'].values():require(sha(v['path'])==v['sha256'],'Video changed')
 chosen=read('representative_selection.json');finite=[(abs(x),i+1) for i,x in enumerate(chosen['all_seed_D_degrees']) if x is not None];winner=max(finite,key=lambda x:(x[0],-x[1]))[1];require(chosen['training_seed']==winner and chosen['env_id']==0 and chosen['eval_seed']==1,'Video selection deviated')
 manual=read('manual_visual_review.json');require(manual['actual_review_completed'] and manual['figures_reviewed']==4 and manual['contact_sheets_reviewed']==2 and manual['full_resolution_frames_reviewed']>=18 and manual['paired_overhead_reviewed'],'Actual visual review incomplete')
 for path,digest in manual['reviewed_files_sha256'].items():require(sha(ROOT/path)==digest,'Reviewed media changed')
 for name in ['results.md','seed_tables.md','cell_seed_statistics.csv','paired_seed_statistics.csv','usd_physical_comparison.csv','usd_physical_comparison.md']:require((ROOT/name).is_file() and (ROOT/name).stat().st_size>0,'Missing report/table')
 require(not subprocess.check_output(['git','diff','--name-only',BASE,'--','source','scripts/locomotion/train.py','scripts/locomotion/eval_cmd.py']),'Existing defaults/rewards/train/evaluator modified')
 groups={'same_five_final_policies_and_102400_state_phase_equivalence':['cohort.json','actor_phase_validation.json','existing_reference_bindings.json'],'zero_offset_full_action_state_bitwise_reproduction':['zero_phase_reproduction.json'],'left_authoritative_USD_and_complete_parameter_difference_table':['symmetric_usd_construction.json','strict_usd_preflight.json','physical_comparison_audit.json','left_authority_audit.json'],'actual_PhysX_parameters_and_original_joint_sign_contract':['runtime_symmetry_audit.json','runtime_strict_model_gate.json','new_model_mirror_sign_contract.json','model_numerical_symmetry_limits.json'],'no_policy_standing20s':['no_policy_standing.json'],'all_factorial_cells_or_required_precheck_stop':['evaluation_records.json','evaluation_progress.json'],'all_initial_pairs_noise_phase_feedback_and_saved_pose_consistency':['incremental_completed_pair_audit.json'],'all_cell_metrics_S_D_and_exact_five_seed_intervals_independent_recomputation':['independent_statistical_audit.json'],'prospective_rules_classification_and_no_training_or_default_promotion':['classification_rules.json','scientific_conclusion.json'],'maximum_D_seed_exact_state_colored_paired_and_front_side_views':['representative_selection.json','colored_asset_integrity.json','paired_overhead_video.json','video_quality_audit.json'],'actual_figures_full_resolution_and_attachment_review':['manual_visual_review.json','figure_manifest.json'],'complete_report_and_seed_tables':['results.md','seed_tables.md','cell_seed_statistics.csv','paired_seed_statistics.csv'],'original_inputs_complete_and_unused_weight_cleanup_authorized':['original_usd_bundle.json','unused_weight_cleanup_receipt.json']}
 if correction is not None:groups['all_cell_metrics_S_D_and_exact_five_seed_intervals_independent_recomputation'].append('post_evaluation_analysis_correction.json')
 evidence={k:{'passed':True,'files':{str(ROOT/n):sha(ROOT/n) for n in names}} for k,names in groups.items()};save('prepublication_requirement_audit.json',{'passed':True,'requirements':evidence,'formal_jobs':len(records),'initial_trials':4096*len(records),'training_updates':0,'prerequisite_passed':sym['prerequisite_passed'],'classification':conclusion['classification'],'publication_separate':True});print('ISSUE20_PREPUBLICATION_REQUIREMENT_AUDIT_PASS',len(evidence),len(records),flush=True)
if __name__=='__main__':main()
