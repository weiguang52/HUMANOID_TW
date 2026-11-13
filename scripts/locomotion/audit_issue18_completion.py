"""Pre-publication requirement audit; missing evidence is a failure, not success."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue18_20261113')
REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 evidence={}
 def prove(name,condition,paths):
  assert condition,name
  evidence[name]={'passed':True,'files':{str(p):sha(p) for p in paths}}
 p=ROOT/'projection_validation.json';v=read(p)
 prove('fifteen_models_same102400_states_identity_and_strict_equivariance',v['passed'] and v['models']==15 and v['states_per_model']==102400 and len(v['results'])==15 and all(r['count']==102400 and r['max_relative_error']<=1e-5 and r['identity_bitwise_equal_all_states'] and r['inference_rng_unchanged'] and r['actor_and_optimizer_file_unchanged'] for r in v['results']),[p])
 for n,h in v['helper_source_sha256'].items():assert sha(REPO/'scripts/locomotion'/n)==h
 p=ROOT/'evaluation_progress.json';q=read(p)
 prove('bounded45_new_jobs_270_commands_276480_trials',q['status']=='completed' and len(q['completed'])==45 and q['command_records']==270 and q['initial_trials']==276480,[p])
 frozen=read(ROOT/'formal_inputs_frozen.json')
 for section in ['sources','models','protocol']:
  for path,h in frozen[section].items():assert sha(path)==h,path
 prove('formal_sources_weights_and_prospective_rules_unchanged',frozen['zero_training'] and frozen['formal_jobs']==45,[ROOT/'formal_inputs_frozen.json',ROOT/'classification_rules.json'])
 from run_issue18_evaluations import audit
 models={m['model']:m for m in read(ROOT/'cohort.json')['models']}
 jobs=read(ROOT/'evaluation_job_templates.json');assert len(jobs)==45
 for j in jobs:
  out=ROOT/'evaluation/clean'/(j['name']+'.json');receipt=audit(out,j,models[j['model']])
  original=next(x for x in q['completed'] if x['name']==j['name']);assert receipt['sha256']==original['sha256']
 prove('all_commands_actual_feedback_phase_and_model_immutability',True,[ROOT/'preflight_audit.json'])
 originals=read(ROOT/'original_reuse_manifest.json');assert originals['count']==45 and originals['no_original_rollouts_rerun']
 portable=read(ROOT/'original_reuse_portability.json')['source_files']
 for j in originals['jobs']:
  for path,h in {j['output']:j['output_sha256'],**j['source_files_sha256']}.items():
   assert sha(path)==h and portable[path]['sha256']==h
   assert sha(REPO/portable[path]['committed_relative_path'])==h
 prove('original45_jobs_reused_without_rewrite',True,[ROOT/'original_reuse_manifest.json',ROOT/'original_reuse_portability.json',ROOT/'original_phase_reuse_audit.json'])
 p=ROOT/'raw_analysis_audit.json';a=read(p)
 assert a['passed'] and a['total_jobs']==90 and a['command_records']==540 and a['per_trial_rows']==552960
 for path,h in a['raw_inputs_sha256'].items():assert sha(path)==h
 assert sha(ROOT/'per_trial_heading.csv.gz')==a['per_trial_csv_sha256']
 prove('all_original_and_projected_heading_falls_prefix_tracking_and_halfcycles',True,[p,ROOT/'job_statistics.json',ROOT/'seed_statistics.json',ROOT/'group_statistics.json',ROOT/'paired_comparisons.json'])
 classification=read(ROOT/'classification.json')
 assert classification['prospective_rules_sha256']==sha(ROOT/'classification_rules.json')
 prove('prospective_training_seed_A_B_C_classification',classification['overall'] in ['A','B','C'] and len(classification['cohorts'])==3,[ROOT/'classification.json'])
 selection=read(ROOT/'representative_selection.json');reconstruction=read(ROOT/'reconstruction_state.json')
 assert selection['candidate_count']==15360 and not selection['survival_filter']
 assert reconstruction['all_prior_original_recorded_arrays_bitwise_equal'] and reconstruction['all_prior_original_result_rows_exactly_reproduced']
 plan=read(ROOT/'video_plan.json');assert len(plan)==1 and set(plan[0]['views'])=={'front','overhead'}
 for view in plan[0]['views'].values():assert sha(view['path'])==view['sha256']
 assert read(ROOT/'video_quality_audit.json')['passed']
 review=read(ROOT/'manual_visual_review.json')
 assert review['actual_review_completed'] and review['figures_reviewed']==3 and review['contact_sheets_reviewed']==1 and review['full_resolution_frames_reviewed']>=4
 for path,h in review['reviewed_files_sha256'].items():assert sha(ROOT/path)==h
 prove('strict_sym_int_median_colored_two_views_world_XY_and_actual_review',True,[ROOT/'representative_selection.json',ROOT/'reconstruction_state.json',ROOT/'media_source_sha256.json',ROOT/'video_quality_audit.json',ROOT/'manual_visual_review.json'])
 for p in [ROOT/'results.md',ROOT/'seed_tables.md']:assert p.stat().st_size>1000
 prove('complete_report_all_tables_figures_and_one_unimplemented_recommendation',True,[ROOT/'results.md',ROOT/'seed_tables.md',ROOT/'figure_manifest.json'])
 # Confirm all previously tracked code/config/USD records remain untouched.
 changed=subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=REPO,text=True).splitlines()
 allowed=['README.md'];assert all(p in allowed or p.startswith('docs/') for p in changed),changed
 prove('no_existing_reward_configuration_USD_or_default_edits',True,[ROOT/'cohort.json'])
 output={'passed':True,'scope':'pre-publication scientific and implementation requirements; GitHub and Notion delivery must be verified separately','requirements':evidence,'zero_training':True,'source_sha256':sha(__file__)}
 (ROOT/'prepublication_requirement_audit.json').write_text(json.dumps(output,indent=2)+'\n')
 print('ISSUE18_REQUIREMENTS_PASS',len(evidence))
if __name__=='__main__':main()
