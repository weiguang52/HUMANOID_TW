"""Require direct evidence for every issue19 gate before archival/publication."""
import hashlib,json,subprocess,os,re
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/root/gpufree-data/tw_stage1/issue19_20261113')
REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
RULE_SHA='9f5fd7739785427abee1ed67adf66b103cb0df262f5380fc652bfe6c39174d33'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 evidence={}
 def prove(name,condition,paths):
  assert condition,name
  evidence[name]=dict(passed=True,files={str(p):sha(p) for p in paths})
 rules=ROOT/'classification_rules.json';assert sha(rules)==RULE_SHA
 initial=read(ROOT/'actor_pretraining_validation.json')
 prove('five_parents_same102400_states_bitwise_issue18_initial_and_single_graph_ONNX',initial['passed'] and initial['models']==5 and len(initial['results'])==5 and all(r['states']==102400 and r['max_equivariance_error']<=1e-5 and r['initial_strict_output_bitwise_equal'] and r['onnx_max_absolute_error']<=1e-5 and r['onnx_gemm_nodes']==8 and r['probability_entropy_and_gradient_checks'] and r['std']['std_parameters_after']==18 for r in initial['results']),[ROOT/'actor_pretraining_validation.json'])
 pre=read(ROOT/'preflight_validation.json');opt=read(ROOT/'preflight_optimizer_validation.json')
 prove('two_update_4096env_smoke_and_disabled_full_state_bitwise_reproduction',pre['passed'] and pre['disabled_model_optimizer_and_runtime_bitwise_equal'] and pre['sym_wrap_two_updates'] and pre['all_three_runs_4096_envs'] and opt['passed'],[ROOT/'preflight_validation.json',ROOT/'preflight_optimizer_validation.json'])
 for name in ['frozen_training_sources.json','frozen_evaluation_sources.json']:
  for p,h in read(ROOT/name)['files'].items():assert sha(p)==h,p
 prove('frozen_inputs_and_prospective_acceptance_unchanged',True,[rules,ROOT/'frozen_training_sources.json',ROOT/'frozen_evaluation_sources.json'])
 tr=read(ROOT/'training_records.json');assert len(tr)==10
 assert {(r['group'],r['training_seed']) for r in tr}=={(g,s) for g in ['sym_wrap','sym_cont'] for s in range(1,6)}
 cohort=read(ROOT/'cohort.json')
 def finite(x):
  if isinstance(x,torch.Tensor):assert torch.isfinite(x).all()
  elif isinstance(x,dict):
   for v in x.values():finite(v)
  elif isinstance(x,(list,tuple)):
   for v in x:finite(v)
 for r in tr:
  s=r['summary'];assert s['additional_updates']==1500 and s['environment_steps']==36000 and s['num_envs']==4096 and not s['std_reset']
  parent=cohort['parents'][str(r['training_seed'])];assert r['parent_sha256']==parent['sha256'] and sha(parent['checkpoint'])==parent['sha256']
  assert sha(r['checkpoint'])==r['sha256']
  saved=torch.load(r['checkpoint'],map_location='cpu',weights_only=False);finite(saved);assert saved['iter']==s['final_iteration']==14992
  folder=ROOT/'training'/(r['group']+'_s'+str(r['training_seed']))
  loaded=read(folder/'loaded_state_audit.json');assert loaded['passed'] and loaded['checkpoint_sha256']==parent['sha256']
  a=read(folder/'actor_structure_audit.json')
  if r['group']=='sym_wrap':
   assert a['enabled'] and a['std_parameters_after']==18
   assert a['parent_std']==next(x['std']['parent_std'] for x in initial['results'] if x['seed']==r['training_seed'])
   assert a['shared_std']==next(x['std']['shared_std'] for x in initial['results'] if x['seed']==r['training_seed'])
  else:assert not a['enabled'] and a['true_noop']
 prove('ten_paired_final_models_exact1500_updates_4096env_parent_optimizer_restored_no_std_reset',True,[ROOT/'training_records.json',ROOT/'cohort.json'])
 configurations={}
 for record in tr:
  name=record['group']+'_s'+str(record['training_seed']);folder=ROOT/'training'/name/'params';folder.mkdir(exist_ok=True)
  source=Path(record['summary']['log_dir'])/'params';assert source.resolve().is_relative_to(REPO/'logs/rsl_rl/tw_stage1')
  for filename in ['env.yaml','agent.yaml']:
   target=folder/filename
   if not target.exists():os.link(source/filename,target)
   assert sha(target)==sha(source/filename)
 for seed in range(1,6):
  pair={}
  for filename in ['env.yaml','agent.yaml']:
   a=ROOT/'training'/('sym_cont_s'+str(seed))/'params'/filename;b=ROOT/'training'/('sym_wrap_s'+str(seed))/'params'/filename
   normalize=lambda text:re.sub(r'^(log_dir|run_name):.*$',lambda match:match.group(1)+': RUN_IDENTIFIER',text,flags=re.M)
   assert normalize(a.read_text())==normalize(b.read_text()),(seed,filename)
   pair[filename]={'sym_cont_sha256':sha(a),'sym_wrap_sha256':sha(b),'identical_except_top_level_run_identifiers':True}
  configurations[str(seed)]=pair
 configpath=ROOT/'matched_configuration_audit.json';configpath.write_text(json.dumps({'passed':True,'training_seed_pairs':5,'allowed_differences':['top-level env.log_dir','top-level agent.run_name'],'pairs':configurations},indent=2))
 prove('all_five_actual_saved_environment_and_PPO_configurations_matched',True,[configpath])

 budget=read(ROOT/'budget_accounting.json');auth=read(ROOT/'infrastructure_interruption/retry_authorization.json')
 assert auth['authorized'] and auth['retry_updates']==1500 and auth['retry_seed']==1 and auth['no_other_budget_extension']
 assert budget['successful_formal_models']==10 and budget['successful_formal_updates']==15000 and budget['updates_per_successful_model']==1500
 assert budget['failed_infrastructure_attempts']==1 and budget['failed_logged_updates']==1392 and budget['user_authorized_retry'] and budget['failed_attempt_excluded_from_formal_results'] and budget['extra_candidate_models']==0
 assert 'authorized_retry' in budget['retry_log_dir']
 for p,h in read(ROOT/'infrastructure_interruption/frozen_recovery_drivers.json')['files'].items():assert sha(p)==h,p
 prove('failed_infrastructure_compute_separate_and_retry_explicitly_authorized',True,[ROOT/'budget_accounting.json',ROOT/'infrastructure_interruption/retry_authorization.json',ROOT/'infrastructure_interruption/incident.json',ROOT/'infrastructure_interruption/frozen_recovery_drivers.json'])
 final=read(ROOT/'final_model_validation.json')
 assert final['passed'] and final['all_final_checkpoints_verified']==10 and final['wrapped_models']==5 and final['states_per_model']==102400
 for r in final['results']:
  assert r['max_equivariance_error']<=1e-5 and r['onnx_max_error']<=1e-5 and sha(r['onnx'])==r['onnx_sha256']
 prove('final_models_finite_symmetric_and_ONNX_validated',True,[ROOT/'final_model_validation.json'])
 jobs=read(ROOT/'evaluation_records.json');assert len(jobs)==60 and sum(j['initial_trials'] for j in jobs)==276480
 expected=read(ROOT/'evaluation_job_templates.json');assert [j['name'] for j in jobs]==[j['name'] for j in expected]
 models={r['group']+'_s'+str(r['training_seed']):r for r in tr}
 commands=0
 for j in jobs:
  d=read(j['output']);assert sha(j['output'])==j['output_sha256'] and d['checkpoint']==models[j['model']]['checkpoint']
  assert d['deterministic_policy'] and d['paired_resets'] and d['issue19_actor']==j['group']
  assert d['issue17_clock']['integer_enabled'] and d['issue17_clock']['period_steps']==20
  assert d['issue18_intervention']['mode']=='original' and d['issue18_intervention']['calls']==1000*len(j['commands'])
  track=d['issue18_tracking'];assert track['checkpoint_sha256']==models[j['model']]['sha256']
  assert track['model_optimizer_state_unchanged'] and track['model_state_before_sha256']==track['model_state_after_sha256']
  assert track['issue18_actual_feedback_verified'] and track['issue18_pre_action_phase_reconstruction_verified']
  assert [r['command'] for r in d['results']]==j['commands']
  physics=track['physics_audit']
  if j['protocol']=='clean':assert physics['mass_equal_nominal'] and physics['action_class']=='JointPositionAction' and not physics['pushes'] and not physics['observation_noise']
  else:assert physics['action_class']=='RandomOneStepDelayAction' and all(physics[k] for k in ['pushes','observation_noise','mass_event','material_event'])
  for result in d['results']:
   assert result['num_envs']==1024 and result['seconds']==20 and result['seed']==j['eval_seed']
   p=Path(track['output'])/(result['command']+'_feedback.json');f=read(p)
   assert f['passed'] and f['applied_action_max_error']==0 and f['last_action_max_error']==0 and f['actually_applied_action_checks']==1000
   assert f['pre_action_episode_step_reconstruction_exact']
   commands+=1
 assert commands==270
 prove('sixty_formal_jobs_paired_protocols_phase_feedback_no_double_wrapper',True,[ROOT/'evaluation_records.json',ROOT/'evaluation_job_templates.json',ROOT/'evaluation_preflight_validation.json'])
 portable=read(ROOT/'reference_reuse_portability.json');assert portable['passed'] and portable['reference_jobs']==45 and portable['all_reference_bytes_already_committed']
 for p,record in portable['files'].items():assert sha(p)==record['sha256'],p
 prove('fortyfive_historical_reference_jobs_reused_and_committed',True,[ROOT/'reference_reuse_manifest.json',ROOT/'reference_reuse_portability.json'])
 raw=read(ROOT/'raw_analysis_audit.json');assert raw['passed'] and raw['total_jobs']==105 and raw['command_records']==495 and raw['initial_trial_rows']==506880
 for p,h in raw['raw_inputs_sha256'].items():assert sha(p)==h,p
 assert sha(ROOT/'per_trial_heading_decomposition.csv.gz')==raw['per_trial_csv_sha256']
 prove('all_trials_heading_tracking_falls_startup_slope_halfcycles_with_censoring',True,[ROOT/'raw_analysis_audit.json',ROOT/'job_statistics.json',ROOT/'per_trial_heading_decomposition.csv.gz'])
 seeds=read(ROOT/'seed_statistics.json');groups=read(ROOT/'group_statistics.json');paired=read(ROOT/'paired_comparisons.json')
 draw=np.indices((5,)*5).reshape(5,-1).T
 def independent(values,saved):
  if any(x is None for x in values):
   assert saved['mean'] is None and saved['ci95'] is None and not saved['complete'];return
  a=np.array(values,dtype=float);assert a.shape==(5,) and np.isfinite(a).all()
  assert abs(float(a.mean())-saved['mean'])<1e-10
  assert np.allclose(np.percentile(a[draw].mean(1),[2.5,97.5]),saved['ci95'],atol=1e-10,rtol=1e-12)
  assert np.array_equal(a,np.asarray(saved['seeds']))
 jobstats=read(ROOT/'job_statistics.json')
 for key,values in seeds.items():
  assert len(values)==5
  protocol,group,cmd=key.split(':')
  for ts,seedvalue in enumerate(values,1):
   source=[jobstats[':'.join([protocol,group,'s'+str(ts),'eval'+str(es),cmd])] for es in [1,2,3]]
   for metric,value in seedvalue.items():
    observed=[x[metric] for x in source]
    if any(x is None for x in observed):assert value is None,(key,ts,metric)
    else:assert value is not None and abs(value-float(np.mean(observed)))<1e-10,(key,ts,metric)
  for metric,saved in groups[key].items():independent([v[metric] for v in values],saved)
 for key,metrics in paired.items():
  protocol,contrast,cmd=key.split(':');ref=contrast.removeprefix('sym_wrap-minus-')
  a=seeds[':'.join([protocol,'sym_wrap',cmd])];b=seeds[':'.join([protocol,ref,cmd])]
  for metric,saved in metrics.items():independent([None if x[metric] is None or y[metric] is None else x[metric]-y[metric] for x,y in zip(a,b)],saved)
 prove('independently_recomputed_all_five_seed_bootstrap_and_paired_intervals',True,[ROOT/'seed_statistics.json',ROOT/'group_statistics.json',ROOT/'paired_comparisons.json'])
 c=read(ROOT/'scientific_conclusion.json');assert c['rules_sha256']==RULE_SHA and c['recommendation_not_implemented'] and not c['default_promoted']
 def mean(protocol,group,cmd,metric):return groups[':'.join([protocol,group,cmd])][metric]['mean']
 def le(a,b):return a is not None and b is not None and a<=b
 def ge(a,b):return a is not None and b is not None and a>=b
 primary_ci=paired['clean:sym_wrap-minus-sym_cont:forward_05']['full20s_survivor_heading_mean_abs_degrees']['ci95']
 primary=primary_ci is not None and primary_ci[1]<0 and le(mean('clean','sym_wrap','forward_05','full20s_survivor_heading_mean_abs_degrees'),30)
 heading=all(le(mean('clean','sym_wrap',cmd,'full20s_survivor_heading_mean_abs_degrees'),30) for cmd in ['stand','forward_03','forward_05','forward_08'])
 tracking=[]
 for cmd in ['forward_05','forward_08']:
  refxy=mean('clean','control_int',cmd,'full_xy_error');refvx=mean('clean','control_int',cmd,'full_vx_mean')
  tracking.append(le(mean('clean','sym_wrap',cmd,'full_xy_error'),None if refxy is None else 1.05*refxy) and ge(mean('clean','sym_wrap',cmd,'full_vx_mean'),None if refvx is None else .95*refvx))
 for cmd in ['yaw_positive','yaw_negative']:
  ref=mean('clean','control_int',cmd,'full_yaw_error')
  tracking.append(le(mean('clean','sym_wrap',cmd,'full_yaw_error'),None if ref is None else 1.10*ref))
 fall=groups['combined:sym_wrap:forward_03']['fall_rate']
 robust=fall['ci95'] is not None and fall['ci95'][1]<.01 and le(mean('combined','sym_wrap','forward_05','fall_rate'),mean('combined','control_int','forward_05','fall_rate'))
 assert c['primary_passed']==primary and c['zero_yaw_heading_passed']==heading and c['tracking_recovered']==all(tracking) and c['robustness_passed']==robust and c['all_criteria_passed']==(primary and heading and all(tracking) and robust)
 prove('all_original_acceptance_conditions_recomputed_without_promotion',True,[ROOT/'scientific_conclusion.json',rules])
 selection=read(ROOT/'representative_selection.json');reconstruction=read(ROOT/'reconstruction_state.json')
 assert selection['candidate_count']==15360 and selection['group']=='sym_wrap' and not selection['survival_filter']
 assert reconstruction['all_prior_original_recorded_arrays_bitwise_equal'] and reconstruction['all_prior_original_result_rows_exactly_reproduced']
 plan=read(ROOT/'video_plan.json');assert len(plan)==1 and set(plan[0]['views'])=={'front','side','overhead'}
 for view in plan[0]['views'].values():assert sha(view['path'])==view['sha256']
 qa=read(ROOT/'video_quality_audit.json');assert qa['passed'] and len(qa['videos'])==3 and qa['encoding_frames']==3000 and qa['sampled_frames']==21
 manual=read(ROOT/'manual_visual_review.json')
 assert manual['actual_review_completed'] and manual['figures_reviewed']==4 and manual['contact_sheets_reviewed']==1 and manual['full_resolution_frames_reviewed']>=9
 for p,h in manual['reviewed_files_sha256'].items():assert sha(ROOT/p)==h,p
 prove('median_trial_three_precolored_views_XY_exact_reconstruction_actual_review',True,[ROOT/'representative_selection.json',ROOT/'reconstruction_state.json',ROOT/'video_quality_audit.json',ROOT/'manual_visual_review.json',ROOT/'media_source_sha256.json'])
 assert 'Ran 5 tests' in (ROOT/'decomposition_unit_tests.log').read_text() and (ROOT/'decomposition_unit_tests.log').read_text().rstrip().endswith('OK')
 for p in [ROOT/'results.md',ROOT/'seed_tables.md']:assert p.stat().st_size>1000
 changed=subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=REPO,text=True).splitlines()
 assert all(p in ['scripts/locomotion/train.py','README.md','docs/stage1_issue19_20261113.md'] for p in changed),changed
 prove('complete_report_seed_tables_tests_and_existing_defaults_rewards_USD_preserved',True,[ROOT/'results.md',ROOT/'seed_tables.md',ROOT/'figure_manifest.json',ROOT/'decomposition_unit_tests.log'])
 audit=dict(passed=True,requirements=evidence,scientific_acceptance_passed=c['all_criteria_passed'],publication_separate=True,source_sha256=sha(__file__))
 (ROOT/'prepublication_requirement_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 print('ISSUE19_PREPUBLICATION_REQUIREMENTS_PASS',len(evidence),'scientific_acceptance',c['all_criteria_passed'],flush=True)
if __name__=='__main__':main()
