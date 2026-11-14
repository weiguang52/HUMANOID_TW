"""Build issue19 full results and exhaustive training-seed tables."""
import json,hashlib
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue19_20261113')
def read(n):return json.loads((ROOT/n).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(x):return 'N/A' if x is None else f'{x:.6g}'
def summary(x):
 if not x or x.get('mean') is None:return 'N/A (incomplete/censored)'
 return fmt(x['mean'])+' ['+', '.join(fmt(v) for v in x['ci95'])+']'
def main():
 c=read('scientific_conclusion.json');g=read('group_statistics.json');seeds=read('seed_statistics.json')
 tr=read('training_records.json');raw=read('raw_analysis_audit.json');initial=read('actor_pretraining_validation.json');final=read('final_model_validation.json')
 assert len(tr)==10 and raw['passed'] and final['passed']
 incident=read('infrastructure_interruption/incident.json');authorization=read('infrastructure_interruption/retry_authorization.json');assert authorization['authorized']
 budget=dict(successful_formal_models=10,updates_per_successful_model=1500,successful_formal_updates=15000,preflight_updates=6,failed_infrastructure_attempts=1,failed_logged_updates=incident['monitor_rows'],failed_executed_update_bounds=[incident['monitor_rows'],incident['monitor_rows']+1],failed_checkpoint_updates=8,user_authorized_retry=True,extra_candidate_models=0,failed_attempt_excluded_from_formal_results=True,retry_log_dir=next(r['summary']['log_dir'] for r in tr if r['group']=='sym_cont' and r['training_seed']==1))
 assert 'authorized_retry' in budget['retry_log_dir']
 (ROOT/'budget_accounting.json').write_text(json.dumps(budget,indent=2))
 lines=['# Issue19: strictly symmetric actor matched continuation','','## Outcome','',
 c['outcome']+'. All acceptance criteria met: '+str(c['all_criteria_passed'])+'. Primary: '+str(c['primary_passed'])+'; heading: '+str(c['zero_yaw_heading_passed'])+'; tracking: '+str(c['tracking_recovered'])+'; robustness: '+str(c['robustness_passed'])+'.',
 'No default configuration was promoted. One next recommendation, not implemented: '+c['recommendation'],
 'The issue background interpretations about policy asymmetry, physical causes and PPO mechanisms remain hypotheses. This intervention alone does not prove those causes. Issue17 scientific acceptance failure and the stopped SuperMimic status remain preserved.','',
 '## Frozen scope and methods','',
 'Two groups, sym_wrap/sym_cont, each five paired training seeds. Both resume the same issue17 sym_int final actor, critic and optimizer:4096 environments,1500 additional updates/model. No std reset, checkpoint selection, parameter scan or extension of the1500-update successful-model budget. One explicitly authorized infrastructure retry is accounted separately below. Critic, rewards and environment distributions remain unchanged. Both retain original sym_int mirror augmentation. Checkpoint logging frequency is reduced equally to conserve disk; final checkpoints determine evaluation.',
 'Rules SHA256: '+sha(ROOT/'classification_rules.json')+'. Five training seeds are bootstrap units; three evaluation seeds are averaged within each. All3125 paired seed resamples form percentile95% intervals. Individual intervals are not simultaneous family-wise coverage.',
 'Deterministic evaluation:1024 environments x3 evaluation seeds x20s; same paired reset seeds/command ordinals as references. Formal new count:60 jobs,270 command records,276480 initial trials. Reused references:45 jobs (issue17 control_int clean/combined and issue18 initial sym_int strict clean). Combined audit:105 jobs,495 records,506880 trials. Preflight and media reconstruction are excluded from formal denominators. No reference rollouts are rerun.',
 'Full20s heading endpoints use no-reset survivors. Censored initial prefixes are retained with exposure, never extrapolated to20s. Clean falls, all-valid segments and prefix metrics are reported alongside survivor endpoints. See raw_analysis_audit.json for historical missing instantaneous series.','',
 '## Actor, std and PPO','',
 'Mean:0.5*(mu(s)+Ma*mu(Ts)), using shared MLP weights in both branches. T/Ma match issue17/18, including gait sin/cos negation. Sampling, log-probability, entropy and KL all use this mean and the same shared diagonal Gaussian. Left/right std has one parameter per reflection orbit (18 parameters); central orbits remain independent.',
 'Initial shared std is the arithmetic parent std mean. Dimension-reduced std Adam first/second moments use orbit means with step retained; all other restored parameters and optimizer states remain unchanged. Full-state loading is audited before this explicit constrained-parameter migration.',
 'For this equivariant mean and orbit-equal std, p(Ma*a|T*s)=p(a|s): Ma is a signed permutation with absolute determinant1. Repeating original old log-probabilities for mirrored PPO samples is therefore self-consistent for sym_wrap, up to floating-point summation. The control retains original sym_int augmentation semantics.',
 'Five initial actors pass102400-state equivariance (max0), bitwise initial issue18-strict equality, probability/entropy/gradient checks and single-graph two-forward ONNX export. All three4096-env two-update preflights pass; disabled model/optimizer/RNG/environment states match prior continuation bitwise. Final actors are revalidated and exported to onnx/, with101 inputs/30 outputs.','',
 '## Main results: mean [training-seed95% CI]','',
 '|Protocol|Group|Command|Abs20s yaw deg|Fall rate|XY error m/s|Yaw error rad/s|Actual vx m/s|Initial exposure s|','|---|---|---|---:|---:|---:|---:|---:|---:|']
 metrics=['full20s_survivor_heading_mean_abs_degrees','fall_rate','full_xy_error','full_yaw_error','full_vx_mean','initial_prefix_mean_duration_s']
 for key,v in sorted(g.items()):lines.append('|'+ '|'.join([*key.split(':'),*[summary(v.get(k)) for k in metrics]])+'|')
 lines+=['','Primary paired sym_wrap-minus-sym_cont95% CI: '+str(c['primary_paired_difference_ci95'])+' degrees.',
 'Full acceptance values, thresholds, diagnostics and recommendation: [scientific_conclusion.json](scientific_conclusion.json).','',
 '## Startup versus continuing yaw','',
 'Startup is signed unwrapped yaw from reset t=0 to2.00s (100 post-step samples). Late per-trial OLS uses t=5.00 through20.00s (751 samples). Full slopes require the initial episode to survive20s. Separate censored-prefix slopes retain at least two available late samples and expose counts/duration. Reset jumps are excluded and episodes never joined. Startup plus15*slope is not a claimed reconstruction of total20s yaw:2-5s is separate.',
 'Half-cycles use actual pre-action episode_length_buf%20 (left0..9,right10..19) and post-step wrapped yaw increments, excluding terminal steps. Complete cycles are from initial episodes only; partial/all-valid totals and exposure are retained. Missing historical phase arrays are reconstructed from saved reset masks, never new simulation.','',
 '|Protocol|Group|Command|Startup signed deg|Late slope deg/s|Late abs slope deg/s|Left deg/cycle|Right deg/cycle|Sum deg/cycle|','|---|---|---|---:|---:|---:|---:|---:|---:|']
 keys=['startup_signed_degrees_mean','late_full20s_slope_degrees_per_second_mean','late_full20s_absolute_slope_mean','complete_cycle_left_mean_degrees','complete_cycle_right_mean_degrees','complete_cycle_sum_mean_degrees']
 for key,v in sorted(g.items()):
  if key.split(':')[-1] in ['forward_05','forward_08']:lines.append('|'+ '|'.join([*key.split(':'),*[summary(v.get(k)) for k in keys]])+'|')
 lines+=['','## Training and final models','','|Group|Seed|Updates|Final iteration|Wall seconds|SHA256|','|---|---:|---:|---:|---:|---|']
 for r in tr:lines.append('|'+ '|'.join([r['group'],str(r['training_seed']),'1500',str(r['summary']['final_iteration']),fmt(r['wall_seconds']),r['sha256']])+'|')
 lines+=['','## Initial std before and after sharing','','|Seed|Joint|Parent std|Shared initial std|','|---:|---|---:|---:|']
 contract=json.loads(Path('validation_artifacts/stage1_issue16_20261111/mirror/moments/control6000_s1_eval1/mirror_runtime_contract.json').read_text())
 for r in initial['results']:
  for joint,a,b in zip(contract['joint_names'],r['std']['parent_std'],r['std']['shared_std']):lines.append('|'+ '|'.join([str(r['seed']),joint,fmt(a),fmt(b)])+'|')
 lines+=['','Final std and ONNX errors: [final_model_validation.json](final_model_validation.json). Full gates: actor_pretraining_validation.json, preflight_validation.json and evaluation_preflight_validation.json.','',
 '## Figures and video','',
 '![Heading](figures/zero_yaw_heading.png)','![Tracking](figures/tracking_and_robustness.png)','![Decomposition](figures/startup_slope_and_halfcycle.png)','![Exposure](figures/clean_falls_and_exposure.png)','',
 'Representative: nearest absolute-heading median across all15360 sym_wrap initial0.5m/s trials, without a survival filter. With censoring, initial prefixes and exposure are explicit. See representative_selection.json. The formal job is reconstructed exactly and all saved arrays compared bitwise; rendering replays saved poses and integrates no new dynamics.',
 '[Front](videos/sym_wrap_front.mp4) | [Side](videos/sym_wrap_side.mp4) | [Overhead](videos/sym_wrap_overhead.mp4) | [Joint color legend](joint_colors.json). Same20s saved trajectory in all views, with worldXY trail. Front/side follow translation and heading; near-overhead follows translation at fixed world heading. Colors use the precolored derivative of the exact experiment USD with physical attributes unchanged. HumanML3D NPY is N/A for stage1 velocity-command tracking.',
 'Automatic encoding/frame/color checks and actual manual plot/frame/attachment review must pass separately before delivery.','',
 '## Complete evidence','',
 '[All seed tables](seed_tables.md); per_trial_heading_decomposition.csv.gz contains endpoints, startup, slopes, censoring and exposure. decomposition_raw/ retains per-trial arrays; evaluation/ retains raw moments, masks, phases, feedback audits and traces. Full scalar dictionaries: job_statistics.json, seed_statistics.json, group_statistics.json and paired_comparisons.json.',
 'Reference bytes remain in previously committed archives, bound by reference_reuse_manifest.json. Missing historical instantaneous series remains unavailable. GitHub and Notion record progress and results; Linear is not used. A failed behavioral gate remains a scientific result, not a successful acceptance claim.']
 at=lines.index('## Frozen scope and methods')
 lines[at:at]=['## Infrastructure interruption and authorized retry','', 'The first sym_cont_s1 attempt was interrupted by a data-disk-full infrastructure error during Git maintenance. It logged1392 updates (at most one further unlogged update may have executed), retained only an update-8 checkpoint, and produced no final model. Its logs, metrics and checkpoint remain in infrastructure_interruption/ and never enter formal result denominators. The user explicitly authorized rerunning this arm from the same original parent and seed for exactly1500 updates. There was no checkpoint selection or hyperparameter change. This failed computation is disclosed separately from the15000 successful formal model updates and six required preflight updates; see budget_accounting.json and the authorization receipt. The final cohort remains exactly five seeds per group.','']
 (ROOT/'results.md').write_text('\n'.join(lines)+'\n')
 table=['# Issue19 exhaustive training-seed tables','','Each seed cell averages its three evaluation seeds. Missing estimates remain N/A. Bootstrap units are these five training seeds, never individual environments.','']
 for key,values in sorted(seeds.items()):
  table+=['## '+key,'','|Metric|Seed1|Seed2|Seed3|Seed4|Seed5|Mean [95%]|','|---|---:|---:|---:|---:|---:|---|']
  for metric in sorted(set.intersection(*(set(v) for v in values))):table.append('|'+ '|'.join([metric,*[fmt(v[metric]) for v in values],summary(g[key][metric])])+'|')
  table.append('')
 (ROOT/'seed_tables.md').write_text('\n'.join(table)+'\n')
 (ROOT/'report_generation.json').write_text(json.dumps({'passed':True,'results_sha256':sha(ROOT/'results.md'),'seed_tables_sha256':sha(ROOT/'seed_tables.md'),'manual_review_separate':True},indent=2))
 print('ISSUE19_REPORT_GENERATED',c['outcome'],flush=True)
if __name__=='__main__':main()
