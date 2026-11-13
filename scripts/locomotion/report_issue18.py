"""Publish complete tables without changing frozen scientific gates."""
import csv,json,hashlib
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue18_20261113')
GROUPS=['control6000','control_int','sym_int']
MODES=['original','strict']
COMMANDS=['forward_03','yaw_positive','yaw_negative','forward_05','forward_08','stand']
ABS='full20s_survivor_heading_mean_abs_degrees'
SIGNED='full20s_survivor_heading_signed_mean_degrees'
def read(n):return json.loads((ROOT/n).read_text())
def fmt(v):return 'NA (missing complete endpoint)' if v is None else f'{v:.3f}'
def interval(r):
 if r['mean'] is None:return 'NA (missing seed retained)'
 return f"{r['mean']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,row))+' |' for row in rows]])
def main():
 audit=read('raw_analysis_audit.json');assert audit['passed']
 g=read('group_statistics.json');s=read('seed_statistics.json')
 pairs=read('paired_comparisons.json');c=read('classification.json')
 v=read('projection_validation.json');selection=read('representative_selection.json')
 outcome=c['overall']
 titles={'A':'A: prospective policy-asymmetry criteria passed','B':'B: residual drift or noncancelling halves under strict projection','C':'C: intermediate, mixed or insufficient evidence'}
 title='C: wrapper destroyed control; no causal conclusion' if c['no_causal_conclusion'] else titles[outcome]
 lines=['# Issue #18: strict symmetric inference diagnostic','',title,'']
 def paragraph(text):lines.extend([text,''])
 def heading(text):lines.extend(['## '+text,''])
 paragraph('Evaluation only; zero training. Fifteen existing final checkpoints: control6000, control_int and sym_int, five training seeds each. Reuse 45 original-policy jobs; run 45 new strict-policy jobs. Each job has 1024 environments and six 20 s commands; each model has three evaluation seeds. Total: 540 command records and 552960 initial trials, including 270 new records and 276480 new trials.')
 if outcome=='C':
  x=c['cohorts']['sym_int'];reduction=(1-x['ratio'])*100
  paragraph(f"The interpretable sym_int cohort reduces forward0.5 mean absolute heading by {reduction:.2f}% ({x['original_forward05_abs_heading']:.3f} to {x['strict_forward05_abs_heading']:.3f} deg), with paired difference interval {x['paired_absolute_heading_difference_ci95']}. This is not A: only one cohort is interpretable and only two of four zero-yaw commands meet30deg. Strict forward0.3 remains62.462deg and forward0.8 remains30.621deg; thresholds are not relaxed.")
  paragraph('Both control cohorts exceed the prospective5percentage-point fall-gain guard at forward0.8 and are excluded from causal interpretation. The sym_int complete-cycle sum interval includes zero, which does not establish exact cancellation. This mixed evidence supports C, not a causal declaration that either policy asymmetry or USD asymmetry is the sole cause.')
  o=g['original:sym_int:forward_05'];a=g['strict:sym_int:forward_05']
  paragraph(f"The heading improvement also has performance tradeoffs: sym_int forward0.5 mean vx changes from {o['full_vx_mean']['mean']:.6f} to {a['full_vx_mean']['mean']:.6f}m/s, while XY error changes from {o['full_xy_error']['mean']:.6f} to {a['full_xy_error']['mean']:.6f}m/s. Heading reduction alone is not full control acceptance; all tracking, falls, prefixes and turn controls remain in the tables.")
 heading('Scope and validity')
 paragraph('pi_sym(s)=0.5[pi(s)+M_a pi(Ts)]. T is the issue17 spatial reflection plus half-cycle transform: 101 actor inputs, both gait sin/cos negated; M_a reflects 30 actions. T squared and M_a squared are identity. Deterministic actor means; no standard-deviation, reward, configuration or USD edits; no default promotion.')
 paragraph('control6000 preserves its legacy floating clock; control_int and sym_int preserve their existing integer clocks. All command order and reset ordinals are unchanged. Strict projection is the policy intervention.')
 paragraph('Every policy call checks last_action observations against actual action_manager.action. Every nonterminal applied action is checked against the strict output. Both 64-environment six-command clock preflights passed; all 45 formal jobs check feedback, exact phase reconstruction, model/optimizer immutability and zero training updates. See preflight_audit.json, evaluation_progress.json and per-command feedback files.')
 paragraph('Any cohort with mean fall-rate gain strictly above 0.05 in any zero-yaw command is labelled wrapper-destroyed-control and excluded from causal A/B interpretation. All its data remain reported. Algebraic policy equivariance does not prove stable closed-loop trajectories or physical symmetry. The issue17 PPO old-log-probability explanation remains an untested mechanism hypothesis.')
 heading('Prospective rules and uncertainty')
 h=hashlib.sha256((ROOT/'classification_rules.json').read_bytes()).hexdigest()
 paragraph('Classification rules were frozen before formal rollouts: classification_rules.json, SHA256 '+h+'.')
 paragraph('Average three evaluation seeds within each training seed, then use five training seeds as sample units. Paired differences preserve training-seed correspondence. Enumerate all 5^5=3125 equally weighted bootstrap draws; report unadjusted percentile 95% intervals. Parallel environments are not independent training seeds.')
 paragraph('Conservative overall convention, disclosed prospectively: A requires at least two interpretable cohorts and every interpretable cohort passing A; B requires at least one and every interpretable cohort passing B; otherwise C. Per-cohort results are retained, including mixed outcomes.')
 paragraph('Per-cohort A additionally requires forward0.5 mean absolute heading reduction at least70%, paired difference95% upper bound below zero, and at least3of4zero-yaw means at most30deg. Per-cohort B, after excluding A and fall-damaged groups, requires forward0.5 absolute heading above100deg or a nonzero complete-cycle half sum under the frozen interval convention. Otherwise C; no extra experiments.')
 paragraph('Complete 20 s headings require 1000 valid initial-episode steps and no first fall. Missing endpoints are never filled with prefixes; missing training seeds are retained. Signed heading is the sum of atan2(sin(delta yaw),cos(delta yaw)); absolute heading is taken per trial before averaging. Survivor endpoints are conditional and must be read alongside falls and prefix exposure.')
 paragraph('Original control6000 stand recordings contain all-trial heading accumulations and independently audited reference trajectories, but not all-trial instantaneous heading arrays. This source limitation is retained; original rollouts were not rerun. All available full arrays are independently reconstructed.')
 heading('Fixed-state equivariance and identity checks')
 paragraph('Same issue17 fixed 102400 states for each of 15 actual models. Tolerance 1e-5. Identity outputs are bitwise equal; actor/optimizer files and inference RNG are unchanged.')
 rows=[[r['model'],r['count'],f"{r['max_relative_error']:.3g}",f"{r['max_absolute_error']:.3g}",r['identity_bitwise_equal_all_states']] for r in v['results']]
 paragraph(table(['Model','States','Max relative E','Max absolute error','Identity bitwise'],rows))
 heading('Cohort classification')
 rows=[]
 for group in GROUPS:
  x=c['cohorts'][group]
  rows.append([group,x['classification'],x['interpretable'],fmt(x['original_forward05_abs_heading']),fmt(x['strict_forward05_abs_heading']),fmt(x['ratio']),str(x['paired_absolute_heading_difference_ci95']),x['commands_at_most30deg']])
 paragraph(table(['Cohort','Class','Interpretable','Original f0.5 abs deg','Strict f0.5 abs deg','Ratio','Paired diff CI95 deg','Commands <=30 deg /4'],rows))
 zero=['stand','forward_03','forward_05','forward_08']
 rows=[[group,*[fmt(c['cohorts'][group]['fall_gain_by_zero_yaw_command'][cmd]) for cmd in zero]] for group in GROUPS]
 paragraph(table(['Cohort','Stand fall gain','f0.3 gain','f0.5 gain','f0.8 gain'],rows))
 heading('All commands: cohort means and five-training-seed intervals')
 paragraph('XY/yaw errors and mean vx use all nonterminal full-horizon samples, including later episodes after a reset. Prefix metrics refer only to the initial episode. Falls count initial trials with a first fall. Turning controls preserve the original same-command reset order, without swapping yaw-positive and yaw-negative resets.')
 paragraph('![Zero-yaw heading](figures/zero_yaw_heading.png)')
 paragraph('![Tracking, falls and prefixes](figures/tracking_falls_prefix.png)')
 metrics=[SIGNED,ABS,'fall_rate','initial_prefix_mean_duration_s','full_xy_error','full_yaw_error','full_vx_mean','full20s_heading_survivors']
 headers=['Command','Mode','20s signed deg','20s abs deg','Fall fraction','Prefix s','XY m/s','Yaw rad/s','Mean vx m/s','Complete /1024']
 for group in GROUPS:
  paragraph('### '+group)
  rows=[[cmd,mode,*[interval(g[f'{mode}:{group}:{cmd}'][k]) for k in metrics]] for cmd in COMMANDS for mode in MODES]
  paragraph(table(headers,rows))
 heading('Forward 0.5 half-cycle yaw decomposition')
 paragraph('Bin each post-step quaternion yaw increment by PRE-action episode_length_buf modulo20: idx0..9 left, idx10..19 right. Exclude terminal/reset transitions. Old phase is reconstructed from recorded valid/reset masks: all 46080 original forward0.5 trials have 1000 valid steps without resets. New jobs save actual counters and verify reconstruction at every step.')
 paragraph('Complete-cycle means use only uninterrupted 20-step cycles in initial episodes; reset gaps are never joined. Average complete cycles within each evaluation job, then evaluation seeds within training seed. Left, right and their sum use exactly the same complete cycles. Retain partial-cycle valid steps and all-valid/prefix half totals and exposures to disclose instability rather than showing complete cycles alone.')
 paragraph('![Half-cycle yaw](figures/half_cycle_yaw.png)')
 rows=[]
 for group in GROUPS:
  for mode in MODES:
   m=g[f'{mode}:{group}:forward_05']
   vals=[interval(m['complete_cycle_'+p+'_mean_degrees']) for p in ['left','right','sum']]
   rows.append([group,mode,*vals,interval(m['complete_cycle_count']),interval(m['partial_cycle_valid_steps'])])
 paragraph(table(['Cohort','Mode','Left deg/cycle','Right deg/cycle','Sum deg/cycle','Complete cycles/job','Partial valid steps/job'],rows))
 for scope in ['full_horizon','initial_prefix']:
  paragraph('### '+scope+': mean totals per initial trial')
  rows=[]
  for group in GROUPS:
   for mode in MODES:
    m=g[f'{mode}:{group}:forward_05']
    keys=[scope+'_'+p+'_'+f for p,f in [('left','mean_total_degrees'),('right','mean_total_degrees'),('left','mean_seconds'),('right','mean_seconds')]]
    rows.append([group,mode,*[interval(m[k]) for k in keys]])
  paragraph(table(['Cohort','Mode','Left total deg','Right total deg','Left exposure s','Right exposure s'],rows))
 heading('Colored front and overhead source-state videos')
 paragraph('Select the nearest-median trial from all 15360 strict sym_int initial trials. No survivor filtering; if censoring exists, disclose and use the all-trial initial-prefix selection scope rather than silently calling it a complete-20s median.')
 paragraph(json.dumps({k:selection[k] for k in ['model','training_seed','eval_seed','env_id','selection_metric','selection_scope','censored_initial_trials','group_median_absolute_heading_degrees']},ensure_ascii=False))
 paragraph('[Front video](videos/sym_int_front.mp4) | [Overhead video](videos/sym_int_overhead.mp4)')
 paragraph('Capture the chosen projected-policy job deterministically and compare formal results and every existing NPZ array bitwise. This reconstruction does not add formal samples or rerun original policies. Replay exact saved poses without integrating dynamics; both views share the same trajectory. Front follows translation and heading; overhead follows translation with fixed world heading. Both show world XY trail and heading. Verify unchanged authored and display-colored USD hashes. Artifacts remain on the remote data disk.')
 heading('Exactly one next-step recommendation; not implemented here')
 if outcome=='A':
  paragraph('Continue from the existing parent with explicit mirror loss enabled (use_mirror_loss=True). Passing A supports retaining parent control while explicitly constraining symmetry; this inference-only experiment does not establish training convergence or prove the proposed PPO mechanism.')
 else:
  text='Do the original/reflected-policy x original/strictly-symmetric-USD 2x2 diagnostic, keeping current weights fixed, to distinguish policy and model/simulation factors.'
  text+=' Wrapper damage makes this experiment causally uninterpretable; do not infer physical asymmetry from it.' if c['no_causal_conclusion'] else ' Structural equivariance alone does not prove USD asymmetry.'
  paragraph(text)
 heading('Reproduction and evidence')
 paragraph('run_issue18_evaluations.py: bounded 45-job queue and frozen inputs. analyze_issue18.py: all 540 records and 552960 initial trials, seed pairing and censoring. validate_issue18_projection.py: all original fixed states and actual checkpoints. test_issue18_analysis.py: reset, phase, wrap, complete-cycle and missing-seed checks.')
 paragraph('Evidence: formal_inputs_frozen.json, original_reuse_manifest.json, raw_analysis_audit.json, projection_validation.json, video_quality_audit.json, manual_visual_review.json and sha256.json. [All seed tables](seed_tables.md), machine-readable job/seed/cohort JSON, per_trial_heading.csv.gz, half_cycle_raw NPZs and formal raw moments are archived.')
 paragraph('Issue17 historical primary failure is preserved. Zero training and no default promotion in issue18; SuperMimic remains stopped.')
 (ROOT/'results.md').write_text('\n'.join(lines))
 seedlines=['# Issue #18: all training-seed command tables','','Each row averages three evaluation seeds. NA retains a missing endpoint; no seed is dropped.','']
 for group in GROUPS:
  for cmd in COMMANDS:
   seedlines.extend(['## '+group+' / '+cmd,''])
   rows=[[mode,ts,*[fmt(s[f'{mode}:{group}:{cmd}'][ts-1][k]) for k in metrics]] for mode in MODES for ts in range(1,6)]
   seedlines.extend([table(['Mode','Train seed',*headers[2:]],rows),''])
 (ROOT/'seed_tables.md').write_text('\n'.join(seedlines))
 for name,data in [('job_statistics.csv',read('job_statistics.json')),('paired_comparisons.csv',pairs)]:
  with (ROOT/name).open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['key','metric','value_json'])
   for key,values in data.items():
    for metric,value in values.items():w.writerow([key,metric,json.dumps(value,ensure_ascii=False)])
 print('ISSUE18_REPORT_WRITTEN',title,flush=True)
if __name__=='__main__':main()
