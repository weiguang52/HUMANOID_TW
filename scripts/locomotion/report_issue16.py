"""Create the issue16 evidence report after actual manual scientific interpretation."""
import json
from pathlib import Path
from issue16_common import ROOT,sha
MAIN=['stand','forward_03','forward_05','forward_08']
ABS='full20s_survivor_heading_mean_abs_degrees'
SIGNED='full20s_survivor_heading_signed_mean_degrees'
def load(n):return json.loads((ROOT/n).read_text())
def n(v):
 return 'NA' if v is None else f'{v:.3f}' if isinstance(v,(int,float)) else str(v)
def ci(v):
 if v is None:return 'NA'
 c=v.get('bootstrap95')
 return n(v['mean'])+(' ['+n(c[0])+', '+n(c[1])+']' if c else ' (n=1; no CI)')
def main():
 for p in ['formal_evaluation_complete','initial_postprocessing_complete','automated_media_complete','derived_automatic_complete']:
  assert (ROOT/p).exists(),p
 conclusion=load('scientific_conclusion.json');manual=load('manual_review.json')
 assert manual['actual_review_completed'] and len(conclusion['next_recommendations'])==1
 gate=load('lineage_heading_gate.json');comparison=load('mirror_comparisons.json')
 runtime=load('runtime_symmetry_audit.json');physical=load('usd_symmetry_audit.json');integrity=load('analysis_integrity_audit.json')
 selection=load('representative_selection.json');default=load('no_policy_statistics.json')['statistics']
 lines=['# Issue16: reset-aware heading gate and mirror-source discrimination','',
 'Scope: [GitHub issue16](https://github.com/weiguang52/HUMANOID_TW/issues/16). Evaluation and analysis only. Zero training updates, no reward/production configuration/USD changes, and no default-model promotion.',
 '',
 '## Finding','',conclusion['summary'],'',
 'Overall classification: **'+conclusion['overall_classification']+'**. This is the observed-evidence classification, with limitations below; authored CAD differences alone do not establish their causal contribution to drift.','']
 lines+=['- '+v for v in conclusion['evidence']]
 lines+=['','Limitations:','']+['- '+v for v in conclusion['limitations']]
 lines+=['','Exactly one next proposal (not implemented): **'+conclusion['next_recommendations'][0]+'**.','',
 '## Frozen design and verified denominator','',
 'The 30-degree primary criterion was frozen before formal evaluation: each of stand/0.3/0.5/0.8 m/s must have mean absolute reset-aware quaternion20s heading drift <=30 degrees. Training-seed bootstrap95 upper bounds are reported separately; they do not silently replace the primary mean gate. Missing or survivor-selected endpoints and one-seed uncertainty remain explicit.',
 '',
 '34 formal jobs =11 existing models x3 evaluation seeds x6 commands, plus one default-PD no-policy stand job. There are199 command records and203776 initial trials. The four zero-yaw commands in every mirror job and the no-policy job provide133 full all-trial heading-series audits. Each formal job uses1024 environments and20s. No original lineage jobs were rerun. '+str(integrity['all_trial_heading_series_audited_commands'])+' complete command arrays passed independent quaternion-delta reconstruction; max accumulation discrepancy '+format(integrity['maximum_independent_heading_delta_sum_error_radians'],'.9g')+' rad (see exact precision in JSON).',
 '',
 'The five original command/reset ordinals plus stand retain issue15 ordering: forward_03,yaw_positive,yaw_negative,forward_05,forward_08,stand. The reflection acts on raw101 actor features before the existing MLP normalizer, with axial/polar signs and all30 actual joint axes, then reflects30 output actions. Critic observations, weights, optimizer and production configuration stay unchanged. Nine unit tests and four final runtime preflights passed; identity wrapping reproduces original metrics/state/RNG exactly.',
 '',
 'The initial preflight operator included a temporal half-cycle shift. Pure spatial reflection must leave the limb-independent per-episode scalar clock unchanged. This was corrected and tested before any formal job, with the superseded preflight preserved separately. Global clock means a limb-independent episode clock, not shared wall time across resets. See [amendment](operator_definition_amendment.json), [scope notes](analysis_scope_notes.md), [protocol](protocol_plan.md), [frozen293 input digests](frozen_evaluation_sources.json), [unit tests](mirror_unit_tests.log), and [final preflight audit](smoke_audit.json).',
 '',
 '## Original lineage: frozen mean-absolute heading gate','',
 'These44 cells reuse committed issue15 data. Survivor20s endpoints are conditional on no reset; prefixes and sums across reset-excluded segments are separate metrics. None of the11 lineage nodes passes all four primary cells.',
 '',
 '|Node|Command|Train n|Survivors / original N|Mean abs deg|Train95 upper deg|Mean gate|Endpoint scope|',
 '|---|---|---:|---:|---:|---:|---|---|']
 for r in gate['rows']:
  lines.append('|'+ '|'.join(map(str,[r['node'],r['command'],r['training_seeds'],f"{r['full20s_survivors']}/{r['original_initial_trials']}",n(r['mean_absolute20s_degrees']),n(r['training_bootstrap95_upper']),r['heading_mean_gate_pass'],r['endpoint_status']]))+'|')
 lines+=['','Full signed seed values, failures, prefix drift/exposure and reset-excluded segment metrics are in [lineage_heading_gate.json](lineage_heading_gate.json).','',
 '## Mirror comparisons','',
 'Each original/mirror training seed is matched; each seed is first averaged over the same three evaluation seeds. Intervals are unadjusted paired training-seed bootstrap95; robust05 has one existing training seed and no training-seed CI. Shared ancestry limits independence and causal identification. Endpoints with no survivors in any seed are not silently discarded.',
 '',
 '|Node|Zero-yaw command|Original signed deg [95]|Mirror signed deg [95]|Original mean abs deg [95]|Mirror mean abs deg [95]|Abs M/O|Observed pattern|',
 '|---|---|---:|---:|---:|---:|---:|---|']
 for r in comparison:
  if r['command'] not in MAIN:continue
  lines.append('|'+ '|'.join([r['node'],r['command'],ci(r['original']['metrics'][SIGNED]),ci(r['mirrored']['metrics'][SIGNED]),ci(r['original']['metrics'][ABS]),ci(r['mirrored']['metrics'][ABS]),n(r.get('mean_absolute_mirrored_over_original')),r['operational_mirror_pattern']])+'|')
 lines+=['','Magnitude band0.8..1.2 and1-degree signed-direction indeterminacy are descriptive frozen diagnostics. Zero signed M+O is a reference, not an equivalence test. The same-command yaw+/-0.3 controls remain raw turning-performance controls: a true reflection maps the command sign, so their same-command direction cannot diagnose zero-yaw bias reversal. See [complete comparisons, paired sums/differences and ratio CI](mirror_comparisons.json).','',
 '|Node|Command|Training seed|Original signed|Mirror signed|Original abs|Mirror abs|Original fall rate|Mirror fall rate|Original prefix s|Mirror prefix s|',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in comparison:
  for p in r['per_seed']:
   keys=['training_seed','original_signed','mirrored_signed','original_abs','mirrored_abs','original_fall_rate','mirrored_fall_rate','original_prefix_duration_s','mirrored_prefix_duration_s']
   lines.append('|'+ '|'.join([r['node'],r['command']]+[n(p[k]) for k in keys])+'|')
 lines+=['','[All job metrics](job_statistics.json), [seed metrics](seed_statistics.json), [group metrics](group_statistics.json), [203776 per-trial rows](per_trial_heading.csv.gz), and [raw audit bindings](analysis_integrity_audit.json) retain the complete denominator and all censoring.','',
 '## Exact robot and effective simulator readback','',
 'Source USD SHA-256: `'+physical['usd_sha256']+'`. All31 bodies,30 revolute joints and33 collision nodes are covered (31 authored descendant meshes with8692878 points plus2 analytic sole boxes). Vertex-cloud Hausdorff distances are tessellation-sensitive source diagnostics; exact PhysX-cooked hulls were not extracted. Central links are compared with their own reflection. Relative inertia residuals must be read with absolute kg m2 residuals.',
 '',
 'A separate64-environment zero-step validation reads actual PhysX masses, COM poses, inertia tensors, limits, drive gains, armature and caps after exact scene initialization, before checkpoint loading or policy invocation. It adds zero formal trials. Installed SDK docstrings establish local COM quaternion xyzw and column-major inertia at COM expressed in the rigid-body-prim frame. Authored USD drive attributes and effective ImplicitActuator overrides are kept distinct.',
 '',
 'Authored/effective body consistency: **'+str(runtime['body_authored_consistency_passed'])+'**; effective joint parameter checks: **'+str(runtime['joint_effective_parameter_checks_passed'])+'**. See [runtime audit and tolerances](runtime_symmetry_audit.json), [raw readback metadata](runtime_readback/metadata.json), [installed SDK evidence](runtime_readback/installed_sdk_getter_evidence.json), [joint parameter CSV](runtime_joint_parameters.csv), and [source geometry audit](usd_symmetry_audit.json).',
 '',
 '|Body|Partner|Mass difference %|COM residual mm|Inertia relative %|Inertia absolute kg m2|Source-cloud distance mm|',
 '|---|---|---:|---:|---:|---:|---:|']
 for r in physical['body_comparisons']:
  lines.append('|'+ '|'.join([r['body'],r['partner'],n(100*r['mass_relative_difference']),n(1000*r['COM_reflection_distance_m']),n(100*r['inertia_frobenius_relative_difference']),f"{r['inertia_frobenius_absolute_difference_kg_m2']:.9g}",n(1000*r.get('collision_vertex_cloud_hausdorff_m',0))])+'|')
 lines+=['','Complete joint sign/default/limit/PD table (actual axes at authored pose):','',
 '|Joint|Partner|Sign|Axis residual deg|Default rad|Reflected default error rad|Hard-limit error deg|Configured PD equal|',
 '|---|---|---:|---:|---:|---:|---:|---|']
 for r in physical['joint_comparisons']:
  lines.append('|'+ '|'.join([r['joint'],r['partner'],str(r['sign']),f"{r['axis_mismatch_degrees']:.9g}",n(r['default_radians']),n(r['default_reflection_absolute_difference_radians']),n(r['limit_max_difference_degrees']),str(r['PD_config_equal'])])+'|')
 lines+=['','## No-policy default-PD stand test','',
 'One1024-environment20s stand job, no actor called and default joint-position targets asserted after every step. No training-seed CI. This tests default-PD stand behavior, not a straight-walking controller. Falling prefixes do not become unconditional20s heading estimates.',
 '',
 '|Metric|Observed value|','|---|---:|']
 for k in ['full20s_heading_survivors','original_trial_count','fall_rate',SIGNED,ABS,'initial_prefix_all_heading_signed_mean_degrees','initial_prefix_all_heading_mean_abs_degrees','initial_prefix_mean_duration_s','all_episode_segments_heading_signed_mean_degrees','all_episode_segments_heading_mean_abs_degrees']:
  lines.append('|'+k+'|'+n(default[k])+'|')
 lines+=['','[No-policy statistics](no_policy_statistics.json) and the raw no_policy/ records preserve all1024 trials and resets.','',
 '## Original median-heading representative and colored replay','',
 f"Selection uses all{selection['candidate_count']} original control6000 forward0.5 trials, with no survival filter (all happened to survive). Pooled median absolute heading drift={selection['group_median_absolute_heading_degrees']:.6f}deg; nearest trial={selection['heading_abs_degrees']:.6f}deg, signed={selection['heading_signed_degrees']:.6f}deg. Training seed{selection['training_seed']}, evaluation seed{selection['eval_seed']}, env{selection['env_id']}. Selection is by heading, not body-yaw MAE.",
 '',
 'The one original source-job reconstruction must reproduce every previously recorded original result and moment array exactly before its poses are used. Replay sets saved poses, with no dynamics integration and no changed step counter. Front/side cameras follow position and heading; the overhead follows translation while preserving world heading. Every20s clip has1000 frames at50fps, world-XY trail/heading inset and cumulative reset-excluded heading overlay. The original and colored USD assets are hashed and1876 original attributes remain unchanged.',
 '',
 '[Selection and source digests](representative_selection.json), [exact reconstruction audit](reconstruction_state.json), [media source binding](media_source_sha256.json), [automatic video audit](video_quality_audit.json), and [actual manual review](manual_review.json).',
 '',
 '|View|Colored original-policy replay|','|---|---|']
 # Video names are authoritative in the generated plan; preserve all exact paths.
 plan=load('video_plan.json')
 for view,item in plan[0]['views'].items():
  relative=Path(item['path']).relative_to(ROOT)
  lines.append('|'+view+'|['+view+'20s video]('+str(relative)+')|')
 lines+=['','[Joint colors and legend](joint_colors.json). [Three-view contact sheet](video_contact_sheets/control6000.png).','','Generated video plan (exact paths):','', '```json',json.dumps(plan,indent=2),'```','',
 '## Figures','']
 for name in ['lineage_heading_gate','mirror_signed_heading','mirror_absolute_heading','paired_heading_sum_and_ratio','usd_reflection_differences','no_policy_heading_and_exposure','representative_world_heading']:
  lines+=['!['+name+'](figures/'+name+'.png)','']
 lines+=['All seven PNG/SVG figures, the three-view contact sheet and six full-resolution critical frames were actually reviewed; exact reviewed-byte hashes are in manual_review.json. [Figure bindings](figure_outputs_sha256.json) retain exact output bytes, including SVG.','',
 '## Reproducibility and archive','',
 'Use cohort.json for the11 existing checkpoint references/digests, formal_jobs.json for the frozen queue, and frozen_evaluation_sources.json for all293 immutable inputs. The source Git parent is770014ee9a4393da6dff34eced89a04d83e788dd. Production/train/eval source and model state stayed unchanged through formal evaluation. Large source artifacts, if chunked for publication, are reconstructible byte-for-byte using lossless_chunk_manifest.json. No mounted dataset was copied to the local client.',
 '',
 'Formal/raw validation: analysis_integrity_audit.json. Source and output binding: analysis_outputs_sha256.json, figure_outputs_sha256.json, media_source_sha256.json. Original baseline source manifest: committed stage1_issue15_20261111/sha256.json. The final Git commit and issue reply provide publication identity; they are not circularly embedded into their own commit.',
 '']
 (ROOT/'results.md').write_text('\n'.join(lines))
 binding={'source_sha256':sha(__file__),'report_sha256':sha(ROOT/'results.md'),'inputs':{str(ROOT/p):sha(ROOT/p) for p in ['scientific_conclusion.json','manual_review.json','lineage_heading_gate.json','mirror_comparisons.json','runtime_symmetry_audit.json','usd_symmetry_audit.json','analysis_integrity_audit.json','representative_selection.json','no_policy_statistics.json','video_plan.json']}}
 (ROOT/'report_outputs_sha256.json').write_text(json.dumps(binding,indent=2)+'\n')
 print('ISSUE16_RESULT_REPORT_WRITTEN',flush=True)
if __name__=='__main__':main()
