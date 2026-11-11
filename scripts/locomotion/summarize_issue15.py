"""Archive the fully reviewed issue15 study remotely, with committed checkpoint references."""
import json,os,re,shutil,subprocess
from pathlib import Path
from issue15_common import ROOT,REPO,sha,BASE,COMMANDS
DEST=REPO/"validation_artifacts/stage1_issue15_20261111"
def load(n):return json.loads((ROOT/n).read_text())
def num(x):return "NA" if x is None else f"{float(x):.6g}"
def show(s,scale=1):
 if s is None:return "NA"
 v=num(s["mean"]*scale);ci=s["bootstrap95"]
 return v+(" ["+", ".join(num(x*scale) for x in ci)+"]" if ci else " (1 training seed; no CI)")
def copy_file(source,target):
 source,target=Path(source),Path(target);target.parent.mkdir(parents=True,exist_ok=True)
 if source.suffix in [".npz",".npy",".mp4",".png",".gz"] and source.stat().st_dev==target.parent.stat().st_dev:
  os.link(source,target)
 else:shutil.copy2(source,target)
 return str(target)
def main():
 assert load("delivery_integrity_audit.json")["passed"] and load("video_physics_manual_audit.json")["passed"]
 assert not DEST.exists(),"Inspect an existing archive before any recovery"
 DEST.mkdir(parents=True)
 excluded={"progress.json","formal_driver_process.json","formal_jobs.json","postprocess_driver.json","push_receipt.json","issue_reply_receipt.json","notion_completion_receipt.json","completion_requirement_audit.json","issue_reply_body.md","issue_reply_body_final.md"}
 for p in ROOT.iterdir():
  if p.is_file() and p.name not in excluded and not p.name.startswith(("git_api_","archive","native_git","local_commit","staged_")) and p.suffix in [".json",".csv",".md",".log",".patch"]:
   copy_file(p,DEST/p.name)
 for name in ["smoke","clean","legacy","per_trial","figures","reconstruction","videos","video_review_frames","video_contact_sheets"]:
  shutil.copytree(ROOT/name,DEST/name,copy_function=copy_file,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
 cohort=load("cohort.json");nodes=load("node_statistics.json");conclusions=load("conclusions.json");jobs=load("job_statistics.json")
 needed={m["sha256"] for m in cohort["models"].values()};found={}
 checkpoint_names={Path(m["checkpoint"]).name for m in cohort["models"].values()}
 tracked=subprocess.check_output(["git","ls-files","validation_artifacts"]).decode().splitlines()
 candidates=sorted([REPO/p for p in tracked if Path(p).suffix==".pt" and Path(p).name in checkpoint_names],key=lambda p:(len(str(p)),str(p)))
 for p in candidates:
  h=sha(p)
  if h in needed and h not in found:found[h]=p
  if set(found)==needed:break
 assert set(found)==needed,("Unarchived input checkpoint hashes",needed-set(found))
 refs={}
 for name,m in cohort["models"].items():
  p=found[m["sha256"]];assert sha(p)==sha(m["checkpoint"])==m["sha256"]
  refs[name]={"node":m["node"],"training_seed":m["training_seed"],"checkpoint":os.path.relpath(p,DEST),"sha256":m["sha256"],"evaluation_model":m["evaluation_model"],"already_committed_at":BASE}
 (DEST/"model_references.json").write_text(json.dumps(refs,indent=2)+"\n")
 order=cohort["display_order"]
 def metric(n,c,k):return nodes[n][c]["metrics"].get(k)
 lines=["# Issue #15: straight-yaw lineage audit","","Issue: https://github.com/weiguang52/HUMANOID_TW/issues/15 . Base "+BASE+".",
 "**Evaluation-only study completed: zero training updates, no training configuration changes and no default promotion.**","",
 "## Findings","",
 f"The frozen >20% criterion identifies {len(conclusions['multiseed_positive_interval_jumps'])} multi-seed positive-interval comparisons and {len(conclusions['single_seed_descriptive20percent_jumps'])} single-seed descriptive comparisons across the fixed30 straight-command diagnostic contrasts. These are unadjusted comparisons, not independent causal discoveries.",
 "Every node/speed is explicitly classified as curve-dominant, oscillation-dominant or uncertain in the decomposition table. Single-seed classifications are descriptive. Do not cancel opposite per-trial curves using only the group signed mean.",
 "**Exactly one next recommendation, not implemented:** "+conclusions["next_recommendations"][0],"",
 *[text for item in conclusions["brief_findings"] for text in [item,""]],
 "The gait-phase input uses the [unchanged0.4s period](../../source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/tw/velocity_env_cfg.py#L39). Contact counts are force-threshold event proxies; they are not assumed to be exact anatomical steps.","",
 "## Frozen scope, protocol and read-only validation","",
 "37requested lineage entries correspond to35distinct checkpoints. Bounded seeds1/2 are byte-identical aliases of requested1500 seeds1/2; no duplicate independent-policy claim. All35 policies were newly evaluated under the same clean protocol, three evaluation seeds each:105jobs,630commands,645120initial trials. Three original-protocol FT5 jobs add2304trials, total108jobs/647424initial trials. No new training or checkpoint selection.",
 "Clean:1024environments x20s per command; deterministic actor, nominal masses, robot/terrain friction1.0, no pushes/observation noise,0ms delay, original requested-action observation. Command order forward0.3,yaw+0.3,yaw-0.3,forward0.5,forward0.8,stand preserves issue13/14 reset ordinals0..4 and appends stand5. Paired START reset seed=eval_seed*1000+ordinal. Policies that fail at different times may have different subsequent episode/reset/RNG paths; identical post-failure draws are not claimed. All policies101actor inputs; only existing CLOCK models use136critic inputs, others134. FT5 out-of-range commands are retained with actual failures.",
 "9prospective unit tests passed, plus two post-freeze supplementary spectral-coverage tests (not extra acceptance gates).64-env original-evaluator versus observational FT5 smoke gave exact metrics and bitwise-equal final root/joint state and CPU/CUDA/Python/NumPy RNG;136-wide CLOCK/stand smoke passed. These are zero-update preflight evaluations, not formal replications. Every job audits unchanged model/optimizer hash, actual clean physical properties and original metrics.221sources plus protocol/cohort/config/preflight inputs were frozen before formal evaluation. Derived analysis/media/report code was written after the freeze and is bound to actual source/output hashes.",
 f"All225existing first-five clean command records (control/treatment/clock) were compared with the exact prior committed issue13/14 data: all original result rows exact = {load('historical_clean_reproduction.json')['all_original_metrics_exactly_reproduced']}; all original moment arrays bitwise equal = {load('historical_clean_reproduction.json')['all_original_moment_arrays_bitwise_equal']}. These flags are observed consistency evidence, not a relaxed/new acceptance gate. [Full historical reproduction/differences](historical_clean_reproduction.json).",
 "[Prospective plan](protocol_plan.md) | [Source/input freeze](frozen_evaluation_sources.json) | [Read-only regression](smoke_audit.json) | [Independent full raw audit](analysis_integrity_audit.json) | [Delivery audit](delivery_integrity_audit.json)","",
 "## Metric definitions and inference","",
 "Yaw MAE = sum_valid |root_ang_vel_b[:,2]-command_yaw| / sum_valid, rad/s. XY error is original body-XY speed L2, m/s. Full20s nonterminal POST-STEP samples include ordinary episodes after failure reset, matching13/14. Also retain initial-episode-prefix sensitivity and original first-failure trial counts. Body-z yaw rate is not generally Euler-heading derivative under tilt.",
 "Average the three evaluation-seed job estimates within each training seed, then equally average training seeds. Bootstrap uses all n^n training-seed resamples (n=2 or5), percentile95%; n=1 explicitly has no training CI. Shared ancestors limit generalization, and all comparisons are multiple-comparison UNADJUSTED. Trial count does not replace training-seed replication. Fall rates are initial-trial first-failure proportions, not failure episodes.",
 "The fixed >20% criterion uses full-node mean yaw MAE change, requiring a positive difference95% interval for multi-seed later nodes. Same-seed multi-node contrasts use paired shared training seeds; if the earlier node has one training seed, later seeds are compared to that FIXED policy, with conditional inference only. Bounded->requested1500 has exact zero matched change and is an alias/cohort-expansion contrast, not a configuration effect. Full-node and matched-subset compositions are explicit in [complete comparisons](lineage_comparisons.json).","",
 "## Actual ancestry and configuration provenance","",
 "The issue display order is NOT a literal continuous parent-child chain. **FT5 and yaw_s1 share FT4; robust05 resumes yaw_s1, not FT5.** The frozen coarse phrase about FT5->robust05 intermediates refers only to a diagnostic comparison; actual saved resume paths below resolve the ancestry. Requested3000/4500 are intermediate/final checkpoints within the SAME continuation run from1500. Robust08/strengthen/bounded restart from robust05. Control/treatment/clock are sibling continuations of requested4500.",
 "```mermaid","graph TD"," FT4 --> FT5"," FT4 --> YAW[yaw_s1]"," YAW --> R05[robust05]"," R05 --> R08[robust08 seed1]"," R05 --> S[strengthen seeds1/2]"," R05 --> B[bounded/requested1500 seeds1/2]"," R05 --> N[requested1500 seeds3/4/5]"," B --> U3[requested3000]"," N --> U3"," U3 --> U45[requested4500 same continuation]"," U45 --> C[control6000]"," U45 --> T[treatment]"," U45 --> K[clock]","```","",
 "[Every saved env/agent config difference](configuration_lineage_audit.json) | [Exact saved resume/checkpoint parent graph](actual_resume_lineage.json). Extra ancestors are inspected for provenance, not silently added as evaluated policies. The FT5->robust05 contrast changes multiple training settings and ancestry; no sole cause can be identified there. Candidate recipe changes on later contrasts must not be confused with causal proof from a single seed or fixed baseline.","",
 "## Unified six-command yaw / xy / falls","",
 "| Node | Train n | Command | Yaw MAE rad/s95% | XY error m/s95% | First falls %95% |","|---|---:|---|---|---|---|"]
 for node in order:
  for cmd in ["stand","forward_03","forward_05","forward_08","yaw_positive","yaw_negative"]:
   lines.append(f"|{node}|{nodes[node][cmd]['training_seeds']}|{cmd}|{show(metric(node,cmd,'full_yaw_error'))}|{show(metric(node,cmd,'full_xy_error'))}|{show(metric(node,cmd,'fall_rate'),100)}|")
 lines+=["","![Unified lineage](figures/lineage_yaw_xy_falls.png)","","## Frozen20% comparisons and actual candidate differences","","| Comparison | Command | Full-node change % | Difference rad/s95% | Frozen criterion | Scope |","|---|---|---:|---|---|---|"]
 for c in conclusions["all_fixed_comparisons_with_configuration_context"]:
  label="descriptive1seed" if c["single_seed_descriptive_only"] else c["comparison_method"]
  if c["alias_cohort_expansion_not_configuration_effect"]:label="same-model alias/cohort expansion"
  lines.append(f"|{c['previous']} -> {c['current']}|{c['command']}|{100*c['relative_increase']:.3f}|{show(c['difference'])}|{c['meets_issue15_descriptive_or_multiseed_criterion']}|{label}|")
 lines+=["","The following30 stand/turning-command contrasts supplement the30prospectively frozen straight-command contrasts. They use the SAME >20%/positive-CI arithmetic, not a new decision rule or extra evaluation; all six command results remain visible. Additional comparisons are also unadjusted.","","| Comparison | Control command | Full-node change % | Difference rad/s95% | Descriptive20%/interval flag |","|---|---|---:|---|---|"]
 for c in conclusions["supplementary_control_command_comparisons"]:
  lines.append(f"|{c['previous']} -> {c['current']}|{c['command']}|{100*c['relative_increase']:.3f}|{show(c['difference'])}|{c['meets_issue15_descriptive_or_multiseed_criterion']}|")
 lines+=["","[Supplementary analysis coverage, not a changed prospective gate](analysis_coverage_addendum.md).",""]
 seen=set()
 for c in conclusions["all_fixed_comparisons_with_configuration_context"]:
  key=c["previous"]+" -> "+c["current"]
  if key not in seen:lines+=["**"+key+":** "+c["configuration_interpretation"],""];seen.add(key)
 lines+=["![Fixed criterion](figures/lineage20percent_comparisons.png)","","## Curve versus oscillation; heading drift","","Bias energy=sum_trial n*(trial mean yaw-target)^2; within energy=sum_trial n*temporal variance. Their sum exactly reconstructs yaw MSE. Within variation includes transients and ordinary post-reset episodes; it alone does not prove a coherent periodic oscillation. Within fraction>0.5 indicates oscillation/variation, <0.5 persistent curve; a multi-seed95% interval containing0.5 is uncertain. Mean signed bias, mean ABS trial bias and RMS trial bias are all retained, preventing opposite curves from canceling. Temporal SD is EACH trial's population SD(ddof0) about its own mean, then equal-trial average, not pooled SD.",
 "Heading uses wxyz-quaternion Euler yaw, summing wrapped successive angle differences over nonterminal steps with reset jumps excluded. Full20s signed/absolute drift is conditional on NO-reset survivors; failed trials retain initial-prefix duration/drift and full reset-excluded episode-segment drift. **Survivor drift is selected/censored and is not an unconditional20s endpoint estimate.** If an evaluation/training seed has no survivors, its endpoint statistic stays NA rather than dropping it.",
 "| Node | Speed | Signed yaw bias rad/s95% | RMS trial bias rad/s95% | Temporal SD rad/s95% | Within fraction95% | Classification |","|---|---|---|---|---|---|---|"]
 for node in order:
  for cmd in ["forward_03","forward_05","forward_08"]:
   label=conclusions["dominance_by_node_and_speed"][node][cmd]["classification"]
   lines.append(f"|{node}|{cmd}|{show(metric(node,cmd,'full_yaw_mean_signed_bias'))}|{show(metric(node,cmd,'full_yaw_rms_trial_bias'))}|{show(metric(node,cmd,'full_yaw_mean_trial_temporal_sd'))}|{show(metric(node,cmd,'full_yaw_within_energy_fraction'))}|{label}|")
 lines+=["","| Node | Speed | 20s survivors / original N | Survivor signed drift deg95% | Survivor mean absolute drift deg95% | All initial-prefix mean abs drift deg95% | All reset-excluded segments mean abs drift deg95% | Mean prefix duration s |","|---|---|---|---|---|---|---|---|"]
 for node in order:
  for cmd in ["forward_03","forward_05","forward_08"]:
   names=cohort["nodes"][node];surv=sum(jobs[f"{cohort['models'][n]['evaluation_model']}:eval{es}:{cmd}"]["full20s_heading_survivors"] for n in names for es in [1,2,3])
   vals=[show(metric(node,cmd,k)) for k in ["full20s_survivor_heading_signed_mean_degrees","full20s_survivor_heading_mean_abs_degrees","initial_prefix_all_heading_mean_abs_degrees","all_episode_segments_heading_mean_abs_degrees","initial_prefix_mean_duration_s"]]
   lines.append(f"|{node}|{cmd}|{surv}/{len(names)*3072}|"+"|".join(vals)+"|")
 lines+=["","![Yaw decomposition and heading](figures/yaw_decomposition_heading.png)","","## Oscillation frequency and gait relation","","Use each trial's initial uninterrupted nonterminal segment>=5s, no reset concatenation/missing-sample interpolation. Subtract mean, apply Hann window, rFFT; The planned residual peak in0.2..10Hz is retained alongside a supplementary strongest peak over ALL positive resolvable frequencies through Nyquist25Hz; the latter avoids silently omitting low/high-frequency behavior. This supplement followed a technical band-coverage check after evaluation began; no frozen decision changes. Per-trial frequency resolution1/duration is saved. Flat/short segments remain unsupported. Left/right foot force norm>1N contact rising edges are debounced0.1s; report ipsilateral stride frequencies and combined footstrike step frequency. Ratios/proximity to1x/2x are descriptive, not evidence of a causal mechanism. For curve-dominant cells the residual spectral peak is context, not a claim that oscillation dominates.",
 "| Node | Speed | Planned/full support / initial N | Planned peak Hz95% | Full-observable peak Hz95% | Mean step Hz95% | Full peak/stride95% | Full peak/step95% | Fraction full peak outside planned band95% |","|---|---|---|---|---|---|---|---|---|"]
 for node in order:
  for cmd in ["forward_03","forward_05","forward_08"]:
   names=cohort["nodes"][node];support=sum(jobs[f"{cohort['models'][n]['evaluation_model']}:eval{es}:{cmd}"]["spectrum_supported_trials"] for n in names for es in [1,2,3])
   fullsupport=sum(jobs[f"{cohort['models'][n]['evaluation_model']}:eval{es}:{cmd}"]["fullband_spectrum_supported_trials"] for n in names for es in [1,2,3])
   vals=[show(metric(node,cmd,k)) for k in ["yaw_peak_hz_mean","fullband_yaw_peak_hz_mean","footstrike_step_hz_mean","fullband_peak_over_stride_mean","fullband_peak_over_step_mean","fullband_peak_outside_planned_band_fraction"]]
   lines.append(f"|{node}|{cmd}|{support}/{fullsupport} / {len(names)*3072}|"+"|".join(vals)+"|")
 lines+=["","![Yaw/step frequency](figures/yaw_peak_and_step_frequency.png)","","[Planned-band mean normalized spectra and full-band support](spectra.json) | [Coverage supplement and unchanged frozen gates](spectral_coverage_addendum.md) | [Every unique trial's metrics, peak/resolution/contacts/support](per_trial_manifest.json). Initial-episode-prefix scalar sensitivity and both heading scopes remain in [seed statistics](seed_statistics.json), [node statistics](node_statistics.json) and raw arrays.","","## Original FT5 protocol and0.184rad/s comparison","","Same FT5 checkpoint hash under both protocols. Original #1 protocol is rerun at128env, eval seeds1/2/3, original six commands/order,20s, noisy observations/startup randomization, no pushes, original ordinary resets. It does NOT use clean nominal physics or paired clean reset streams. Historical0.183814rad/s is seed1 forward0.3 under that protocol, not a training-seed population mean.",
 f"All original metrics exactly reproduced across all three original-protocol jobs: **{conclusions['original_FT5_protocol_all_three_eval_seeds_exact']}**. Full per-metric differences and original/rerun falls are in [protocol comparison](original_ft5_protocol_comparison.json).","",
 "| Eval seed | Command | Original yaw rad/s | Rerun yaw rad/s | Original/rerun falls |","|---|---|---:|---:|---|"]
 for e in load("original_ft5_protocol_comparison.json"):
  for row in e["rows"]:lines.append(f"|{e['eval_seed']}|{row['command']}|{num(row['original_yaw'])}|{num(row['rerun_yaw'])}|{row['original_falls']}/{row['rerun_falls']}|")
 lines+=["",f"Unified-clean FT5 forward0.3 yaw MAE: **{show(metric('ft5','forward_03','full_yaw_error'))}**. Formula is identical; protocol/sample/reset differences change this same-policy estimate. Cross-protocol values must not be attributed to policy change. Any residual lack of exact historical reproduction is disclosed above and cannot be claimed away as proved protocol causality.",
 "","![FT5 protocol comparison](figures/original_ft5_protocol_comparison.png)","","## Two unfiltered true-median yaw examples","","FT5:all3072initial trials; control6000:all15360initial trials. Nearest group-median full-horizon trial yaw MAE, no survival filter; ties training seed,eval seed,envid. Reconstruct original1024-env job only for exact source states, verify all original metrics/raw moment/yaw/heading arrays, do not expand formal denominators.",
 "| Group | Model/eval seed/env | Trial yaw / group median rad/s | Reset frames | Front | Side |","|---|---|---|---|---|---|"]
 reconstruction=load("reconstruction_state.json")
 for e in load("video_plan.json"):
  s=reconstruction[e["group"]];front=Path(e["views"]["front"]["path"]).name;side=Path(e["views"]["side"]["path"]).name
  lines.append(f"|{e['group']}|{s['model']}/{s['eval_seed']}/{s['env_id']}|{num(s['yaw_error'])}/{num(s['group_median_yaw_error'])}|{s['reset_frames']}|[Front](videos/{front})|[Side](videos/{side})|")
 lines+=["","All four clips20s,1000frames,50fps,960x720,H264. Exact recorded root pose/world velocity/joint positions replayed without physics re-integration; display joint velocities are zero. Front/side source frames align within a group; FT5/control are not synchronized. Camera follows translation AND heading. Explicit world XY trail/heading inset and numeric reset-excluded heading drift make turning visible despite heading-following views. Precolored exact experiment USD preserves1876original attributes, no dynamic material binding. [Joint-color legend](joint_colors.json). HumanML source NPY **N/A: Stage1 velocity commands**.",
 "Four encodings/28sampled frames, two contact sheets, eight full-resolution arm-motion/heading-turn review frames and six statistical figures received actual manual review. This is sampled framing/color/joint-attachment/inset review, not every-frame pixel verification. [Video QA](video_quality_audit.json) | [Manual observations](video_physics_manual_audit.json).",
 "","![Representative world heading](figures/representative_world_heading.png)","","## Models, raw evidence and remote delivery","","All35distinct input models were already committed;37node aliases resolve to exact existing archive blobs. They are not retrained or silently replaced. Every reference hash was checked against the evaluated original checkpoint. [All model references](model_references.json).",
 "| Node/model | Existing committed full checkpoint | SHA256 |","|---|---|---|"]
 for name,m in refs.items():lines.append(f"|{name}|[Checkpoint]({m['checkpoint']})|{m['sha256']}|")
 lines+=["","All raw forward yaw/heading/contact series, masks/moments, scalar/per-trial statistics, six figures and four videos remain on the remote data disk and are included here. Lossless binary archive files use same-filesystem hard links to immutable finalized originals to avoid duplicating large raw arrays; their bytes and Git blobs are independently hashed. Logs strip ANSI/trailing whitespace; original SVG bytes are preserved exactly to retain plot source/output hash bindings. No mounted files were copied locally.",
 "[Analysis output/source binding](analysis_outputs_sha256.json) | [Figure binding](figure_outputs_sha256.json) | [Trajectory figure binding](trajectory_figure_sha256.json) | [Media/asset binding](media_source_sha256.json) | [Conclusions](conclusions.json). sha256.json covers every archive file except itself; publication independently verifies committed blobs and actual GitHub branch/tree/parent. Remote Linux is used for commit/push.",
 "","Tasks/results synchronize to [Notion](https://www.notion.so/403325d8231d8172995bf19510b8d760), replacing Linear per user instruction. This study does not reopen failed issue12/14 hypotheses or stopped SuperMimic policy acceptance."]
 (DEST/"README.md").write_text("\n".join(lines)+"\n")
 for p in DEST.rglob("*.log"):
  s=re.sub(r"\x1b\[[0-9;?]*[A-Za-z]","",p.read_text(errors="replace"));p.write_text("\n".join(x.rstrip() for x in s.splitlines()).rstrip()+"\n")
 # Preserve exact generated SVG bytes; whitespace is part of the source/output binding.
 manifest={str(p.relative_to(DEST)):sha(p) for p in DEST.rglob("*") if p.is_file() and p!=DEST/"sha256.json"}
 (DEST/"sha256.json").write_text(json.dumps(manifest,indent=2)+"\n");print("ISSUE15_ARCHIVED",len(manifest),flush=True)
if __name__=="__main__":main()
