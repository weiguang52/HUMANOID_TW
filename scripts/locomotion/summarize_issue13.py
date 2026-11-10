"""Archive the completed reviewed issue13 analysis on the remote disk."""
import csv,json,re,shutil,hashlib
from pathlib import Path
from issue13_common import ROOT,REPO,BASE,CLEAN,FORWARD,sha
from issue13_tracking import SCOPES
DEST=REPO/"validation_artifacts/stage1_issue13_20261110"
OLD=REPO/"validation_artifacts/stage1_issue12_20261110"
def load(name):return json.loads((ROOT/name).read_text())
def num(x):return "NA" if x is None else f"{float(x):.6g}"
def pct(x):return "NA" if x is None else f"{100*float(x):+.2f}%"
def interval(values,percent=False):
 if values is None:return "NA"
 f=pct if percent else num
 return "["+f(values[0])+", "+f(values[1])+"]"
def main():
 assert load("delivery_integrity_audit.json")["passed"] and load("video_physics_manual_audit.json")["passed"]
 assert not DEST.exists(),"Inspect an existing archive; do not blindly overwrite it"
 DEST.mkdir(parents=True)
 exclude={"progress.json","formal_driver_process.json","postprocess_driver.json","formal_job_state.json","push_receipt.json","issue_reply_receipt.json","completion_requirement_audit.json","issue_reply_body.md"}
 for p in ROOT.iterdir():
  if p.is_file() and p.name not in exclude and not p.name.startswith(("git_api_","archive","native_git","local_commit","staged_")) and p.suffix in [".json",".csv",".md",".log",".patch"]:
   shutil.copy2(p,DEST/p.name)
 for folder in ["smoke","clean","combined","per_trial","figures","reconstruction","videos","video_review_frames","video_contact_sheets"]:
  shutil.copytree(ROOT/folder,DEST/folder,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
 models=load("input_models.json");old=json.loads((OLD/"sha256.json").read_text());references={}
 for name,m in models.items():
  path="models/"+name+"/"+Path(m["checkpoint"]).name
  assert sha(OLD/path)==old[path]==m["sha256"]
  references[name]={"sha256":m["sha256"],"checkpoint":"../"+OLD.name+"/"+path,"group":m["group"],"training_seed":m["training_seed"]}
 (DEST/"model_references.json").write_text(json.dumps(references,indent=2))
 assert sha(OLD/"joint_colors.json")==old["joint_colors.json"];shutil.copy2(OLD/"joint_colors.json",DEST/"joint_colors.json")
 comp=load("paired_comparisons.json");gates=load("primary_gate_comparisons.json");decision=load("decision.json");sensitivity=load("classification_sensitivity.json")
 names={"forward_03":"0.3 m/s","forward_05":"0.5 m/s","forward_08":"0.8 m/s","yaw_positive":"yaw +0.3 rad/s","yaw_negative":"yaw -0.3 rad/s"}
 def get(condition,cmd,part,metric,scope=SCOPES[0]):return comp[":".join([condition,cmd,scope,part,metric])]
 verdict={"A_cost_localized_to_recovery_window":"A: the prespecified point bounds localize the tracking cost to recovery.",
  "B_global_side_effect":"B: at least one clean/outside tracking contrast supports a global side effect.",
  "unable_to_distinguish":"The frozen criteria cannot distinguish the two explanations."}[decision["decision"]]
 lines=["# Issue #13: where does post-push pitch recovery lose tracking?","",
  "Issue: https://github.com/weiguang52/HUMANOID_TW/issues/13 . Base commit: "+BASE+".","",
  "**Analysis only: zero new training updates. The issue #12 rejection and all production defaults remain unchanged.**","",
  "## Result","",verdict,"",
  "Frozen classification: **"+decision["decision"]+"**. It does not grant policy acceptance or change the previous 10% tracking gate.","",
  "**The only next-step recommendation:** "+decision["only_next_step_recommendation"]["one_recommendation"],
  decision["only_next_step_recommendation"]["basis"]+" No proposed structural change was implemented.","",
  "## Frozen decision contrasts","",
  "All clean xy/yaw errors for all five commands, plus combined OUTSIDE-window xy at all three forward speeds, are the 13 decision contrasts. A requires every treatment mean <=1.05*control mean. B requires any treatment mean >1.05*control mean AND its paired absolute-difference 95% lower endpoint >0. Otherwise the result is unresolved. A uses the specified point bounds, not an invented confidence non-inferiority test.","",
  "| Condition | Command | Metric (unit) | Control mean ± seed SD | Treatment mean ± seed SD | Relative change | Paired T-C 95% | <=5% point bound | Confirmed >5% |",
  "|---|---|---|---|---|---|---|---|---|"]
 for key,d in gates.items():
  condition,cmd,scope,part,metric=key.split(":");unit="m/s" if metric=="xy_error" else "rad/s"
  lines.append(f"| {condition} {part} | {names[cmd]} | {metric} ({unit}) | {num(d['control']['mean'])} ± {num(d['control']['sample_sd'])} | {num(d['treatment']['mean'])} ± {num(d['treatment']['sample_sd'])} | {pct(d['relative_change'])} | {interval(d['paired_difference']['bootstrap95'])} | {d['point_not_worse_than5percent']} | {d['point_worse_than5percent_and_positive95']} |")
 lines += ["","![Frozen contrasts](figures/frozen_gate_contrasts.png)","",
  "Inference unit: FIVE matched training seeds. First average the three evaluation-seed summaries within each training seed, then give the five seeds equal weight. All 5^5=3,125 paired joint index resamples yield percentile 95% intervals. Seed SD uses ddof=1. **Multiple comparisons are not corrected**; the any-of-13 B rule and all exploratory intervals retain that limitation. Evaluation environments are not independent training replications. Shared upstream ancestry and fixed reset streams limit generalization.","",
  "Zero control means are handled explicitly: T=0 passes the point bound, T>0 is a degradation; undefined ratios are NA, with no pseudocount or invented infinity. Absolute paired intervals remain available.","",
  "[All paired differences and intervals](paired_metric_comparisons.csv) | [Full per-seed statistics](paired_comparisons.json) | [Decision](decision.json) | [Prospective protocol](protocol_plan.md)","",
  "## Cohort and clean-condition fidelity","",
  "Ten exact issue #12 final checkpoints, five control/treatment pairs; no checkpoint selection, optimizer updates or std reset. Clean: 30 jobs, 1,024 environments x three evaluation seeds x five separate commands x 20 seconds = 153,600 initial trials. Combined: 30 jobs x three forward speeds x 1,024 x 20 seconds = 92,160 initial trials. Total: **60 formal jobs, 240 command records, 245,760 initial trials**. Two later reconstruction jobs only capture selected existing trajectories and do not enlarge these denominators.","",
  "Clean uses the existing evaluator's clean switch: nominal per-link masses, actual robot static/dynamic friction 1.0/1.0, terrain friction 1.0/1.0, ordinary JointPositionAction (zero delay), no pushes, no observation noise and no material/mass randomization. These actual physics values are audited in every clean job. Policy/critic observations stay 101/134; model and optimizer state digests do not change under inference.","",
  "Clean retains the original evaluator order: forward 0.3, yaw +0.3, yaw -0.3, forward 0.5, forward 0.8. Combined retains 0.3/0.5/0.8. Paired resets use eval_seed*1000+command_index within each protocol and across models; different protocols do not claim identical physical draws.","",
  "![Clean tracking](figures/clean_tracking.png)","",
  "## Falls, reported without substituting for tracking","",
  "| Condition | Command | Control falls/N | Control seed-mean 95% | Treatment falls/N | Treatment seed-mean 95% | Paired T-C 95% (percentage points) |",
  "|---|---|---|---|---|---|---|"]
 for key,v in load("fall_results.json").items():
  condition,cmd=key.split(":");c,t=v["control"],v["treatment"];ci=v["paired_difference"]["bootstrap95"]
  lines.append(f"| {condition} | {names[cmd]} | {c['pooled_falls']}/15360 ({100*c['seed_summary']['mean']:.4f}%) | {interval([100*x for x in c['seed_summary']['bootstrap95']])}% | {t['pooled_falls']}/15360 ({100*t['seed_summary']['mean']:.4f}%) | {interval([100*x for x in t['seed_summary']['bootstrap95']])}% | {interval([100*x for x in ci])} |")
 lines += ["","Each policy/command pools 3,072 initial trials. Intervals use the five training seeds. Zero observed falls do not establish zero population risk. Every original issue #12 combined output field and all original trial/failure arrays were exactly reproduced by the augmented evaluator.","",
  "## Inside and outside the push-recovery window","",
  "Primary sampling matches issue #7/#12: non-terminal POST-STEP samples across the full 20 seconds, including later ordinary episodes after a failure reset. All initial environment/trial IDs remain represented. Per-trial sample counts and undefined zero-exposure cells are retained. The initial-episode-only sensitivity stops each environment at its first reset.","",
  "Primary recovery is post-event time [push, push+1s): post-step samples k..k+49 at dt=0.02. Clocks clear on reset BEFORE interval pushes, preventing cross-episode leakage. The training reward was computed before interval pushes, so its reward-aligned k+1..k+50 convention is separately reported as a one-step boundary sensitivity. They are not silently equated.","",
  "| Speed | Partition | Control xy (m/s) | Treatment xy | Paired xy T-C 95% | Control yaw (rad/s) | Treatment yaw | Paired yaw T-C 95% |",
  "|---|---|---|---|---|---|---|---|"]
 for cmd in FORWARD:
  for part in ["all","inside","outside"]:
   x=get("combined",cmd,part,"xy_error");y=get("combined",cmd,part,"yaw_error")
   lines.append(f"| {names[cmd]} | {part} | {num(x['control']['mean'])} | {num(x['treatment']['mean'])} | {interval(x['paired_difference']['bootstrap95'])} | {num(y['control']['mean'])} | {num(y['treatment']['mean'])} | {interval(y['paired_difference']['bootstrap95'])} |")
 evalrows=list(csv.DictReader((ROOT/"evaluation_metric_rows.csv").open()))
 lines += ["","![Conditional tracking](figures/window_tracking.png)","",
  "| Speed | Partition | Control valid frame-samples | Treatment valid frame-samples | Control mean sample fraction | Treatment mean sample fraction |",
  "|---|---|---:|---:|---|---|"]
 for cmd in FORWARD:
  for part in ["inside","outside"]:
   d=get("combined",cmd,part,"sample_fraction")
   counts={g:sum(int(float(x["valid_samples"])) for x in evalrows if x["condition"]=="combined" and x["command"]==cmd and x["scope"]==SCOPES[0] and x["partition"]==part and x["group"]==g) for g in ["control","treatment"]}
   lines.append(f"| {names[cmd]} | {part} | {counts['control']:,} | {counts['treatment']:,} | {pct(d['control']['mean'])} | {pct(d['treatment']['mean'])} |")
 lines += ["","Frame-samples are descriptive exposures, not the inferential sample size. The primary means pool valid frames within each evaluation job to reproduce #12; equal-trial conditional means are additionally retained as xy_equal_trial_mean/yaw_equal_trial_mean in all paired tables. Per-trial mean errors are in the losslessly compressed CSVs; both sampling scopes and both boundary conventions are fully retained in the raw moment NPZs.","",
  "| Speed | Overall xy T-C | Inside weighted contribution T-C 95% | Outside weighted contribution T-C 95% |",
  "|---|---|---|---|"]
 for cmd in FORWARD:
  d=get("combined",cmd,"all","xy_error");i=get("combined",cmd,"inside","xy_weighted_contribution");o=get("combined",cmd,"outside","xy_weighted_contribution")
  lines.append(f"| {names[cmd]} | {num(d['paired_difference']['mean'])} | {num(i['paired_difference']['mean'])} {interval(i['paired_difference']['bootstrap95'])} | {num(o['paired_difference']['mean'])} {interval(o['paired_difference']['bootstrap95'])} |")
 lines += ["","![Weighted decomposition](figures/weighted_error_contributions.png)","",
  "The inside/outside exposure-weighted contributions add exactly to the original overall error before and after averaging evaluation/training seeds. Contribution changes combine conditional error and occupancy/reset effects; these unadjusted comparisons do not prove a causal mechanism.","",
  "## Systematic slowing versus temporal fluctuation","",
  "Forward velocity is body-frame vx, consistent with the command. Bias = mean vx - target vx. Fluctuation = each trial's temporal population SD of vx about its own conditional mean (ddof=0), then an equal-trial mean over trials with samples. Pooled SD, weighted within-trial variance and between-trial mean variance are retained separately; within+between reconstructs total variance within each evaluation job/partition. They are not interchangeable. Signed velocity/bias contrasts use absolute paired intervals; relative changes are not interpreted for signed values.","",
  "| Condition/partition | Speed | Control vx | Treatment vx | Paired bias T-C 95% | Control temporal SD vx | Treatment temporal SD vx | Paired SD T-C 95% |",
  "|---|---|---|---|---|---|---|---|"]
 for condition,part in [("clean","all"),("combined","all"),("combined","inside"),("combined","outside")]:
  for cmd in FORWARD:
   v=get(condition,cmd,part,"vx_mean");bias=get(condition,cmd,part,"vx_bias");sd=get(condition,cmd,part,"vx_temporal_sd_mean")
   lines.append(f"| {condition}/{part} | {names[cmd]} | {num(v['control']['mean'])} | {num(v['treatment']['mean'])} | {num(bias['paired_difference']['mean'])} {interval(bias['paired_difference']['bootstrap95'])} | {num(sd['control']['mean'])} | {num(sd['treatment']['mean'])} | {num(sd['paired_difference']['mean'])} {interval(sd['paired_difference']['bootstrap95'])} |")
 lines += ["","All values in the velocity/SD table are m/s. Lateral and angular means/temporal SDs and all paired intervals are also in the complete tables.","",
  "![Bias and fluctuation](figures/bias_and_variability.png)",""]
 for condition,part in [("clean","all"),("combined","outside")]:
  b=get(condition,"forward_08",part,"vx_bias")["paired_difference"];s=get(condition,"forward_08",part,"vx_temporal_sd_mean")["paired_difference"]
  direction="lower" if b["bootstrap95"][1]<0 else ("higher" if b["bootstrap95"][0]>0 else "not resolved in direction")
  lines.append(f"At 0.8 m/s under {condition}/{part}, forward-speed change is {num(b['mean'])} m/s, paired 95% {interval(b['bootstrap95'])} ({direction}); temporal SD change is {num(s['mean'])} m/s, paired 95% {interval(s['bootstrap95'])}.")
 lines += ["","## Prespecified sensitivities","",
  "| Analysis | Classification |","|---|---|",
  "| Primary full-horizon post-event window | "+decision["decision"]+" |",
  "| Reward-aligned one-step boundary | "+sensitivity["reward_aligned_boundary"]["decision"]+" |",
  "| Initial episode only | "+sensitivity["initial_episode_only"]["decision"]+" |","",
  "The primary criterion is never reselected after seeing a sensitivity result. Full sensitivity means and intervals are retained, including zero-exposure counts.","",
  "![Sensitivity](figures/window_scope_sensitivity.png)","",
  "## Two representative trials and four exact-state videos","",
  "For each group choose the trial nearest the group's median full-horizon 0.8 m/s xy error across all 15,360 combined trials; ties: training seed, evaluation seed, environment ID. No survival filter is applied. These are median illustrations, not worst-case or new acceptance samples. Selected original 1,024-env evaluations are reconstructed only to capture states; all original metrics, full records and moments match exactly.","",
  "| Group | Model / eval seed / env | Selected xy / group median | Push times (s) | Reset frames | Front | Side |","|---|---|---|---|---|---|---|"]
 reconstructed=load("reconstruction_state.json");plan=load("video_plan.json")
 for e in plan:
  s=reconstructed[e["group"]];front="videos/"+Path(e["views"]["front"]["path"]).name;side="videos/"+Path(e["views"]["side"]["path"]).name
  lines.append(f"| {e['group']} | {e['model']} / {s['eval_seed']} / {s['env_id']} | {num(s['xy_error'])} / {num(s['group_median_xy'])} | {', '.join(f'{t:.2f}' for t in s['push_seconds'])} | {s['reset_frames']} | [Front]({front}) | [Side]({side}) |")
 lines += ["","Each clip has 1,000 frames at 50 fps, 960x720, 20 seconds. Playback writes the exact recorded post-step root/joint states and does NOT integrate new physics. The two views within a selected trial share exact source samples/times; control and treatment examples are not synchronized to each other. A red timeline marks actual push times, with the following one-second interval shaded; reset frames are explicitly labeled and excluded from metrics.","",
  "The display uses the exact precolored experiment asset: 1,876 original non-material/physics attributes preserved, no material binding after initialization. Camera follows translation AND heading. Display mass/materials are nominal; the source randomized physics are documented but not re-integrated. [Joint-color legend](joint_colors.json). Stage1 velocity commands have no HumanML source NPY: **N/A**.","",
  "All four encodings/frame counts, 24 sampled frames (including a near-push response sample per clip), two contact sheets, four full-resolution motion frames and six statistical figures were checked. Attachment/color/framing review is sampled, not a claim about every pixel of every frame. See [automated video audit](video_quality_audit.json) and [actual manual review](video_physics_manual_audit.json).","",
  "## Validation, artifacts and traceability","",
  "Ten analytic/window/reset/bootstrap unit tests pass. Original-vs-observed clean and combined runtime smoke runs have bitwise-equal original output fields, final root/joint states and full CPU/CUDA/NumPy/Python RNG states. Combined full-trial/failure arrays also match. The observer samples no RNG and writes no actions, policy observations, reward configuration or physics. Independent CPU reconstruction checks every packed time mask/count, sums/second moments, mixture/variance identity and raw reference-environment moments for all 240 command records.","",
  "The protocol and 42 behavior/dependency/statistical source hashes were frozen before formal evaluation. Later derived-analysis/plot/replay/report code is bound by source/output SHA records, rather than being falsely described as frozen before formal data. Only the requested clean evaluation condition switches differ from combined; no production or training configuration is edited.","",
  "[Requirement audit](delivery_integrity_audit.json) | [Full input/moment integrity](analysis_integrity_audit.json) | [Smoke proof](runtime_smoke_audit.json) | [Frozen sources](frozen_sources.json) | [Analysis source/output hashes](analysis_outputs_sha256.json) | [Per-trial CSV manifest](per_trial_csv_manifest.json) | [Representative selection](representative_selection.json)","",
  "Raw per-trial count/sum/sum-of-squares arrays, packed masks, per-step cohort moments, sampled raw reference values, actual startup physics/configs and full combined trial/failure records are included. The primary human-readable per-trial CSVs contain 430,080 rows: clean/all plus combined/all/inside/outside; all additional scopes/boundaries remain losslessly in raw NPZs. All IDs and zero-exposure cells remain present. Model checkpoints are unchanged and already committed in the issue #12 bundle; verified references avoid duplicating them.","",
  "| Model | Existing full checkpoint | SHA256 |","|---|---|---|"]
 for name,e in references.items():lines.append(f"| {name} | [Checkpoint]({e['checkpoint']}) | {e['sha256']} |")
 lines += ["","All originals and generated artifacts remain on the remote data disk. No mounted files were copied locally; visual review uses in-memory SSH image chunks. Archived logs strip ANSI/trailing whitespace and SVG trailing whitespace is normalized without changing geometry. sha256.json covers every archive file except itself; worktree/index/commit objects and actual GitHub branch/tree are verified on publication.","",
  "GitHub results and task status are synchronized to [Notion](https://www.notion.so/403325d8231d8172995bf19510b8d760). Notion replaces Linear. No new training or acceptance/default promotion is authorized by this diagnostic outcome."]
 spaced=[]
 for line in lines:
  if line.startswith("|") and spaced and spaced[-1].strip() and not spaced[-1].startswith("|"):spaced.append("")
  spaced.append(line)
 (DEST/"README.md").write_text("\n".join(spaced)+"\n")
 for p in DEST.rglob("*.log"):
  text=re.sub(r"\x1b\[[0-9;?]*[A-Za-z]","",p.read_text(errors="replace"));p.write_text("\n".join(x.rstrip() for x in text.splitlines()).rstrip()+"\n")
 for p in DEST.rglob("*.svg"):p.write_text("\n".join(x.rstrip() for x in p.read_text().splitlines()).rstrip()+"\n")
 manifest={str(p.relative_to(DEST)):sha(p) for p in DEST.rglob("*") if p.is_file() and p.name!="sha256.json"};(DEST/"sha256.json").write_text(json.dumps(manifest,indent=2))
 print("ISSUE13_ARCHIVED",len(manifest),decision["decision"],flush=True)
if __name__=="__main__":main()
