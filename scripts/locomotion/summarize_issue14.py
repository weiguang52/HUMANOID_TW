"""Archive the reviewed issue14 study remotely, with exact old-model references."""
import csv,json,re,shutil
from pathlib import Path
from issue14_common import ROOT,REPO,BASE,WEIGHT,sha
from issue13_tracking import SCOPES
DEST=REPO/"validation_artifacts/stage1_issue14_20261111"
def load(n):return json.loads((ROOT/n).read_text())
def num(x):return "NA" if x is None else f"{float(x):.6g}"
def ci(x):return "["+", ".join(num(v) for v in x)+"]"
def main():
 assert load("delivery_integrity_audit.json")["passed"] and load("video_physics_manual_audit.json")["passed"]
 assert not DEST.exists(),"Inspect an existing archive; never blindly overwrite"
 DEST.mkdir(parents=True)
 exclude={"progress.json","formal_driver_process.json","postprocess_driver.json","formal_training_state.json","formal_job_state.json","push_receipt.json","issue_reply_receipt.json","notion_completion_receipt.json","completion_requirement_audit.json","issue_reply_body.md","issue_reply_body_final.md"}
 for p in ROOT.iterdir():
  if p.is_file() and p.name not in exclude and not p.name.startswith(("git_api_","archive","native_git","local_commit","staged_")) and p.suffix in [".json",".csv",".md",".log",".patch"]:
   shutil.copy2(p,DEST/p.name)
 for folder in ["smoke","eval_smoke","training","clean","combined","per_trial","figures","reconstruction","videos","video_review_frames","video_contact_sheets"]:
  shutil.copytree(ROOT/folder,DEST/folder,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
 new=load("final_models.json");history=load("historical_models.json");parents=load("input_parents.json");modelrefs={};parentrefs={}
 for name,m in new.items():
  path=DEST/"models"/name/Path(m["checkpoint"]).name;path.parent.mkdir(parents=True);shutil.copy2(m["checkpoint"],path);assert sha(path)==m["sha256"]
  shutil.copytree(Path(m["run_dir"])/"params",path.parent/"params")
  modelrefs[name]={"group":"clock","training_seed":m["training_seed"],"checkpoint":str(path.relative_to(DEST)),"sha256":m["sha256"],"additional_updates":1500}
 for name,m in history.items():
  old=REPO/"validation_artifacts/stage1_issue12_20261110/models"/name/Path(m["checkpoint"]).name;assert sha(old)==m["sha256"]
  modelrefs[name]={"group":m["group"],"training_seed":m["training_seed"],"checkpoint":"../stage1_issue12_20261110/models/"+name+"/"+old.name,"sha256":m["sha256"],"reused_unchanged":True}
 for name,m in parents.items():
  old=REPO/"validation_artifacts/stage1_issue10_20261109/models"/name/Path(m["checkpoint"]).name;assert sha(old)==m["sha256"]
  parentrefs[name]={"checkpoint":"../stage1_issue10_20261109/models/"+name+"/"+old.name,"sha256":m["sha256"],"iteration":m["iteration"]}
 (DEST/"model_references.json").write_text(json.dumps(modelrefs,indent=2));(DEST/"parent_model_references.json").write_text(json.dumps(parentrefs,indent=2));shutil.copy2(ROOT/"joint_colors.json",DEST/"joint_colors.json")
 oldarchive=REPO/"validation_artifacts/stage1_issue13_20261110";oldmanifest=json.loads((oldarchive/"sha256.json").read_text());refs={}
 for row in load("analysis_integrity_audit.json")["raw_command_audits"]:
  if row["group"]=="clock":continue
  for p,h in row["source_files_sha256"].items():
   rel=str(Path(p).relative_to("/root/gpufree-data/tw_stage1/issue13_20261110"));assert sha(oldarchive/rel)==oldmanifest[rel]==h
   refs["../stage1_issue13_20261110/"+rel]=h
 (DEST/"historical_artifact_references.json").write_text(json.dumps({"source_commit":BASE,"archive":"../stage1_issue13_20261110","manifest_sha256":sha(oldarchive/"sha256.json"),"all_reused_input_files_sha256":refs},indent=2))
 comp=load("paired_comparisons.json");falls=load("fall_results.json");decision=load("decision.json");groups=["control","treatment","clock"]
 names={"forward_03":"0.3m/s","forward_05":"0.5m/s","forward_08":"0.8m/s","yaw_positive":"yaw+0.3rad/s","yaw_negative":"yaw-0.3rad/s"}
 def get(pair,cond,cmd,part,metric):return comp[":".join([pair,cond,cmd,SCOPES[0],part,metric])]
 def stats(g,cond,cmd,part,metric):
  d=get("clock_vs_treatment" if g=="treatment" else "clock_vs_control",cond,cmd,part,metric);return d["treatment" if g=="clock" else "control"]
 primary=decision["primary"]["paired_difference"];status="supported" if decision["primary_mitigation_supported"] else "NOT supported"
 lines=["# Issue #14: critic-only push-recovery clock","",
 "Issue: https://github.com/weiguang52/HUMANOID_TW/issues/14 . Base: "+BASE+".","",
 "**Five fixed-budget training runs and both evaluation protocols completed. No extra formal updates, weight/encoding search, std reset or default promotion.**","",
 "## Frozen result","",
 f"Primary mitigation is **{status}**. Clean0.8 xy CLOCK minus old TREATMENT = {num(primary['mean'])}m/s, paired95% {ci(primary['bootstrap95'])}. The upper endpoint must be strictly below0.",
 "All complete acceptance gates pass: **"+str(decision["all_gates_pass"])+"**.",
 "The clock hypothesis stops after this study because its primary criterion was not supported." if decision["stop_clock_hypothesis"] else "Primary mitigation is supported; full acceptance still requires every separate frozen criterion below. This does not establish broad deployment validity.",
 "The previous issue12 rejection and issue13 diagnosis are historical results, not rewritten by this comparison. Production defaults remain unchanged.","",
 "| Frozen criterion | Passed |","|---|---|"]
 for name,value in decision["gates"].items():lines.append(f"| {name} | {value} |")
 lines+=["","![Primary and full gates](figures/primary_and_acceptance.png)","",
 "Inference unit: FIVE matched training seeds. Within each training seed average the three evaluation-seed summaries, then weight all five seeds equally. All5^5=3,125 paired joint bootstrap index samples give percentile95% intervals; **multiple comparisons are unadjusted**. The environment count is not an independent training-replication count. Shared ancestry and reused fixed reset streams limit generalization. Point tracking bounds are the specified<=5% group-mean bounds; no confidence non-inferiority criterion is substituted. The clean0.8 vx requirement passes only if the clock-minus-control95% interval is not entirely negative.","",
 "[Prospective protocol](protocol_plan.md) | [Frozen decision](decision.json) | [All paired intervals](paired_metric_comparisons.csv) | [Per-seed statistics](paired_comparisons.json)","",
 "## The sole intervention and preserved full-state resume","",
 "Only the optional critic recovery clock differs: active0/1 and normalized remaining time, critic134->136. Actor/policy101 and hidden512/256/128 layers remain unchanged. The existing post-push pitch reward weight is exactly"+str(WEIGHT)+"; all original reward, physical, push/noise/delay distributions and old observation terms remain unchanged.","",
 "The feature uses the SAME original reward push_window and push_clock. At dt0.02 it is active at ages0..50 inclusive; remaining=clamp(1-age/50,0,1), zero outside. At age50 flag=1,remaining=0; at age51 both zero. Reset clears the original shared timer. Isaac ordering is reward -> reset -> interval push -> observation: a newly marked push appears in the next observation at age0 and first affects the next reward at age1. This preserves existing reward semantics rather than substituting issue13's analysis boundary.","",
 "Construct the original134-wide critic/buffers, strictly restore each original4500 parent actor/critic/full Adam/std/iteration, audit equality, THEN append two zero weight columns and two zero Adam moment columns. Old optimizer steps/parameter groups and every old tensor element are preserved. No new random initialization; same-state actor output is bitwise equal before/after migration. [Source equivalence](historical_source_equivalence_audit.json) | [Default-off AST proof](default_off_source_audit.json) | [Runtime smoke proof](runtime_smoke_audit.json)","",
 "## Fixed training budget and monitoring","",
 "Exactly the same five requested4500 parents used by issue12, seeds1..5,4096environments,1500additional PPO updates each (7500total). Full actor/critic/Adam/std restored, no std reset, no intermediate checkpoint choice. The final checkpoint iteration equals the original parent iteration+1499, following the runner's existing inclusive logging convention. The original control/treatment groups and their exact issue13 evaluations are reused, not retrained.","",
 "Preflight:10unit tests; two-update legacy vs switch-off PPO states/std exactly equal, reward CSV bytes equal; clock-on two-update smoke;1000-step clock-on deterministic parent rollout exactly equal to archived original treatment rewards/costs/root/joint/all CPU/CUDA/Python/NumPy RNG. Six smoke updates at64envs are separate from7500formal updates, and no smoke checkpoint is used as a formal candidate/parent. Two zero-update64-env evaluation adapter smokes validate clean/combined136-wide loading. Early preflight dtype/import/signature issues were corrected before the final gates/source freeze.","",
 "| Model | Parent | Additional updates | Final checkpoint | Initial critic expansion | Training audit |","|---|---|---|---|---|"]
 for name,m in new.items():
  seed=m["training_seed"];parent=parentrefs[f"requested_s{seed}_u4500"];ref=modelrefs[name]
  lines.append(f"| {name} | [requested4500]({parent['checkpoint']}) |1500| [Full model/optimizer]({ref['checkpoint']}) |134->136,zero new weights/moments| [Audit](training/{name}/critic_clock_migration.json) |")
 lines+=["","Every formal run's env/agent YAML was compared with its archived issue12 TREATMENT: only the new critic term and log/run metadata differ. Its initial4096-env physical state and complete RNG match that treatment bitwise. All1500iterations and Adam step increments are audited; original reward contribution/identity monitoring is retained. [Training audits](training_audits.json). Per-model params, load/expansion audit, initial states and all1500reward-monitor rows are included.","",
 "## Evaluation and clean-condition fidelity","",
 "New CLOCK group:30formal jobs=120commands=122,880initial trials (76,800clean +46,080combined). Old control/treatment:60jobs=240commands=245,760initial trials reused exactly from the committed issue13 bundle. Combined three-group summaries cover368,640initial trials without pretending reused trials are new replications.","",
 "Each final model:1024environments x3evaluation seeds x20s. Clean commands retain original order forward0.3,yaw+0.3,yaw-0.3,forward0.5,forward0.8; combined0.3/0.5/0.8. Deterministic actor and original paired reset seed=eval_seed*1000+command_index. Different protocols do not claim identical physical draws. Actual clean masses are nominal, robot/terrain static+dynamic friction1.0, no pushing/noise/material/mass randomization,ordinary zero-delay JointPositionAction. Policy/critic dims are101/136 only for CLOCK,101/134 for historical groups; inference never changes model/Adam state.","",
 "| Clean command | Metric | Control mean | Treatment mean | Clock mean | Clock-control difference95% | Clock-treatment difference95% |","|---|---|---|---|---|---|---|"]
 for cmd in ["forward_03","yaw_positive","yaw_negative","forward_05","forward_08"]:
  for metric in ["xy_error","yaw_error"]:
   values=[stats(g,"clean",cmd,"all",metric)["mean"] for g in groups];c=get("clock_vs_control","clean",cmd,"all",metric)["paired_difference"];t=get("clock_vs_treatment","clean",cmd,"all",metric)["paired_difference"]
   lines.append(f"| {names[cmd]} |{metric}|"+ "|".join(num(v) for v in values)+f"|{num(c['mean'])} {ci(c['bootstrap95'])}|{num(t['mean'])} {ci(t['bootstrap95'])}|")
 lines+=["","xy units m/s; yaw units rad/s. All group95%/seed SD and all pairwise differences are in the complete tables.","","![Clean tracking](figures/clean_tracking.png)","",
 "## First-fall rates, with all three groups","",
 "| Condition | Command | Group | Falls/N | Seed mean95% (%) |","|---|---|---|---|---|"]
 for key,v in falls.items():
  cond,cmd=key.split(":")
  for g in groups:
   f=v["groups"][g];s=f["seed_summary"]
   lines.append(f"|{cond}|{names[cmd]}|{g}|{f['pooled_falls']}/{f['pooled_n']} ({100*s['mean']:.4f}%)|{ci([100*x for x in s['bootstrap95']])}|")
 lines+=["","[All paired fall differences](fall_results.json). Every seed pools3072initial trials per command. Zero observed falls do not establish zero population risk. Fall reduction cannot substitute for failed tracking/speed gates.","",
 "## Inside/outside push windows and bias versus fluctuation","",
 "The issue13 analysis definition is preserved: full20s nonterminal POST-STEP samples, including ordinary episodes after failure resets. Main push window is[push,push+1s),ages0..49; a separate ages1..50 reward-aligned boundary and initial-episode-only scope remain in raw moment/stat tables. These descriptive partitions do not change the training clock. Per-trial count/sum/sumsq, packed masks, per-step moments and raw reference-environment samples are retained. Exposure-weighted inside+outside contributions reconstruct overall error; counts/exposures and equal-trial means remain available.","",
 "| Speed | Partition | Group | xy(m/s) | yaw(rad/s) | vx(m/s) | mean trial temporal SD vx(m/s) |","|---|---|---|---|---|---|---|"]
 for cmd in ["forward_03","forward_05","forward_08"]:
  for part in ["all","inside","outside"]:
   for g in groups:
    vals=[stats(g,"combined",cmd,part,k)["mean"] for k in ["xy_error","yaw_error","vx_mean","vx_temporal_sd_mean"]]
    lines.append(f"|{names[cmd]}|{part}|{g}|"+ "|".join(num(v) for v in vals)+"|")
 lines+=["","![Window tracking](figures/window_tracking.png)","",
 "Bias=mean body vx minus command. Temporal fluctuation is EACH trial's population SD about its own conditional mean(ddof0), then an equal-trial mean among trials with samples. Pooled SD and within/between-trial variance are separately retained and their identity audited; they are not interchangeable. Signed bias/velocity ratios are not interpreted. Undefined zero-exposure cells remain present, with no pseudocounts.","",
 "![Bias and fluctuation](figures/bias_and_variability.png)","",
 "## Secondary achieved-speed/fall comparison","",
 "| Condition | Command | Group | Measured vx95%(m/s) | Temporal SD vx | First-fall seed mean95%(%) |","|---|---|---|---|---|"]
 for x in csv.DictReader((ROOT/"actual_speed_fall_table.csv").open()):
  lines.append(f"|{x['condition']}|{names[x['command']]}|{x['group']}|{num(x['mean_actual_vx'])} {ci([float(x['actual_vx95_low']),float(x['actual_vx95_high'])])}|{num(x['mean_temporal_sd_vx'])}|{100*float(x['fall_seed_mean']):.4f} {ci([100*float(x['fall95_low']),100*float(x['fall95_high'])])}|")
 for context in load("actual_speed_fall_interpretation.json"):
  lines += ["", "At "+names[context["command"]]+", common achieved-speed-bin coverage (control/treatment/clock) = "+" / ".join(f"{100*context['fraction_trials_in_common_bins'][g]:.1f}%" for g in groups)+"."]
  for pair in ["clock_vs_control","clock_vs_treatment"]:
   x=context["comparisons"][pair]
   lines.append(pair+": "+x["speed_direction"]+"; "+x["fall_direction"]+". "+x["interpretation"])
 lines+=["","![Actual speed and falls](figures/actual_speed_vs_falls.png)","",
 "The table explicitly compares falls at the speeds each group ACTUALLY achieved; equal commanded speed is not claimed to be equal achieved speed. Secondary fixed0.05m/s bins over0..1 with under/overflow retain every trial and provide common-support counts/falls for all three groups. Five-seed paired bootstrap intervals are given ONLY where all five paired seed denominators are nonzero; unsupported cells remain NA. Open plot markers show pooled sparse-support descriptive rates, not invented seed-level estimates.","",
 "**Achieved speed is post-treatment and can itself be affected by falls/resets.** Conditioning on its full-horizon trial mean has survival/occupancy bias. These tables can expose remaining speed mismatch or lack of common support, but do not identify a causal speed-adjusted recovery effect. A lower fall rate with lower achieved speed remains compatible with slowing; matched bins alone cannot prove a recovery mechanism.","",
 "![Common achieved-speed bins](figures/actual_speed_common_bins.png)","",
 "[Per-trial achieved speeds/failures](actual_speed_trial_rows.csv) | [Common-bin counts/rates](actual_speed_fall_bins.csv) | [Supported paired bin intervals](actual_speed_bin_paired_comparisons.json). Initial-episode sensitivity is reported without reselecting the primary criterion: [Sensitivity](classification_sensitivity.json).","",
 "## Three true-median examples and six colored videos","",
 "Each group's0.8 combined example is nearest its group median xy error across all15360initial trials, no survival filter; ties trainingseed,evalseed,envid. Median examples illustrate, not determine acceptance. Historical selections/media are byte-identically reused; the new selected1024-env evaluation is reconstructed only to capture the same source states, with original results/arrays/moments exactly reproduced and no new formal denominator.","",
 "| Group | Model/eval seed/env | xy/group median | Actual push times(s) | Reset frames | Front | Side |","|---|---|---|---|---|---|---|"]
 reconstruction=load("reconstruction_state.json")
 for e in load("video_plan.json"):
  s=reconstruction[e["group"]];front="videos/"+Path(e["views"]["front"]["path"]).name;side="videos/"+Path(e["views"]["side"]["path"]).name
  lines.append(f"|{e['group']}|{s['model']}/{s['eval_seed']}/{s['env_id']}|{num(s['xy_error'])}/{num(s['group_median_xy'])}|{', '.join(f'{v:.2f}' for v in s['push_seconds']) or 'none'}|{s['reset_frames']}|[Front]({front})|[Side]({side})|")
 lines+=["","All clips20s,1000frames,50fps,960x720,H264. Exact recorded root pose/world velocity and joint positions are displayed with NO physics re-integration; display joint velocities are set to zero, so this is a pose replay rather than a new dynamics rollout; front/side share exact source frames within each group, while different groups are NOT synchronized. Numeric push times, red ticks/window shading and reset labels are shown. Video recovery labels/shading use the issue13 POST-EVENT ANALYSIS interval[push,push+1s),ages0..49; they are not a visualization of the critic feature/reward mask, whose existing inclusive age50 endpoint is separately preserved. Camera follows translation AND heading. Display uses the separately precolored exact experiment asset with1876original attributes preserved; no runtime material rebinding. Nominal display materials/masses do not re-integrate source randomization. [Joint-color legend](joint_colors.json). Stage1 velocity commands have no HumanML sourceNPY: **N/A**.","",
 "Six encodings and36sampled frames, three contact sheets, six full-resolution motion/near-push frames and six statistical figures were actually checked. Review is sampled attachment/color/framing, not exhaustive every-frame pixel validation. [Video audit](video_quality_audit.json) | [Actual manual observations](video_physics_manual_audit.json). Any initial cosmetic ground grain is described in that review, without changing source trajectories.","",
 "## Raw evidence and reproducibility","",
 "[Requirement audit](delivery_integrity_audit.json) | [Raw-data/moment audit](analysis_integrity_audit.json) | [Frozen43training dependencies](frozen_training_sources.json) | [Derived source/output binding](analysis_outputs_sha256.json) | [Per-trial645120row manifest](per_trial_csv_manifest.json) | [Representative selection](representative_selection.json) | [Old input file references](historical_artifact_references.json)","",
 "New raw clean/combined moments, original full-trial/failure records, startup physics/configs, all1500-update monitors and full final checkpoints/Adam states are archived here. Historical raw inputs and ten checkpoints remain in their original committed bundles with exact verified references; no duplicate retraining/re-evaluation is represented as new evidence. All645120human-readable primary per-trial rows are included, while both sampling scopes and both boundaries remain losslessly in raw moments. Derived analysis/plot/media/report code was written after the formal freeze and is bound by actual source/output SHA, not falsely described as prospectively frozen.","",
 "| Model | Full checkpoint | SHA256 |","|---|---|---|"]
 for name,m in modelrefs.items():lines.append(f"|{name}|[Checkpoint]({m['checkpoint']})|{m['sha256']}|")
 lines+=["","All originals/products stay on the remote data disk; no mounted files were copied locally. Logs strip ANSI/trailing whitespace and SVG whitespace is normalized without changing geometry. sha256.json covers every archive file except itself; expected worktree/index/commit blobs and actual GitHub branch/tree are checked at publication.","",
 "GitHub issue replies and status synchronize to [Notion](https://www.notion.so/403325d8231d8172995bf19510b8d760); Notion replaces Linear. This fixed-budget study does not authorize additional updates or tuning after a failed primary result."]
 (DEST/"README.md").write_text("\n".join(lines)+"\n")
 for p in DEST.rglob("*.log"):
  s=re.sub(r"\x1b\[[0-9;?]*[A-Za-z]","",p.read_text(errors="replace"));p.write_text("\n".join(x.rstrip() for x in s.splitlines()).rstrip()+"\n")
 for p in DEST.rglob("*.svg"):p.write_text("\n".join(x.rstrip() for x in p.read_text().splitlines()).rstrip()+"\n")
 manifest={str(p.relative_to(DEST)):sha(p) for p in DEST.rglob("*") if p.is_file() and p!=DEST/"sha256.json"};(DEST/"sha256.json").write_text(json.dumps(manifest,indent=2));print("ISSUE14_ARCHIVED",len(manifest),flush=True)
if __name__=="__main__":main()
