"""Archive complete reviewed issue12 results; all files remain on the remote disk."""
import csv,json,re,shutil
from pathlib import Path
from issue12_common import ROOT,REPO,COMMANDS,sha
DEST=REPO/"validation_artifacts/stage1_issue12_20261110"
BASE=REPO/"validation_artifacts/stage1_issue10_20261109"
def load(n):return json.loads((ROOT/n).read_text())
def pct(x):return "NA" if x is None else f"{100*x:.4f}%"
def num(x):return "NA" if x is None else f"{x:.4f}"
def ci(x,percent=False):
    if x is None:return "NA"
    f=pct if percent else num
    return "["+f(x[0])+","+f(x[1])+"]"
def dci(x):return "["+f"{100*x[0]:+.4f},{100*x[1]:+.4f}"+"]pp"
def csvrows(n):return list(csv.DictReader((ROOT/n).open()))

def main():
    assert load("delivery_integrity_audit.json")["passed"] and load("video_physics_manual_audit.json")["passed"]
    assert not DEST.exists(),"Inspect/verify an existing archive rather than blindly overwriting it"
    DEST.mkdir(parents=True)
    excluded={"progress.json","formal_driver_process.json","postprocess_driver.json","calibration_driver.json",
              "formal_training_state.json","push_receipt.json","issue_reply_receipt.json",
              "issue_reply_body.md","completion_requirement_audit.json"}
    for p in ROOT.iterdir():
        if p.is_file() and p.name not in excluded and p.suffix in [".json",".csv",".md",".log"]:
            if not p.name.startswith(("git_api_","archive")):shutil.copy2(p,DEST/p.name)
    for folder in ["traces","records","calibration","smoke","training","frozen_source_snapshots",
                   "figures","videos","video_contact_sheets","video_review_frames"]:
        shutil.copytree(ROOT/folder,DEST/folder,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    models=load("final_models.json");parents=load("input_models.json")
    old=json.loads((BASE/"sha256.json").read_text());refs={}
    for name,m in parents.items():
        rel=f"models/{name}/{Path(m['checkpoint']).name}"
        assert sha(BASE/rel)==old[rel]==m["sha256"]
        refs[name]={"sha256":m["sha256"],"relative_link":"../"+BASE.name+"/"+rel}
    (DEST/"parent_checkpoint_references.json").write_text(json.dumps(refs,indent=2))
    for name,m in models.items():
        folder=DEST/"models"/name;folder.mkdir(parents=True)
        target=folder/Path(m["checkpoint"]).name;shutil.copy2(m["checkpoint"],target);assert sha(target)==m["sha256"]
        for kind in ["env","agent"]:
            out=DEST/"run_configs"/name/f"{kind}.yaml";out.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(Path(m["run_dir"])/"params"/f"{kind}.yaml",out)
    for seed,item in load("calibration_state.json").items():
        s=json.loads((Path(item["audit_dir"])/"run_summary.json").read_text())
        out=DEST/"calibration_configs"/f"seed{seed}";out.mkdir(parents=True)
        for kind in ["env","agent"]:shutil.copy2(Path(s["log_dir"])/"params"/f"{kind}.yaml",out/f"{kind}.yaml")
    assert sha(BASE/"joint_colors.json")==old["joint_colors.json"];shutil.copy2(BASE/"joint_colors.json",DEST/"joint_colors.json")
    a=load("acceptance.json");g=load("group_results.json");e=load("paired_effects.json")
    tr=load("tracking_comparison.json");w=load("frozen_weight.json")
    q4=load("q4_forward_comparison.json");lag=load("delay_effects.json");change=load("paired_delay_rr_changes.json")
    lines=["# Issue12: post-push pitch recovery, five matched training-seed pairs","",
      "Issue: https://github.com/weiguang52/HUMANOID_TW/issues/12 . Base67d9879.",
      "**One opt-in reward; production defaults remain off/unchanged.**","",
      "## Result and frozen decision","",
      "**"+a["decision"]+"**. All primary/xy tracking gates passed: **"+str(a["all_primary_tracking_gates_passed"])+"**.",
      "Low-speed treatment minus control = "+f"{100*e['forward_03']['mean_difference']:+.4f}pp"+
      ", paired five-training-seed95% "+dci(e["forward_03"]["bootstrap95"])+". "
      "Treatment low-speed mean95%="+ci(g["treatment:forward_03"]["seed_summary"]["bootstrap95"],True)+".",
      "At 0.8 m/s, xy error changes from "+num(tr["xy"]["forward_08"]["control"]["mean"])+" to "+num(tr["xy"]["forward_08"]["treatment"]["mean"])+" m/s ("+f"{100*(tr['xy']['forward_08']['ratio_of_seed_means']-1):+.2f}%"+"); the frozen 10% tracking gate "+("passes." if tr["xy"]["forward_08"]["not_worse_than_10percent"] else "fails. The fall-rate improvement therefore does not establish an accepted controller."),
      "No weight scan, extra budget, checkpoint choice or default promotion. An ineffective/tracking-degraded scheme stops here.","",
      "| Frozen main/tracking criterion | Passed |","|---|---|"]
    for k,v in a["frozen_gates"].items():lines.append(f"| {k} | {v} |")
    lines+=["","Secondary results are descriptive and are not retroactively added as significance gates.","",
      "## Fixed cohort and fall rates","",
      "Five own requested4500 parents -> control0 / treatmentw, each+1500updates,4096envs. "
      "Ten finals represent6000requested-fork updates, not the complete upstream lifetime. Parent model_11994.pt -> final model_13493.pt; actual new updates, not filenames, determine budget. "
      "All actor/critic/optimizer state is resumed, no std reset. See training_audits.json and matched_pair_config_rng_audit.json.",
      "All10final models x3evaluation seeds x3commands x1024initial trials x20s = **92160initial trials**,30jobs/90command records. "
      "Deterministic policy, paired resets, bounded combined #7 protocol.","",
      "| Speed | Control falls/N | Control mean ± seedSD | Control mean95% | Treatment falls/N | Treatment mean ± seedSD | Treatment mean95% | Paired T-C95% |",
      "|---|---:|---:|---|---:|---:|---|---|"]
    for cmd in COMMANDS:
        c=g["control:"+cmd];t=g["treatment:"+cmd];cs=c["seed_summary"];ts=t["seed_summary"]
        lines.append(f"| {cmd} | {c['pooled_descriptive']['falls']}/15360 | {pct(cs['mean'])} ± {pct(cs['sample_sd'])} | {ci(cs['bootstrap95'],True)} | {t['pooled_descriptive']['falls']}/15360 | {pct(ts['mean'])} ± {pct(ts['sample_sd'])} | {ci(ts['bootstrap95'],True)} | {dci(e[cmd]['bootstrap95'])} |")
    lines+=["","![Fall rates](figures/fall_rates.png)","","![Paired differences](figures/paired_effects.png)","",
      "[All90evaluation rows](evaluation_rates.csv) | [All30policy/command rows](policy_rates.csv) | [Group summaries](group_results.json) | [Five-pair differences](paired_effects.json)","",
      "Each policy/command pools3072initial trials. FIVE paired training seeds are the inference units; all5^5=3125joint index resamples give percentile95% intervals. "
      "SampleSD uses ddof1. Shared upstream parent and fixed evaluation streams limit generalization; only3072unique startup physical draws are reused across models, not92160independent settings. "
      "Fall-rate bootstrap sums and strict0/1% endpoints use exact integer percentile numerators, avoiding float subtraction falsely classifying an exact0 as negative. Five-seed tails are uncertain. Secondary intervals are not simultaneous95%; zero observed rates do not prove zero population risk.","",
      "## Velocity tracking","",
      "| Speed | Control xy(m/s) | Treatment xy(m/s) | xy T/C | xy ratio95% | <=1.10 | Control yaw(rad/s) | Treatment yaw(rad/s) |",
      "|---|---:|---:|---:|---|---|---:|---:|"]
    for cmd in COMMANDS:
        x=tr["xy"][cmd];y=tr["yaw"][cmd]
        lines.append(f"| {cmd} | {num(x['control']['mean'])} | {num(x['treatment']['mean'])} | {num(x['ratio_of_seed_means'])} | {ci(x['ratio_bootstrap95'])} | {x['not_worse_than_10percent']} | {num(y['control']['mean'])} | {num(y['treatment']['mean'])} |")
    lines+=["","![Tracking](figures/tracking_errors.png)","",
      "The #7 metrics are non-terminal-frame means, including later ordinary auto-reset episodes; not first-initial-trial-only velocity samples. "
      "Each policy/command averages its three evaluation summaries; group means equally weight five training seeds. Full xy/yaw means, paired changes and intervals are in tracking_comparison.json. Only xy has the frozen10% gate.","",
      "## One reward and prospective weight calculation","",
      "Penalty rate **-w*(pitch^2+0.1*pitch_rate^2)** for1.0s after each original push, zero otherwise. "
      "ZYX Euler pitch; exact pitch_rate=cos(roll)*body_omega_y-sin(roll)*body_omega_z. No COM, direction, velocity or support gate is added. "
      "Privileged timing is reward-only; policy101/critic134 dimensions and networks remain unchanged.",
      "Reward is computed before reset and interval pushes. A push at global counterk affects the next50actual rewards k+1..k+50 atdt=.02; reset clears only affected IDs. "
      "The original push sampler/physics callback is called exactly once. Default None changes no config/events; explicit0/w study arms share pure clock/reset instrumentation.",
      "Calibration: five fixed parents, original training distribution,4096envs x1024steps=20.48s each, stochastic frozen actor and randomized initial episode lengths. "
      "No optimizer, parameter or normalizer updates. R is old positive net reward/step; Q is gated raw cost including window-outside zeros.",
      "R="+f"{w['old_mean_reward_per_control_step']:.12f}"+", Q="+f"{w['raw_gated_cost_mean']:.12f}"+
      ". Fix **w=.05*R/(.02*Q)="+repr(w["weight"])+"**, before formal training. "
      "Expected new contribution="+f"{w['expected_new_mean_reward_per_step']:.12f}"+
      "/step:5% pooled, subject to float32 arithmetic. This is not exactly5% for every seed or every later iteration.","",
      "| Calibration seed | Old reward/step | Gated raw cost | Active fraction | Implied new/old magnitude |",
      "|---:|---:|---:|---:|---:|"]
    for seed,item in w["source_state"].items():
        x=item["row"];fraction=w["weight"]*.02*x["raw_cost_mean"]/x["old_reward_per_step"]
        lines.append(f"| {seed} | {x['old_reward_per_step']:.9f} | {x['raw_cost_mean']:.9f} | {pct(x['active_fraction'])} | {pct(fraction)} |")
    lines+=["","[Frozen weight/input rows](frozen_weight.json) | [Protocol](protocol_plan.md) | [Protocol SHA](protocol_plan_sha256.json) | [Weight SHA](frozen_weight_sha256.json)","",
      "## Actual training contribution","",
      "Every PPO update records mean PER ENV/CONTROL-STEP over4096x24samples:1500rows/model. "
      "Manager dt weighting is included. Ratios divide mean contributions, avoiding unstable per-sample denominators.",
      "| Model | New reward/step | Old net/step | abs(new)/abs(old) | abs(new)/abs(total incl. new) | abs(new)/(xy+yaw) | Update ratio p95 | Update ratio max | Updates >tracking sum |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in csvrows("reward_monitor_summary.csv"):
        def f(k):return None if x[k]=="" else float(x[k])
        lines.append(f"| {x['model']} | {num(f('new_reward_per_step'))} | {num(f('old_reward_per_step'))} | {pct(f('overall_abs_new_over_abs_old'))} | {pct(f('overall_abs_new_over_abs_total'))} | {pct(f('overall_abs_new_over_tracking'))} | {pct(f('iteration_ratio_p95'))} | {pct(f('iteration_ratio_max'))} | {x['iterations_new_abs_exceeds_tracking_sum']} |")
    lines+=["","![Monitoring](figures/reward_contributions.png)","",
      "Ratios to old net reward and total net reward including the new term are both explicit; near-zero net reward can make a ratio large. Per-update actual-total ratios and p95/max/missing counts are retained in reward_monitor_timeseries.csv and reward_monitor_summary.csv. "
      "Display smoothing is50updates only; summaries/max/exceedances use unsmoothed rows. Claims about recorded update averages do not assert that every instantaneous sample is small. "
      "All1500rows/model and combined reward_monitor_timeseries.csv remain available. Contribution drift is reported; w is never rescaled.","",
      "## Same #11 strata: Q4-forward and latency","",
      "Exact #11 amplitude/mass/static/dynamic cuts are SHA-verified, never re-estimated from outcomes. All survivors stay in the initial-trial denominators. "
      "Push risks use at-risk full administrative1s windows: post-first-failure pushes are retained but excluded, after19s tail windows excluded, overlap censoring checked.",
      "| Speed | Control Q4-forward events/windows | Treatment events/windows | Control exposed trials | Treatment exposed trials | Paired event-count change95% |",
      "|---|---:|---:|---:|---:|---|"]
    for cmd in COMMANDS:
        d=q4[cmd];c=d["control"];t=d["treatment"];cr=c["pooled_window_risk"];tt=t["pooled_window_risk"]
        lines.append(f"| {cmd} | {cr['falls']}/{cr['n']} | {tt['falls']}/{tt['n']} | {c['exposed_initial_trials']}/15360 | {t['exposed_initial_trials']}/15360 | {ci(d['paired_event_count_change']['bootstrap95'])} failures/seed |")
    lines+=["","![Q4-forward](figures/q4_forward_risk.png)","",
      "| Speed | Control20/0ms RR95% | Treatment20/0ms RR95% | Paired T-C RR change95% |","|---|---|---|---|"]
    for cmd in COMMANDS:
        c=lag["control:"+cmd];t=lag["treatment:"+cmd];d=change[cmd]
        lines.append(f"| {cmd} | {num(c.get('relative_risk'))} {ci(c.get('relative_risk_bootstrap95'))} | {num(t.get('relative_risk'))} {ci(t.get('relative_risk_bootstrap95'))} | {num(d.get('treatment_minus_control_rr'))} {ci(d.get('paired_seed_bootstrap95'))} |")
    lines+=["","![Delay RR](figures/delay_relative_risk.png)","",
      "RR compares equal-seed20/0ms initial-trial risks. Zero-reference quantities/intervals are NA, never invented infinite ratios or pseudocounts. "
      "Q4 counts, exposed trials, total failures and window risks are distinct units. Wilson intervals assume Bernoulli independence and can be too narrow for repeated pushes/shared draws. "
      "Strata are unadjusted descriptions, not independently causal effects or proof of physical irrecoverability.",
      "[Trial strata](trial_stratified_risks.csv) | [Push strata](push_stratified_risks.csv) | [Joint physical strata](trial_joint_strata.csv) | [Five-seed Q4 rows](q4_forward_seed_windows.csv) | [Failure mechanisms](failure_mechanism_summary.csv)","",
      "## Twenty colored worst-command videos","",
      "**Independent illustrative ONE-ENV rollouts**, fixed seed1,20s/1000frames/50fps. Not the1024-env acceptance sample, exact original failures or synchronized multi-camera captures. "
      "Single-command videos use reset index0; the formal three-command protocol uses indices0/1/2. "
      "Worst command is highest pooled three-eval-seed fall rate; ties prefer0.8,0.5,0.3.",
      "Camera follows translation AND heading. Exact precolored experiment USD, no runtime material binding,1876original physics/non-material attributes preserved. "
      "Stage1 velocity commands have no HumanML3D sourceNPY(N/A). [Joint-to-color legend](joint_colors.json).","",
      "| Model | Worst command | Formal falls/3072 | Front | Side |","|---|---|---:|---|---|"]
    selection=load("worst_command_selection.json")
    for v in load("video_plan.json"):
        name=v["model"];front="videos/"+Path(v["views"]["front"]["output"]).name;side="videos/"+Path(v["views"]["side"]["output"]).name
        lines.append(f"| {name} | {v['command']} | {selection[name]['falls']}/3072 | [Front]({front}) | [Side]({side}) |")
    lines+=["","All20000frames encoding/count checked;100frames sampled for color/full-body/framing. Ten contact sheets, selected full-resolution motion frames and six figures are manually reviewed. "
      "Attachment review is sampled, not an assertion about every pixel in every frame. See video_quality_audit.json and video_physics_manual_audit.json. "
      "The render-only adapter changes static first-frame warmup12->60 and labels; the frozen formal evaluator is unchanged.","",
      "## Validation and transparent source ledger","",
      "Before training:10timer/config/monitor tests, five successful no-update calibrations, three64-env runtime arms. After diagnostic precision regression:11feature tests;8statistics tests. "
      "Original-vs-zero step rewards and complete initial/final root/joint/RNG arrays were bitwise equal. All parent actor/critic/optimizer states match exactly, including std. "
      "All five real4096-env pair starts are also state/RNG audited. Config comparisons permit only reward weight plus run metadata between arms.",
      "All scientific settings/weight and initial29source/dependency hashes were frozen first. During the first ZERO-weight control, a diagnostic-only precision fix was added: CUDA division/multiplication bydt can yield harmless absolute roundoff>1e-6 at large correct penalties. "
      "The monitoring residual is now max(abs(reported-expected)/(1+abs(expected)))<=1e-6. Only that diagnostic expression and its test changed; reward/result/clock/policy/RNG/weight and the other27hashes did not. "
      "The zero-weight first control's old/new residual is identically0. Initial manifest and exact old source snapshots remain; latest metadata explicitly discloses this after-start audit amendment.",
      "The first seed1calibration hit a monitor-clear inference-tensor I/O error after its no-update sampling. It was repaired before w freeze/formal training. Retry saved numerical fields are identical; not a performance retry, optimizer budget increase or weight search. Evidence retained.",
      "[Requirement audit](delivery_integrity_audit.json) | [Runtime smoke](runtime_smoke_audit.json) | [Resume/timing sources](runtime_resume_timing_semantics.json) | [Initial hashes](frozen_training_sources_initial.json) | [Latest hashes](frozen_training_sources.json) | [Precision addendum](monitor_precision_addendum.json)","",
      "## Complete artifacts and tracking","",
      "Ten full actor+critic+optimizer finals are in models/(model-name)/model_13493.pt. Five4500parents already exist in the issue10bundle, linked with verified SHA in parent_checkpoint_references.json. "
      "Per-run configs/load/RNG audits,1500-row monitors, all raw first-failure traces and full trial/push records are included. "
      "Human-readable CSVs are losslessly gzip-partitioned by group/command solely for manageable transport; partition_records_manifest.json gives all row counts and compressed/decompressed SHA. No denominator is dropped.",
      "All artifacts/original raw logs remain on the remote data disk. No mounted files were copied to local storage; image QA uses in-memory SSH chunks. "
      "Archived logs strip ANSI/trailing whitespace; archive SVG trailing whitespace is normalized without changing geometry. sha256.json covers every archive file except itself; staged/committed blobs and actual remote branch/tree are verified.",
      "Notion replaces Linear: https://www.notion.so/403325d8231d8172995bf19510b8d760 . Final commit/issue reply and completion status are synchronized there."]
    lines += ["", "## Final checkpoints and resume audits", "",
              "| Model | Final full checkpoint | Parent SHA prefix | Config | Load/state audit | Monitor |",
              "|---|---|---|---|---|---|"]
    for name,m in models.items():
        cp="models/"+name+"/"+Path(m["checkpoint"]).name
        config="run_configs/"+name+"/env.yaml"
        audit="training/"+name+"/loaded_state_audit.json"
        monitor="training/"+name+"/reward_contributions.csv"
        lines.append(f"| {name} | [Checkpoint]({cp}) | {m['parent_sha256'][:12]} | [Env]({config}) | [Audit]({audit}) | [CSV]({monitor}) |")
    spaced=[]
    for line in lines:
        if line.startswith("|") and spaced and spaced[-1].strip() and not spaced[-1].startswith("|"):spaced.append("")
        spaced.append(line)
    (DEST/"README.md").write_text("\n".join(spaced)+"\n")
    for p in DEST.rglob("*.log"):
        text=re.sub(r"\x1b\[[0-9;?]*[A-Za-z]","",p.read_text(errors="replace"))
        p.write_text("\n".join(x.rstrip() for x in text.splitlines()).rstrip()+"\n")
    for p in DEST.rglob("*.svg"):
        p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()).rstrip()+"\n")
    manifest={str(p.relative_to(DEST)):sha(p) for p in DEST.rglob("*") if p.is_file() and p.name!="sha256.json"}
    (DEST/"sha256.json").write_text(json.dumps(manifest,indent=2))
    print("ISSUE12_ARCHIVED",len(manifest),"FILES",a["decision"],flush=True)
if __name__=="__main__":main()
