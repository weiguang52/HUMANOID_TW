"""Package issue #7 evidence, without training or default changes."""
import csv,json,hashlib,re,shutil,math,collections
from pathlib import Path
from confidence_stats import wilson
ROOT=Path('/root/gpufree-data/tw_stage1/issue7_20261109')
DEST=Path('validation_artifacts/stage1_issue7_20261109')
assert (ROOT/'high_resolution_complete').exists() and (ROOT/'replays_complete').exists()
assert (ROOT/'video_quality_audit.json').exists()
assert (ROOT/'next_step_proposal.md').exists()
assert json.loads((ROOT/'raw_trace_integrity_audit.json').read_text())['passed']
summary=json.loads((ROOT/'failure_mode_summary.json').read_text())
assert summary['total_trials']==36864 and summary['total_first_failures']==2825
DEST.mkdir(exist_ok=True)
for p in ROOT.iterdir():
    if p.is_file() and (p.suffix in ['.json','.csv','.md'] or p.suffix=='.log' and p.name!='archive.log') and not p.name.startswith('smoke_'):
        shutil.copy2(p,DEST/p.name)
check=DEST/'instrumentation_check';check.mkdir(exist_ok=True)
for name in ['smoke_plain.json','smoke_recorded.json','smoke_plain.log','smoke_recorded.log']:
    shutil.copy2(ROOT/name,check/name)
shutil.copytree(ROOT/'smoke_traces',check/'traces',dirs_exist_ok=True)
for folder in ['traces','figures','videos']:
    shutil.copytree(ROOT/folder,DEST/folder,dirs_exist_ok=True)
shutil.copy2(ROOT/'issue7_video_contact_sheet.png',DEST/'issue7_video_contact_sheet.png')
shutil.copy2('validation_artifacts/stage1_issue6_20261109/joint_colors.json',DEST/'joint_colors.json')
models=json.loads((ROOT/'input_models.json').read_text())
for name,value in models.items():
    p=Path(value['checkpoint']);assert hashlib.sha256(p.read_bytes()).hexdigest()==value['sha256']
    value['repository_path']=str(p.relative_to(Path.cwd()))
(DEST/'input_models.json').write_text(json.dumps(models,indent=2))
for p in ROOT.glob('*_eval[123].json'):
    d=json.loads(p.read_text())
    assert len(d['results'])==3 and d['paired_resets'] and d['deterministic_policy']
    c=d['perturbation_config'];assert c['mass_operation']=='scale' and c['mass_distribution_params']==[.5,1.5]
    assert c['push_velocity_range']=={'x':[-.3,.3],'y':[-.3,.3]}
    assert c['friction_static']==[.4,1.4] and c['friction_dynamic']==[.3,1.2]
    assert c['action_class']=='RandomOneStepDelayAction'
    for x in d['results']:assert x['num_envs']==1024 and x['seconds']==20 and all(math.isfinite(v) for v in x['metrics'].values())
def rows(name):return list(csv.DictReader((ROOT/name).open()))
rates=rows('rates_wilson95.csv');pooled=rows('pooled_rates_wilson95.csv');events=rows('failure_events.csv');delays=rows('delay_trial_rates.csv')
lines=['# Issue #7: high-resolution challenge evaluation and first-failure analysis','','Issue: https://github.com/weiguang52/HUMANOID_TW/issues/7 . Base36e38ee. User confirmed analysis only: no retraining and no default/observation/reward changes.','','## Protocol and uncertainty','','Four fixed checkpoints: issue6 bounded-strengthen training seeds1/2 and issue5 strengthened-push seeds1/2. Same bounded combined challenge for all: nominal mass*U(0.5,1.5), friction static[0.4,1.4]/dynamic[0.3,1.2], evaluation pushes +/-0.3m/s, observation noise, random per-episode0/20ms action delay. Training push +/-0.45m/s is not used as the evaluation push.','','Each policy uses1024 environments,3evaluation seeds,0.3/0.5/0.8m/s,20s per command, deterministic action selection and paired reset seed=eval_seed*1000+command_index.12jobs,36command estimates,36864trials. Larger batches use different random streams from the historical128-env runs; they improve resolution but are not exact historical episode replays.','','95% intervals use the binomial Wilson score formula, checked against SciPy at boundary/interior counts. One fall now changes the estimate by0.09765625percentage points. Per-seed intervals are conditional on a fixed policy/task; pooled3072-trial intervals do not capture between-training-seed uncertainty or guarantee independence of shared pseudo-random streams. They must not be read as a general deployment guarantee or a paired-policy difference test.','','## Pooled fall rates with95% intervals','','Each row combines the three evaluation seeds, denominator3072; all per-seed rows follow below and in rates_wilson95.csv.','','| Model | Speed command | Falls | Rate | Wilson95% interval |','|---|---|---:|---:|---|']
for row in pooled:
    lines.append(f"| {row['model']} | {row['command']} | {row['falls']}/{row['trials']} | {float(row['rate']):.3%} | [{float(row['wilson95_low']):.3%}, {float(row['wilson95_high']):.3%}] |")
lines+=['','No policy demonstrates a pooled upper95% bound below1% even at0.3m/s. At0.5/0.8m/s failures remain substantial. This analysis does not establish a robustness improvement from issue6 retraining.','','## Per-evaluation-seed rates','','| Model | Eval seed | Command | Falls/1024 | Rate | Wilson95% interval |','|---|---:|---|---:|---:|---|']
for row in rates:
    lines.append(f"| {row['model']} | {row['eval_seed']} | {row['command']} | {row['falls']} | {float(row['rate']):.3%} | [{float(row['wilson95_low']):.3%}, {float(row['wilson95_high']):.3%}] |")
lines+=['','## Recorded first failures','','All2825first true terminations have an up-to1s window ending at the terminal pre-reset state.2758have the complete51samples;67early terminations have only available episode history (39..50samples), explicitly marked by validity masks. No pre-episode history is fabricated.51samples at50Hz span1s. Termination-manager recording occurs after physics/termination computation and before auto-reset. All2825first failures were base_height terminations (threshold0.08m), not timeout or base_contact. The exact termination term list is recorded per environment. Subsequent auto-reset episodes do not add trials.','','RawNPZs and metadata under traces/ contain root pose/velocity, all joint positions, base height/roll/pitch, foot force/contact/cycle phase, hard/soft joint-limit proximity, current action delay, actual last push delta-v in world/body axes, elapsed push time/contact at push and commands. Push intervals are5..8s, so at most one push occurs in the1s window; older/no-push cases remain explicit. Cycle phase is onset-based and marked-1until a previous cycle is available. Hard/soft limit proximity uses0.02rad; separate touching proxy0.001rad and outside-range tolerance1e-5 are recomputed from raw positions. These flags are not direct constraint-force measurements. Relative trace paths accompany original remote paths in failure_events.csv.','','The64-env recorded/plain smoke had12falls in both and identical continuous metrics (maximum difference0). The push wrapper invokes the original sampler once, without extra RNG draws; measured push deltas were checked. Wilson tests passed. Checkpoint hashes were verified unchanged after evaluation.','','## Failure-mode proportions','','Frozen primary categories cross push recency (latest push within1s vs older/no push) with terminal Euler tilt dominance (sagittal if abs(pitch)>abs(roll), lateral otherwise; near-upright below5degrees). A mode is dominant only at>=50%overall and ranked first for each of4policies. This is a descriptive endpoint/temporal label, not a causal mechanism.','','| Mode | First failures | Share |','|---|---:|---:|']
for mode,value in summary['groups']['mode'].items():lines.append(f"| {mode} | {value['failures']} | {value['fraction_of_failures']:.3%} |")
lines+=['','Dominant descriptive mode: **'+str(summary['dominant_mode'])+'**. The recent-push sagittal mode is71.181%,85.746%,82.000%,91.679%within new_s1/new_s2/old_s1/old_s2 respectively. mode_by_policy_speed.csv exposes command-level heterogeneity.','','![Mode shares](figures/failure_mode_shares.png)','','## Grouped diagnostics (case shares, not causal risks)','','| Feature | Group | Failures | Share of all failures |','|---|---|---:|---:|']
for feature in ['push_direction_body','push_magnitude_bin','delay_ms','contact_at_push','contact_at_failure','hard_limit_any_last1s','soft_limit_any_last1s','hard_limit_touch_proxy_last1s','outside_hard_limits_last1s']:
    for group,value in summary['groups'][feature].items():lines.append(f"| {feature} | {group} | {value['failures']} | {value['fraction_of_failures']:.3%} |")
lines+=['','Push magnitude is horizontal world delta-v norm; bins are<0.15,0.15..0.30,>=0.30m/s. Direction is the dominant signed body-frame component of the latest push. Older pushes remain labeled as older; no-push trials are separate. Push-direction/magnitude/contact-at-push case tables lack nonfailure push-event denominators and must not be called relative risks. Flight at termination can be a consequence of collapse.','','Joint proximity is nondiscriminating here: every failure window has a near-limit joint, while99.994%of at-risk steps already have at least one soft-limit-near joint. Tiny outside-range tolerances/normal gait constraints do not establish that hard-stop impact caused the collapse.','','## Delay trial denominators and exposure','','The randomized episode delay is recorded before any failure/reset. The following pooled counts mix all speeds/evaluation seeds within each policy; detailed stratified rates/CIs are in delay_trial_rates.csv.','','| Model | Delay ms | Falls / at-risk trials | Rate | Wilson95% interval |','|---|---:|---:|---:|---|']
for model in sorted(models):
    for delay in [0,20]:
        subset=[x for x in delays if x['model']==model and int(x['delay_ms'])==delay]
        k=sum(int(x['falls']) for x in subset);n=sum(int(x['trials']) for x in subset);ci=wilson(k,n)
        lines.append(f"| {model} | {delay} | {k}/{n} | {k/n:.3%} | [{ci[0]:.3%}, {ci[1]:.3%}] |")
lines+=['','Across all four policies,0ms:948/18604=5.096%;20ms:1877/18260=10.279%. The direction agrees in all four policies. This is conditional descriptive evidence for investigating actuator-feedback timing, not proof of a unique delay cause. At-risk step exposures for delay/contact/current soft-limit proximity/recent push are saved in failure_mode_summary.json; multiplying by0.02gives environment-seconds. Case-window limit flags must not be directly compared to instantaneous exposure as a causal hazard.','','## Representative evidence and colored videos','','Representatives were selected by the frozen full-window/median-failure-time rule, not by apparent visual severity. Each video replays exact recorded root and joint states ending at the first pre-reset termination. It is a kinematic display with no physics integration or policy rerun;51frames/50fps cover51end-inclusive samples spanning1s (encoded duration1.02s). The overlay explicitly states recorded-state replay. No source HumanML3D NPY exists for these stage1 velocity commands.']
for rank in [1,2]:
    selection=json.loads((ROOT/f'representative_mode{rank}.json').read_text())
    lines+=['',f"### Mode{rank}: {selection['mode']}",'',f"Source: {selection['model']}, eval seed{selection['eval_seed']}, {selection['command']}, env{selection['env_id']}; failure at{selection['failure_second']:.2f}s.",'',f"![Trace](figures/mode{rank}_representative_trace.png)",'',f"[Colored front](videos/mode{rank}_front.mp4) / [colored side](videos/mode{rank}_side.mp4)."]
lines+=['','Exact precolored USD retains1876original attributes and non-material relationships; no runtime material binding. Cameras follow translation and heading. Joint legend: joint_colors.json. Video audit and manual review verify encoding/framing/colors/small-part attachment. Rendering does not create another evaluation result or validate a new dynamics trajectory.','','## Single next-step proposal; not implemented','','See [next_step_proposal.md](next_step_proposal.md). Change only the policy last_action observation parameter to action_name=JointPositionAction, exposing the previously applied delay-selected command rather than the environment-wide last requested action. Input dimension/architecture stay unchanged. Hold critic inputs/rewards/randomization/training budget fixed in a future controlled test. Current observation and default configurations remain unchanged.','','## Delivery and limits','','No training occurred in this issue; four input checkpoint hashes remain unchanged. All artifacts were generated remotely. Archived logs remove ANSI/trailing whitespace; raw originals remain remote. No mounted files were copied to local storage. Notion replaces Linear: https://www.notion.so/403325d8231d8172995bf19510b8d760 . The issue reply records analysis completion without claiming the low-speed performance target is achieved.']
(DEST/'README.md').write_text('\n'.join(lines)+'\n')
for log in DEST.rglob('*.log'):
    value=re.sub(r'\x1b\[[0-9;?]*[A-Za-z]','',log.read_text(errors='replace'))
    log.write_text('\n'.join(line.rstrip() for line in value.splitlines()).rstrip()+'\n')
manifest={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file() and p.name!='sha256.json'}
(DEST/'sha256.json').write_text(json.dumps(manifest,indent=2))
print('ARCHIVED',len(manifest),'files; raw first-failure windows',summary['total_first_failures'],flush=True)
