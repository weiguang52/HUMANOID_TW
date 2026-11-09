"""Archive issue #8 single-observation results and frozen acceptance."""
import json,csv,hashlib,math,re,shutil
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from confidence_stats import wilson
ROOT=Path('/root/gpufree-data/tw_stage1/issue8_20261109')
DEST=Path('validation_artifacts/stage1_issue8_20261109')
REF=Path('validation_artifacts/stage1_issue7_20261109')
assert (ROOT/'all_training_evaluation_video_complete').exists()
assert json.loads((ROOT/'raw_trace_integrity_audit.json').read_text())['passed']
assert (ROOT/'video_quality_audit.json').exists() and (ROOT/'video_physics_manual_audit.json').exists()
configs=json.loads((ROOT/'config_difference_audit.json').read_text());assert len(configs)==2
tests=json.loads((ROOT/'action_observation_tests.json').read_text());assert tests['passed'] and tests['tests']==4
state=json.loads((ROOT/'training_state.json').read_text());assert len(state)==2
DEST.mkdir(exist_ok=True)
for p in ROOT.iterdir():
    if p.is_file() and (p.suffix in ['.json','.csv','.md'] or p.suffix=='.log' and p.name!='archive.log'):shutil.copy2(p,DEST/p.name)
for folder in ['traces','figures','videos']:shutil.copytree(ROOT/folder,DEST/folder,dirs_exist_ok=True)
shutil.copy2(ROOT/'issue8_video_contact_sheet.png',DEST/'issue8_video_contact_sheet.png')
shutil.copy2(REF/'joint_colors.json',DEST/'joint_colors.json')
baseline=DEST/'baseline_issue7';baseline.mkdir(exist_ok=True)
ref_manifest=json.loads((REF/'sha256.json').read_text())
for name in ['rates_wilson95.csv','pooled_rates_wilson95.csv','delay_trial_rates.csv','failure_events.csv','failure_mode_summary.json','input_models.json']:
    assert hashlib.sha256((REF/name).read_bytes()).hexdigest()==ref_manifest[name]
    shutil.copy2(REF/name,baseline/name)
identities=json.loads((REF/'input_models.json').read_text())
for name,value in identities.items():
    assert hashlib.sha256(Path(value['checkpoint']).read_bytes()).hexdigest()==value['sha256']
(DEST/'baseline_identity_audit.json').write_text(json.dumps({'source_commit':'9b6149dbfa57507ffa6ddfc1645697609c3c9472','baseline_data_and_checkpoint_hashes_verified':True,'models':identities},indent=2))
def finite(value):
    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all();return 1
    if isinstance(value,dict):return sum(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(finite(v) for v in value)
    return 0
audits={}
for name,entry in state.items():
    p=Path(entry['checkpoint']);model=torch.load(p,map_location='cpu',weights_only=False)
    old=torch.load(Path(f"validation_artifacts/stage1_issue6_20261109/models/bounded_strengthen_s{entry['seed']}/model_8995.pt"),map_location='cpu',weights_only=False)
    shapes=lambda x:{group+'/'+k:list(v.shape) for group in ['actor_state_dict','critic_state_dict'] for k,v in x[group].items() if isinstance(v,torch.Tensor)}
    assert shapes(model)==shapes(old)
    count=len(re.findall(r'Learning iteration \d+/\d+',(ROOT/f'{name}_train.log').read_text()))
    assert count==1500 and model['iter']==8995
    d=DEST/'models'/name;d.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,d/p.name);shutil.copytree(Path(entry['run_dir'])/'params',d/'params',dirs_exist_ok=True)
    for log in Path(entry['run_dir']).glob('events*'):shutil.copy2(log,d/log.name)
    a=EventAccumulator(entry['run_dir'],size_guidance={'scalars':0});a.Reload();metrics={}
    for tag in a.Tags()['scalars']:
        values=a.Scalars(tag);tail=values[-100:];assert all(math.isfinite(v.value) for v in values)
        metrics[tag]={'first_step':values[0].step,'last_step':values[-1].step,'last':values[-1].value,'last100_mean':sum(v.value for v in tail)/len(tail)}
    (d/'training_metrics.json').write_text(json.dumps(metrics,indent=2))
    audits[name]={'iteration':model['iter'],'logged_iterations':count,'finite_tensor_count':finite(model),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'parent_sha256':entry['parent_sha256'],'architecture_tensor_shapes_equal_issue6':True,'tensor_shapes':shapes(model)}
(DEST/'checkpoint_audit.json').write_text(json.dumps(audits,indent=2))
def csvrows(path):return list(csv.DictReader(path.open()))
pooled=csvrows(ROOT/'pooled_rates_wilson95.csv');perseed=csvrows(ROOT/'rates_wilson95.csv');delays=csvrows(ROOT/'delay_trial_rates.csv');events=csvrows(ROOT/'failure_events.csv')
old_pooled=csvrows(REF/'pooled_rates_wilson95.csv');old_events=csvrows(REF/'failure_events.csv')
frozen=json.loads((ROOT/'frozen_baseline_delay_gaps.json').read_text())
commands=['forward_03','forward_05','forward_08']
def speedrow(rows,model,command):return next(x for x in rows if x['model']==model and x['command']==command)
acceptance={};delay_aggregates={}
for seed in [1,2]:
    model=f'applied_s{seed}';d={}
    for delay in [0,20]:
        subset=[x for x in delays if x['model']==model and int(x['delay_ms'])==delay]
        k=sum(int(x['falls']) for x in subset);n=sum(int(x['trials']) for x in subset);ci=wilson(k,n)
        assert n==frozen[str(seed)][str(delay)]['trials']
        d[str(delay)]={'falls':k,'trials':n,'rate':k/n,'wilson95':ci}
    signed=d['20']['rate']-d['0']['rate'];gap=abs(signed);base_gap=frozen[str(seed)]['absolute_gap']
    upper={c:float(speedrow(pooled,model,c)['wilson95_high']) for c in commands[:2]}
    pass_ci=all(x<.01 for x in upper.values());pass_gap=gap<base_gap
    nonworse=all(float(speedrow(pooled,model,c)['rate'])<=float(speedrow(old_pooled,f'new_bounded_s{seed}',c)['rate']) for c in commands[:2])
    acceptance[model]={'pooled_low_mid_upper95':upper,'ci_upper_below1pct':pass_ci,'signed_delay_gap':signed,'absolute_delay_gap':gap,'baseline_absolute_delay_gap':base_gap,'delay_gap_shrunk':pass_gap,'low_mid_not_worse_than_issue6':nonworse,'all_criteria_pass':pass_ci and pass_gap}
    delay_aggregates[model]=d
both=all(x['all_criteria_pass'] for x in acceptance.values())
if both:recommendation='continue independent full-pipeline seed/generalization validation'
elif all(x['delay_gap_shrunk'] and x['low_mid_not_worse_than_issue6'] for x in acceptance.values()):recommendation='continue the observation hypothesis only in a separately authorized controlled study; gates remain unmet'
else:recommendation='stop treating this variant as a demonstrated fix; change the hypothesis rather than extending this budget ad hoc'
(DEST/'acceptance.json').write_text(json.dumps({'per_model':acceptance,'both_training_seeds_pass':both,'defaults_promoted':False,'frozen_recommendation':recommendation},indent=2))
(DEST/'delay_aggregates.json').write_text(json.dumps(delay_aggregates,indent=2))
for job in ROOT.glob('*_eval[123].json'):
    d=json.loads(job.read_text());assert d['last_action_observation']=='applied' and d['policy_last_action_params']=={'action_name':'JointPositionAction'} and d['critic_last_action_params']=={}
    assert d['observation_dims']==tests['observation_dims']
    c=d['perturbation_config'];assert c['mass_operation']=='scale' and c['mass_distribution_params']==[.5,1.5]
    assert c['push_velocity_range']=={'x':[-.3,.3],'y':[-.3,.3]} and c['action_class']=='RandomOneStepDelayAction'
    assert c['friction_static']==[.4,1.4] and c['friction_dynamic']==[.3,1.2]
    assert d['paired_resets'] and d['deterministic_policy']
    for x in d['results']:assert x['num_envs']==1024 and x['seconds']==20 and all(math.isfinite(v) for v in x['metrics'].values())
lines=['# Issue #8: applied-action policy observation, equal-budget two-seed comparison','','Issue: https://github.com/weiguang52/HUMANOID_TW/issues/8 . Base9b6149d. Production defaults remain requested-action observation.','','## Single intervention and validation','','Opt-in --last_action_observation applied adds only action_name=JointPositionAction to policy.last_action parameters. Isaac Lab then returns the action term raw_actions, which the existing delay class has selected from current/previous requested action. Critic last_action still returns environment-wide requested action. No extra observation dimension or network change; no reward/distribution changes. Saved configuration audits allow only this policy parameter and run/log names relative to matching issue6 models. Checkpoint tensor shapes match issue6 exactly.','','Four real64-env ActionManager tests passed:0ms requested/applied equality;20ms observation equals the delay-selected action and actual target buffer; full/partial resets clear episode history; critic/config/dimensions unchanged.64-env2-update resume smoke passed.','','## Fixed training and evaluation','','Two seeds1/2,4096 environments,1500updates each, full optimizer resume from the same robust05 parent SHA256 c200e355d29fda60225761550ea80642bb222778ce50c00ae3deb3d95fa7a38d; no std reset or checkpoint selection. Training pushes+/-0.45m/s and bounded mass nominal*U(0.5,1.5), original friction/noise/random0/20ms delay. Both seeds share the same seed1 parent; not full-pipeline independent replications.','','Evaluation follows issue7:1024 environments,3eval seeds,0.3/0.5/0.8m/s,20s, deterministic policy, paired reset seed=eval_seed*1000+command_index. Combined evaluation pushes remain+/-0.3m/s. New6jobs/18command estimates/18432trials. Historical controls reuse issue7 same-protocol fixed-policy evaluations, not128-env estimates. No fresh requested-action control training is duplicated: matching issue6 runs already use the exact parent/budget/seeds and audited unchanged configuration. Issue6 is the primary single-change control; issue5 has a different training mass distribution and is contextual reference only.','','Wilson95% intervals are conditional binomial intervals for each fixed policy/task. Pooled3072-trial intervals do not measure training-seed uncertainty, guarantee cross-policy independence, or test paired differences. Larger batches are not exact historical128-env episode replays.','','## Side-by-side pooled rates and95% intervals','','Every cell has denominator3072 (three eval seeds); counts, rates and Wilson interval.','','| Training seed | Command | Issue5 strengthened, requested | Issue6 bounded-trained, requested | Issue8 applied observation |','|---|---|---|---|---|']
def cell(row):return f"{row['falls']}/3072; {float(row['rate']):.3%} [{float(row['wilson95_low']):.3%}, {float(row['wilson95_high']):.3%}]"
for seed in [1,2]:
    for command in commands:lines.append(f"| {seed} | {command} | {cell(speedrow(old_pooled,f'old_strengthen_s{seed}',command))} | {cell(speedrow(old_pooled,f'new_bounded_s{seed}',command))} | {cell(speedrow(pooled,f'applied_s{seed}',command))} |")
lines+=['','## Per-evaluation-seed new-model rates','','| Model | Eval seed | Command | Falls/1024 | Rate | Wilson95% interval |','|---|---:|---|---:|---:|---|']
for row in perseed:lines.append(f"| {row['model']} | {row['eval_seed']} | {row['command']} | {row['falls']} | {float(row['rate']):.3%} | [{float(row['wilson95_low']):.3%}, {float(row['wilson95_high']):.3%}] |")
lines+=['','## Delay0/20ms split and frozen gap criterion','','Each group pools the same balanced3speeds/3eval seeds within one training policy. The archived baseline gap was frozen before training. The gap criterion is a strict reduction of the absolute point-estimate20ms-minus0ms rate difference, not a significance claim; signed gaps and both rates expose shrinking by worsening0ms. Detailed stratified counts/CIs are in delay_trial_rates.csv.','','| Model | Delay ms | Falls/trials | Rate | Wilson95% interval |','|---|---:|---:|---:|---|']
for seed in [1,2]:
    for model,d in [(f'issue6_s{seed}',frozen[str(seed)]),(f'applied_s{seed}',delay_aggregates[f'applied_s{seed}'])]:
        for delay in ['0','20']:
            v=d[delay];ci=v.get('wilson95',wilson(v['falls'],v['trials']))
            lines.append(f"| {model} | {delay} | {v['falls']}/{v['trials']} | {v['rate']:.3%} | [{ci[0]:.3%}, {ci[1]:.3%}] |")
lines+=['','## Frozen acceptance and recommendation','','Per training seed: both pooled0.3/0.5m/s Wilson95% upper bounds strictly<1%, and absolute delay gap strictly smaller than matching issue6 baseline. Both training seeds must pass.']
for name,a in acceptance.items():
    lines+=['',f"- {name}: upper-bound criterion {'PASS' if a['ci_upper_below1pct'] else 'FAIL'};0.3upper={a['pooled_low_mid_upper95']['forward_03']:.3%},0.5upper={a['pooled_low_mid_upper95']['forward_05']:.3%}; delay gap {'SHRANK' if a['delay_gap_shrunk'] else 'DID NOT SHRINK'} ({100*a['baseline_absolute_delay_gap']:.3f}pp -> {100*a['absolute_delay_gap']:.3f}pp); signed new gap={100*a['signed_delay_gap']:.3f}pp; all criteria {'PASS' if a['all_criteria_pass'] else 'FAIL'}."]
lines+=['','Both training seeds: '+('PASS' if both else 'FAIL')+'. Defaults promoted: NO.','','Frozen-rule recommendation: **'+recommendation+'**. This is a recommendation only; no further training is launched. Evidence is the acceptance values above, including low/mid-speed non-worsening against matching issue6 control. The experiment budget is complete even when performance acceptance fails.','','## First-failure trajectories and recent-push sagittal share','','The unchanged issue7 pre-reset recorder/mode definitions are reused. Raw traces contain actual push delta-v/timing, delay, contacts/cycle phase, root/joints and soft/hard limit flags. First true termination per environment is counted once; at most1s available history, with early failures explicitly marked. The mode means latest push within1s plus abs(pitch)>abs(roll) at termination; it is descriptive, not a unique causal mechanism.','','| Seed | Policy | Recent-push sagittal cases / all first failures | Share |','|---|---|---:|']
for seed in [1,2]:
    for label,rows,model in [('issue5 requested',old_events,f'old_strengthen_s{seed}'),('issue6 requested',old_events,f'new_bounded_s{seed}'),('issue8 applied',events,f'applied_s{seed}')]:
        subset=[x for x in rows if x['model']==model];k=sum(x['mode']=='recent_push__sagittal' for x in subset)
        lines.append(f"| {seed} | {label} | {k}/{len(subset)} | {k/len(subset):.3%} |")
audit=json.loads((ROOT/'raw_trace_integrity_audit.json').read_text())
lines+=['',f"Raw audit: {audit['first_failure_windows']}first-failure windows; {audit['full_1s_windows']}full1s; minimum{audit['minimum_available_frames']}frames for early failures. Terminal pre-reset heights, push bounds, delay labels, finite states and checkpoint hashes validated. See failure_events.csv, failure_mode_summary.json, mode_by_policy_speed.csv and figures/ for all grouped diagnostics/evidence. Case shares are not causal relative risks.",'','## Colored worst-command front/side videos','','For each new model, choose highest summed challenge fall count over three eval seeds, ties0.3/0.5/0.8. Videos are20s one-environment seed1 diagnostics using the applied observation; they are not1024-env acceptance. Camera follows translation and heading. Exact precolored USD retains original physics attributes/non-material relationships; no runtime binding. Velocity commands have no HumanML3D sourceNPY. Joint legend: joint_colors.json.']
for p in sorted((DEST/'videos').glob('*.mp4')):lines.append(f'- [{p.stem}](videos/{p.name})')
assert len(list((DEST/'videos').glob('*.mp4')))==4
lines+=['','Video encoding/frame/framing/color checks and manual small-part review are recorded in the audit files.','','## Tracking and artifact integrity','','All artifacts were generated remotely; raw originals remain on the data disk. Archived logs strip ANSI decoration/trailing whitespace. No mounted files are copied to local storage. Notion replaces Linear: https://www.notion.so/403325d8231d8172995bf19510b8d760 . Issue reply states measured gates and whether to continue/stop/change the hypothesis.']
(DEST/'README.md').write_text('\n'.join(lines)+'\n')
for log in DEST.rglob('*.log'):
    text=re.sub(r'\x1b\[[0-9;?]*[A-Za-z]','',log.read_text(errors='replace'))
    log.write_text('\n'.join(x.rstrip() for x in text.splitlines()).rstrip()+'\n')
manifest={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file() and p.name!='sha256.json'}
(DEST/'sha256.json').write_text(json.dumps(manifest,indent=2))
print('ARCHIVED',len(manifest),'files;',json.dumps(acceptance),'RECOMMEND',recommendation,flush=True)
