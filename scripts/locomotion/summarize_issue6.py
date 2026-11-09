"""Archive issue #6 mass audit, controlled comparisons and failed gates."""
import csv,hashlib,json,math,re,shutil,statistics
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ROOT=Path('/root/gpufree-data/tw_stage1/issue6_20261109')
DEST=Path('validation_artifacts/stage1_issue6_20261109')
PRIOR=Path('validation_artifacts/stage1_issue5_20261109')
assert (ROOT/'all_training_evaluation_video_complete').exists()
assert (ROOT/'video_quality_audit.json').exists()
assert '\nOK\n' in (ROOT/'mass_tests.log').read_text()
state=json.loads((ROOT/'training_state.json').read_text());assert len(state)==2
DEST.mkdir(exist_ok=True)
for p in ROOT.iterdir():
    if p.is_file() and (p.suffix=='.json' or p.name=='frozen_plan.md' or p.suffix=='.log' and p.name!='archive.log'):
        shutil.copy2(p,DEST/p.name)
commands=['forward_03','forward_05','forward_08']
def finite(value):
    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all();return 1
    if isinstance(value,dict):return sum(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(finite(v) for v in value)
    return 0
audit={}
for name,entry in state.items():
    checkpoint=Path(entry['checkpoint']);model=torch.load(checkpoint,map_location='cpu',weights_only=False)
    count=len(re.findall(r'Learning iteration \d+/\d+',(ROOT/f'{name}_train.log').read_text()))
    assert count==1500 and model['iter']==8995
    d=DEST/'models'/name;d.mkdir(parents=True,exist_ok=True)
    shutil.copy2(checkpoint,d/checkpoint.name)
    shutil.copytree(Path(entry['run_dir'])/'params',d/'params',dirs_exist_ok=True)
    for p in Path(entry['run_dir']).glob('events*'):shutil.copy2(p,d/p.name)
    audit[name]={'finite_tensor_count':finite(model),'logged_iterations':count,'iteration':model['iter'],'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'parent_sha256':entry['parent_sha256']}
    a=EventAccumulator(entry['run_dir'],size_guidance={'scalars':0});a.Reload();metrics={}
    for tag in a.Tags()['scalars']:
        values=a.Scalars(tag);tail=values[-100:]
        assert all(math.isfinite(v.value) for v in values)
        metrics[tag]={'first_step':values[0].step,'last_step':values[-1].step,'last':values[-1].value,'last100_mean':sum(v.value for v in tail)/len(tail)}
    (d/'training_metrics.json').write_text(json.dumps(metrics,indent=2))
(DEST/'checkpoint_audit.json').write_text(json.dumps(audit,indent=2))
inputs={
    'common_parent':PRIOR/'models/common_parent/model_7496.pt',
    'old_robust08_s1':Path('validation_artifacts/stage1_issues_20261108/models/robust08/model_8995.pt'),
    'old_robust08_s2':PRIOR/'models/robust08_s2/model_8995.pt',
    'old_strengthen_s1':PRIOR/'models/strengthen_s1/model_8995.pt',
    'old_strengthen_s2':PRIOR/'models/strengthen_s2/model_8995.pt'}
(DEST/'input_models.json').write_text(json.dumps({k:{'repository_path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_commit':'7e413284c5411a72e84aa1b5acf611faa769f043'} for k,p in inputs.items()},indent=2))
lines=['# Issue #6: bounded mass randomization and strengthened-push retraining','','Issue: https://github.com/weiguang52/HUMANOID_TW/issues/6 . Defaults remain legacy.','','## Mass correction and 4096-environment audit','','Only base_link is randomized. Original robust08 adds U(-0.08,0.15)kg to nominal0.07990783453kg and safety-clamps at1e-6kg. The opt-in bounded profile multiplies nominal by U(0.5,1.5), giving theoretical base range0.0399539173..0.1198617518kg. Same startup timing and inertia scaling. Static friction remains[0.4,1.4], dynamic[0.3,1.2]. Other body properties remain unchanged. This bounded research prior is not hardware identification or mean/variance-matched to the original additive distribution.','','Every audit uses4096 actual PhysX environments, all31 bodies. Requested masses are captured within the real mass event immediately before safety clamping, then actual applied masses are read back. Full per-body min/max/mean/quantiles, <1% nominal fractions, actual clamp counts and requested-clamp counts are saved in four mass_audit JSONs and mass_distribution_table.csv.','','| Profile | Seed | Base min kg | Base mean kg | Base max kg | Below1% nominal | At safety clamp | Requested at/below clamp |','|---|---:|---:|---:|---:|---:|---:|---:|']
csvrows=[]
for profile in ['legacy','bounded']:
    for seed in [1,2]:
        d=json.loads((ROOT/f'mass_audit_{profile}_seed{seed}.json').read_text())
        assert d['num_envs']==4096 and len(d['bodies'])==31
        assert d['friction_static']==[.4,1.4] and d['friction_dynamic']==[.3,1.2]
        for b in d['bodies']:
            csvrows.append({'profile':profile,'seed':seed,**{k:v for k,v in b.items() if k!='applied_quantiles_kg'}})
            if profile=='bounded':
                assert b['minimum_nominal_ratio']>=.5-1e-6 and b['applied_max_kg']<=1.5*b['nominal_min_kg']+1e-7
                assert b['at_safety_clamp_count']==b['requested_at_or_below_clamp_count']==b['below_1pct_nominal_count']==0
        b=next(x for x in d['bodies'] if x['body']=='base_link')
        lines.append(f"| {profile} | {seed} | {b['applied_min_kg']:.9g} | {b['applied_mean_kg']:.9g} | {b['applied_max_kg']:.9g} | {b['below_1pct_nominal_count']}/4096 ({b['below_1pct_nominal_fraction']:.4%}) | {b['at_safety_clamp_count']} | {b['requested_at_or_below_clamp_count']} |")
with (DEST/'mass_distribution_table.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(csvrows[0]),lineterminator="\n");w.writeheader();w.writerows(csvrows)
lines+=['','The installed Isaac Lab sampler passed four unit tests: lower endpoint and partial reset without accumulation; uniform bounds and untouched non-target bodies; old negative draws hit the safety clamp; legacy/default and disabled event behavior stay unchanged. Smoke64-env2-update resume also passed. Initial audit adapter signature failure was corrected before any formal training; its debug log remains remote and is excluded.','','## Protocol and controlled comparisons','','Both new models use4096 environments and1500 updates from the exact shared robust05 parent, SHA256 c200e355d29fda60225761550ea80642bb222778ce50c00ae3deb3d95fa7a38d, with full optimizer resume and no std reset or checkpoint selection. Training push is +/-0.45m/s. Both seeds share one seed1 parent; they are not full-pipeline independent replications.','','Evaluation: deterministic policy,128 environments,20s per0.3/0.5/0.8m/s command; eval seeds1/2/3, paired reset seed=eval_seed*1000+command_index. Combined evaluation retains original +/-0.3m/s pushes, random0/20ms delay and noise. Material-only disables these non-target factors. New mass distribution is used in both conditions.24 fresh quantitative jobs, plus archived legacy references.','','To separate changed evaluation physics from retraining, historical robust08 checkpoints are also evaluated with bounded material; historical strengthened-push checkpoints with bounded combined. Paired reset seeds do not imply identical trajectories or identical startup physics across changed distributions.','','Each cell below lists eval seeds1/2/3, denominator128 each.','','| Model | Evaluated mass | Condition | 0.3m/s | 0.5m/s | 0.8m/s |','|---|---|---|---|---|---|']
def rows(name,condition,legacy=False):
    result=[]
    for seed in [1,2,3]:
        p=(PRIOR if legacy else ROOT)/f'{name}_{condition}_eval{seed}.json'
        data=json.loads(p.read_text())
        assert data['paired_resets'] and data['deterministic_policy'] and data['issue5_condition']==condition
        assert [x['command'] for x in data['results']]==commands
        if not legacy:
            assert data['mass_randomization']=='bounded'
            c=data['perturbation_config'];assert c['mass_operation']=='scale' and c['mass_distribution_params']==[.5,1.5]
            assert c['friction_static']==[.4,1.4] and c['friction_dynamic']==[.3,1.2]
            assert c['action_class']==('RandomOneStepDelayAction' if condition=='combined' else 'JointPositionAction')
            assert data['pushes']==(condition=='combined') and data['observation_noise']==(condition=='combined')
            if condition=='combined':assert c['push_velocity_range']=={'x':[-.3,.3],'y':[-.3,.3]}
            for b in data['mass_distribution']['bodies']:
                assert b['minimum_nominal_ratio']>=.5-1e-6 and b['at_safety_clamp_count']==b['below_1pct_nominal_count']==0
        for x in data['results']:
            assert x['num_envs']==128 and x['seconds']==20 and 0<=x['failed_trials']<=128
            assert abs(x['fall_rate']-x['failed_trials']/128)<1e-7
            assert all(math.isfinite(v) for v in x['metrics'].values())
        result+=data['results']
        if legacy:
            d=DEST/'legacy_references';d.mkdir(exist_ok=True);shutil.copy2(p,d/p.name)
    return result
table={}
def add(name,condition,legacy=False):
    rs=rows(name,condition,legacy);key=f"{name}_{'legacy' if legacy else 'bounded'}_{condition}"
    table[key]={c:[x['failed_trials'] for x in rs if x['command']==c] for c in commands}
    lines.append('| '+name+' | '+('legacy' if legacy else 'bounded')+' | '+condition+' | '+' | '.join('/'.join(map(str,table[key][c])) for c in commands)+' |')
    return rs
for seed in [1,2]:
    add(f'robust08_s{seed}','material',True)
    add(f'old_robust08_s{seed}','material')
    add(f'strengthen_s{seed}','combined',True)
    add(f'old_strengthen_s{seed}','combined')
    for condition in ['combined','material']:add(f'bounded_strengthen_s{seed}',condition)
assert len(list(ROOT.glob('*_eval*.json')))==24
(DEST/'comparison_counts.json').write_text(json.dumps(table,indent=2))

effects={}
material_rates=[]
lines+=['','## Separating evaluation-physics correction from retraining','','Each pooled count below sums three evaluation seeds (denominator384 per speed). The mass-only comparison uses the same robust08 policy before/after the evaluation correction. The combined comparison first holds the old strengthened policy fixed while changing evaluation mass, then holds bounded evaluation fixed while changing the trained policy.','','| Seed | Command | Robust08 material legacy -> bounded | Old strengthened combined legacy -> bounded | Bounded-trained combined |','|---|---|---|---|---|']
for seed in [1,2]:
    effects[str(seed)]={}
    for c in commands:
        def pooled(key):return sum(table[key][c])
        material_old=pooled(f'robust08_s{seed}_legacy_material')
        material_new=pooled(f'old_robust08_s{seed}_bounded_material')
        combined_old=pooled(f'strengthen_s{seed}_legacy_combined')
        combined_fixed=pooled(f'old_strengthen_s{seed}_bounded_combined')
        combined_retrained=pooled(f'bounded_strengthen_s{seed}_bounded_combined')
        effects[str(seed)][c]={'denominator':384,'material_legacy':material_old,'material_bounded_same_policy':material_new,'combined_legacy':combined_old,'combined_bounded_same_policy':combined_fixed,'combined_bounded_retrained':combined_retrained,'combined_eval_correction_delta':combined_fixed-combined_old,'combined_retraining_delta_under_bounded_eval':combined_retrained-combined_fixed}
        lines.append(f'| {seed} | {c} | {material_old} -> {material_new} | {combined_old} -> {combined_fixed} | {combined_retrained} |')
    m_old=sum(effects[str(seed)][c]['material_legacy'] for c in commands)/1152
    m_new=sum(effects[str(seed)][c]['material_bounded_same_policy'] for c in commands)/1152
    material_rates.append(f'- Robust08 training seed{seed}: material-only pooled fall rate {m_old:.4%} -> {m_new:.4%} (same policy, changed mass distribution).')
lines+=['',*material_rates]
(DEST/'controlled_comparison_effects.json').write_text(json.dumps(effects,indent=2))

acceptance={}
for name in state:
    rs=rows(name,'combined')
    acceptance[name]={'target_03_05_pass':all(x['fall_rate']<.01 for x in rs if x['command'] in commands[:2]),'falls_08':[x['failed_trials'] for x in rs if x['command']==commands[2]]}
both=all(v['target_03_05_pass'] for v in acceptance.values())
(DEST/'acceptance.json').write_text(json.dumps({'models':acceptance,'both_training_seeds_pass':both,'defaults_promoted':False},indent=2))
lines+=['','## Acceptance and interpretation','','Frozen target: every evaluation seed at0.3 and0.5m/s strictly<1%, with0.8m/s recorded.']
for name,v in acceptance.items():lines.append('- '+name+': '+('PASS' if v['target_03_05_pass'] else 'FAIL')+';0.8m/s falls '+str(v['falls_08']))
lines+=['','Both training seeds: '+('PASS' if both else 'FAIL')+'. No default configuration/policy is promoted. Completed fixed-budget experiments do not imply a passed performance gate.','','Crossed comparisons are descriptive. Correcting mass changes mean, variance and tails, so differences cannot be attributed solely to removing safety-clamped samples. New-vs-old trained-policy comparisons under the same bounded evaluation reduce that confounding but two conditional training seeds do not prove broad robustness. The original #5 no-single-dominant-factor conclusion is not rewritten by these new protocols.']
variance={}
for prefix,condition in [('old_robust08','material'),('old_strengthen','combined'),('bounded_strengthen','combined'),('bounded_strengthen','material')]:
    variance[prefix+'_'+condition]={}
    for c in commands:
        rates=[sum(x['failed_trials'] for x in rows(f'{prefix}_s{seed}',condition) if x['command']==c)/384 for seed in [1,2]]
        variance[prefix+'_'+condition][c]={'training_seed_rates':rates,'mean':statistics.mean(rates),'sample_sd':statistics.stdev(rates)}
(DEST/'training_seed_variance.json').write_text(json.dumps(variance,indent=2))
lines+=['','Measured conclusion: the sampling defect is fixed (no clamp/near-zero masses), but this fixed-budget retraining does not improve falls under the common bounded combined evaluation: five of six training-seed/speed pooled counts are higher than the old strengthened policies, and one is equal. Material-only correction with fixed robust08 policies changes rates in opposite directions across training seeds. No consistent policy-robustness benefit is demonstrated.','']
lines+=['','Descriptive two-training-seed means and sample standard deviations are saved in training_seed_variance.json; they are not confidence guarantees.','','## Colored worst-command front/side videos','','Worst command is selected from summed combined failures over three evaluation seeds, ties0.3/0.5/0.8. Each video is a one-environment diagnostic replay, not128-env acceptance. Camera follows translation and heading; front/side labels are in filenames. Exact precolored USD is compared with original physics and non-material relationships; no dynamic material binding. Velocity commands have no HumanML3D source NPY. Joint color legend: joint_colors.json.']
d=DEST/'videos';d.mkdir(exist_ok=True)
for p in sorted((ROOT/'videos').glob('*_worst_*.mp4')):
    shutil.copy2(p,d/p.name);lines.append(f'- [{p.stem}](videos/{p.name})')
assert len(list(d.glob('*.mp4')))==4
shutil.copy2(PRIOR/'joint_colors.json',DEST/'joint_colors.json')
shutil.copy2(ROOT/'issue6_video_contact_sheet.png',DEST/'issue6_video_contact_sheet.png')
lines+=['','All artifacts were generated remotely. Archived logs strip ANSI decoration/trailing whitespace; raw originals and debug outputs remain on the remote data disk. No mounted file is copied to local storage.','','## Project tracking','','Notion replaces Linear: https://www.notion.so/403325d8231d8172995bf19510b8d760 . GitHub issue reply records measured results and any unmet gate. SuperMimic remains stopped; this is stage1 velocity-control research.']
(DEST/'README.md').write_text('\n'.join(lines)+'\n')
for log in DEST.rglob('*.log'):
    value=re.sub(r'\x1b\[[0-9;?]*[A-Za-z]','',log.read_text(errors='replace'))
    log.write_text('\n'.join(line.rstrip() for line in value.splitlines()).rstrip()+'\n')
manifest={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file() and p.name!='sha256.json'}
(DEST/'sha256.json').write_text(json.dumps(manifest,indent=2))
print('ARCHIVED',len(manifest),'files; acceptance',acceptance,flush=True)
