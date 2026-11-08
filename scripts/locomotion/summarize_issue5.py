"""Archive complete issue #5 evidence without hiding seed-level failures."""
import hashlib,json,math,re,shutil,statistics,subprocess
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ROOT=Path('/root/gpufree-data/tw_stage1/issue5_20261109')
DEST=Path('validation_artifacts/stage1_issue5_20261109')
assert (ROOT/'all_training_evaluation_video_complete').exists(), 'Required jobs remain'
plan=json.loads((ROOT/'training_plan.json').read_text());state=json.loads((ROOT/'training_state.json').read_text());assert len(state)==5
assert (ROOT/'video_quality_audit.json').exists(), 'Visual validation remains'
DEST.mkdir(exist_ok=True)
for pattern in ['*.json','frozen_plan.md','delay_tests.log','remove_smoke.log','strengthen_smoke.log','training_runner.log','factor_runner.log','video_audit.log','*_worst_*.log']:
    for p in ROOT.glob(pattern):shutil.copy2(p,DEST/p.name)

def tensor_audit(value):
    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all();return 1
    if isinstance(value,dict):return sum(tensor_audit(v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(tensor_audit(v) for v in value)
    return 0

audit={}
for name,entry in state.items():
    d=DEST/'models'/name;d.mkdir(parents=True,exist_ok=True);checkpoint=Path(entry['checkpoint'])
    model=torch.load(checkpoint,map_location='cpu',weights_only=False)
    count=len(re.findall(r'Learning iteration \d+/\d+',(ROOT/(name+'_train.log')).read_text()))
    assert count==1500 and model['iter']==int(checkpoint.stem.split('_')[-1])
    audit[name]={'finite_tensor_count':tensor_audit(model),'iteration':model['iter'],'logged_iterations':count,'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'parent_sha256':entry['parent_sha256']}
    shutil.copy2(checkpoint,d/checkpoint.name);shutil.copytree(Path(entry['run_dir'])/'params',d/'params',dirs_exist_ok=True)
    shutil.copy2(ROOT/(name+'_train.log'),d/'train.log')
    for p in Path(entry['run_dir']).glob('events*'):shutil.copy2(p,d/p.name)
    a=EventAccumulator(entry['run_dir'],size_guidance={'scalars':0});a.Reload();metrics={}
    for tag in a.Tags()['scalars']:
        values=a.Scalars(tag);tail=values[-100:];metrics[tag]={'first_step':values[0].step,'last_step':values[-1].step,'last':values[-1].value,'last100_mean':sum(v.value for v in tail)/len(tail)}
    (d/'training_metrics.json').write_text(json.dumps(metrics,indent=2))
for name,p in [('robust08_s1',Path('validation_artifacts/stage1_issues_20261108/models/robust08/model_8995.pt')),('common_parent',Path(plan['parent']))]:
    d=DEST/'models'/name;d.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d/p.name);shutil.copytree(p.parent/'params',d/'params',dirs_exist_ok=True)
(DEST/'checkpoint_audit.json').write_text(json.dumps(audit,indent=2))
CONDITIONS=['clean','push','material','delay','noise','combined'];COMMANDS=['forward_03','forward_05','forward_08']

def rows(name,condition):
    result=[]
    for seed in [1,2,3]:
        p=ROOT/f'{name}_{condition}_eval{seed}.json';data=json.loads(p.read_text());assert data['paired_resets'] and data['issue5_condition']==condition
        assert [x['command'] for x in data['results']]==COMMANDS
        for x in data['results']:
            assert x['num_envs']==128 and x['seconds']==20 and all(math.isfinite(v) for v in x['metrics'].values())
            result.append(x)
    return result

lines=['# Issue #5: factor attribution and multi-training-seed replication','','Issue: https://github.com/weiguang52/HUMANOID_TW/issues/5 . Defaults unchanged. All work and artifacts remain on the remote data disk.','','## Protocol','','Existing robust08 seed1 and its seed2 replication use clean, push-only, friction/mass-only, fixed20ms-delay-only, observation-noise-only and original combined challenge. Every condition uses3 evaluation seeds,128 environments and20s per command at0.3/0.5/0.8m/s. Per-command reset seed is evaluation_seed*1000+command_index. Non-target perturbations are disabled. Combined retains random0/20ms delay; isolated delay is fixed20ms. These are new paired streams, not an exact replay of historical unpaired random streams.','','## Single-factor fall counts','','Each cell lists evaluation seeds1/2/3; each denominator is128.','','| Condition | Command | Training seed1 | Training seed2 | Mean fall rate across training seeds | Sample SD across training seeds |','|---|---|---|---|---:|---:|']
variance={}
for condition in CONDITIONS:
    rr=[rows(f'robust08_s{s}',condition) for s in [1,2]]
    variance[condition]={}
    for command in COMMANDS:
        counts=[[x['failed_trials'] for x in rs if x['command']==command] for rs in rr]
        rates=[sum(cs)/(3*128) for cs in counts];mean=statistics.mean(rates);sd=statistics.stdev(rates)
        variance[condition][command]={'counts_by_training_seed':counts,'training_seed_mean_rates':rates,'mean':mean,'sample_sd':sd}
        cells=['/'.join(map(str,cs)) for cs in counts]
        lines.append(f'| {condition} | {command} | {cells[0]} | {cells[1]} | {mean:.3%} | {sd:.3%} |')
(DEST/'training_seed_variance.json').write_text(json.dumps(variance,indent=2))
attr=json.loads((ROOT/'factor_attribution.json').read_text())

replication={}
for training_seed in [1,2]:
    by_seed={condition:[sum(x['failed_trials'] for x in rows(f'robust08_s{training_seed}',condition)[i*3:(i+1)*3])/(3*128) for i in range(3)] for condition in CONDITIONS}
    scores={k:statistics.mean(v) for k,v in by_seed.items()}
    factors=['push','material','delay','noise']
    ranked=sorted(factors,key=lambda k:(-scores[k],factors.index(k)))
    factor=ranked[0];excess=scores[factor]-scores['clean'];full=scores['combined']-scores['clean']
    stable=all(factor==sorted(factors,key=lambda k:(-by_seed[k][i],factors.index(k)))[0] for i in range(3))
    dominant=full>0 and excess>=.05 and scores[factor]-scores[ranked[1]]>=.05 and excess>=.5*full and stable
    replication[str(training_seed)]={'scores':scores,'scores_by_eval_seed':by_seed,'dominant_factor':factor if dominant else None,'strongest_isolated_factor':factor}
assert replication['1']['dominant_factor']==attr['dominant_factor']
(DEST/'attribution_training_seed_replication.json').write_text(json.dumps(replication,indent=2))
lines+=['','The identical frozen attribution rule yields '+str({k:v['dominant_factor'] for k,v in replication.items()})+' across training seeds. See attribution_training_seed_replication.json. This replication does not change the already frozen training selection.']

lines+=['','## Attribution and limits','',attr['claim']+'. The predeclared rule and full seed1 scores are in factor_attribution.json and frozen_plan.md. The training factor is '+plan['factor']+'. If no dominant factor meets the frozen criteria, the training ablations are exploratory probes of the strongest isolated stressor, not proof of its causal dominance.','', 'Only two training seeds are available, both conditional on the identical seed1 robust05 parent and full optimizer. The sample SD above is descriptive and does not establish broad training-seed robustness. A combined failure rate higher than isolated conditions is consistent with interacting stresses, but these runs do not identify a unique interaction mechanism.','','## Ablation outcomes','','Remove disables only the selected component. Strengthen changes only its strength relative to the original combined robust08 package. All five formal runs use4096 environments,1500 logged updates and the exact common parent; no std reset or checkpoint selection. See training_plan.json and saved configurations.','','| Model | Condition | 0.3m/s falls | 0.5m/s falls | 0.8m/s falls |','|---|---|---|---|---|']
acceptance={}
for name in ['robust08_s1',*state]:
    conditions=CONDITIONS if name.startswith('robust08') else ['clean',plan['factor'],'combined']
    for condition in conditions:
        rs=rows(name,condition);cells=['/'.join(str(x['failed_trials']) for x in rs if x['command']==c) for c in COMMANDS]
        lines.append(f'| {name} | {condition} | '+' | '.join(cells)+' |')
        if condition=='combined':acceptance[name]={'target_03_05_pass':all(x['fall_rate']<.01 for x in rs if x['command'] in COMMANDS[:2]),'falls_08':[x['failed_trials'] for x in rs if x['command']=='forward_08']}
lines+=['','## Acceptance','','Combined-challenge target: every evaluation seed at0.3 and0.5m/s must have fall rate strictly below1%;0.8m/s is recorded without threshold changes.','']
for name,result in acceptance.items():lines.append(f'- {name}: '+('PASS' if result['target_03_05_pass'] else 'FAIL')+';0.8m/s counts '+str(result['falls_08']))
for mode in ['remove','strengthen']:
    passed=all(acceptance[f'{mode}_s{s}']['target_03_05_pass'] for s in [1,2]);lines.append(f'- {mode}, both training seeds: '+('PASS' if passed else 'FAIL'))
(DEST/'acceptance.json').write_text(json.dumps(acceptance,indent=2))

effects={}
lines+=['','## Matched-training-seed interpretation','','Neither training seed meets the frozen single-dominant-factor rule. Push is the strongest isolated stressor, so the selected training ablations remain exploratory. Removing push exposure worsens combined-challenge falls at all three tested speeds in both training seeds. Strengthening only push exposure reduces the counts at all three speeds in both training seeds, but neither seed passes the low/mid-speed target. These are descriptive comparisons conditional on one shared parent; two seeds and this isolated-factor design do not establish a unique interaction mechanism.','','The following pooled counts sum three evaluation seeds (denominator384), without replacing the stricter per-evaluation-seed acceptance.','','| Training seed | Command | Original combined | Remove push combined | Strengthen push combined |','|---|---|---:|---:|---:|']
for seed in [1,2]:
    effects[str(seed)]={}
    for command in COMMANDS:
        counts={mode:sum(x['failed_trials'] for x in rows(f'{prefix}_s{seed}','combined') if x['command']==command) for mode,prefix in [('original','robust08'),('remove','remove'),('strengthen','strengthen')]}
        effects[str(seed)][command]={'denominator':384,**counts,'remove_minus_original':counts['remove']-counts['original'],'strengthen_minus_original':counts['strengthen']-counts['original']}
        lines.append(f"| {seed} | {command} | {counts['original']}/384 | {counts['remove']}/384 | {counts['strengthen']}/384 |")
(DEST/'matched_training_seed_effects.json').write_text(json.dumps(effects,indent=2))

lines+=['','No default policy/configuration is promoted by this research delivery. A completed experiment is not equivalent to performance acceptance.','','## Physical mass audit','','The original source base mass is0.07990783598kg. Isaac Lab clamps randomized masses to1e-6kg and rescales inertia. Some material conditions include extremely light bodies; inspect physics_diagnostics in the recorded JSON. These are part of the original package and must not be interpreted as realistic hardware identification. No negative mass is applied. Mass diagnostics were added after the earliest clean/push runs; absent diagnostic fields are not fabricated.','','## Worst-command colored front/side videos','','The worst instruction per model is selected by total combined-challenge failures over all3 evaluation seeds, ties resolved in0.3/0.5/0.8 order. Videos are diagnostic one-environment seed1 replays, with failure/reset statistics in their JSON; they do not replace128-env acceptance. Camera follows translation and heading. Exact precolored USD is checked against original composed attributes and non-material relationships; no runtime material binding. Velocity commands have no HumanML3D source NPY.','']
(DEST/'videos').mkdir(exist_ok=True)
video_plan=json.loads((ROOT/'video_plan.json').read_text());assert len(video_plan)==6
for p in sorted((ROOT/'videos').glob('*_worst_*.mp4')):
    shutil.copy2(p,DEST/'videos'/p.name);lines.append(f'- [{p.stem}](videos/{p.name})')
assert len(list((DEST/'videos').glob('*.mp4')))==12
colors=Path('/root/gpufree-data/datasets/practice9/custom_robot/usd/urdf0711_training_30dof_tw123_yaw_waist_v3_colored.colors.json');shutil.copy2(colors,DEST/'joint_colors.json')
lines+=['','Joint color legend: [joint_colors.json](joint_colors.json). Numerical and visual validation records accompany the report. Archived logs remove ANSI decoration and trailing whitespace; raw originals remain on the remote data disk. Failed/intermediate debug outputs remain excluded on the remote data disk.','','## Project tracking','','Notion replaces Linear: https://www.notion.so/403325d8231d8172995bf19510b8d760 . GitHub issue replies record the measured result and any unmet gate. SuperMimic remains stopped; this is a separate stage1 velocity-control attribution task.']
shutil.copy2(ROOT/'issue5_video_contact_sheet.png',DEST/'issue5_video_contact_sheet.png')
(DEST/'README.md').write_text('\n'.join(lines)+'\n')
# Preserve raw logs on the remote disk; archive readable text without ANSI
# decoration or trailing whitespace. Numerical lines and ordering are unchanged.
for log in DEST.rglob('*.log'):
    value=re.sub(r'\x1b\[[0-9;?]*[A-Za-z]', '', log.read_text(errors='replace'))
    log.write_text('\n'.join(line.rstrip() for line in value.splitlines()).rstrip()+'\n')
manifest={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file() and p.name!='sha256.json'};(DEST/'sha256.json').write_text(json.dumps(manifest,indent=2));print('Archived',len(manifest),'files; review visual evidence and report before commit')
