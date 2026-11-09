"""Package issue #9 five-vs-five training-seed calibration."""
import json,csv,hashlib,re,shutil,math
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ROOT=Path('/root/gpufree-data/tw_stage1/issue9_20261109')
DEST=Path('validation_artifacts/stage1_issue9_20261109')
assert (ROOT/'all_training_evaluation_video_complete').exists()
assert (ROOT/'video_physics_manual_audit.json').exists() and (ROOT/'video_quality_audit.json').exists()
assert json.loads((ROOT/'raw_trace_integrity_audit.json').read_text())['passed']
assert '\nOK\n' in (ROOT/'statistics_tests.log').read_text()
config=json.loads((ROOT/'config_difference_audit.json').read_text());assert len(config['runs'])==6 and len(config['matched_pairs'])==3
identity=json.loads((ROOT/'training_source_identity.json').read_text())
for name,digest in identity['sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
state=json.loads((ROOT/'training_state.json').read_text())
assert set(state)=={f'{v}_s{s}' for v in ['requested','applied'] for s in [3,4,5]}
inputs=json.loads((ROOT/'input_models.json').read_text());assert len(inputs)==10
for value in inputs.values():assert hashlib.sha256(Path(value['checkpoint']).read_bytes()).hexdigest()==value['sha256']
statistics=json.loads((ROOT/'seed_level_statistics.json').read_text());planning=json.loads((ROOT/'power_planning.json').read_text());accept=json.loads((ROOT/'acceptance.json').read_text())
parent=json.loads((ROOT/'parent_availability_audit.json').read_text());assert not parent['independent_other_training_seed_available']
DEST.mkdir(exist_ok=True)
for p in ROOT.iterdir():
    if p.is_file() and (p.suffix in ['.json','.csv','.md','.npz'] or p.suffix=='.log' and p.name!='archive.log'):shutil.copy2(p,DEST/p.name)
for folder in ['traces','figures','videos']:shutil.copytree(ROOT/folder,DEST/folder,dirs_exist_ok=True)
shutil.copy2(ROOT/'issue9_video_contact_sheet.png',DEST/'issue9_video_contact_sheet.png')
shutil.copy2('validation_artifacts/stage1_issue8_20261109/joint_colors.json',DEST/'joint_colors.json')
references={}
for tag,bundle in [('issue7','stage1_issue7_20261109'),('issue8','stage1_issue8_20261109')]:
    ref=Path('validation_artifacts')/bundle;manifest=json.loads((ref/'sha256.json').read_text());d=DEST/'baseline_sources'/tag;d.mkdir(parents=True,exist_ok=True)
    for name in ['rates_wilson95.csv','pooled_rates_wilson95.csv','input_models.json']:
        assert hashlib.sha256((ref/name).read_bytes()).hexdigest()==manifest[name]
        shutil.copy2(ref/name,d/name)
    references[tag]=json.loads((ref/'input_models.json').read_text())
for seed in [1,2]:
    assert inputs[f'requested_s{seed}']['sha256']==references['issue7'][f'new_bounded_s{seed}']['sha256']
    assert inputs[f'applied_s{seed}']['sha256']==references['issue8'][f'applied_s{seed}']['sha256']
def finite(value):
    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all();return 1
    if isinstance(value,dict):return sum(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(finite(v) for v in value)
    return 0
def shapes(value):return {group+'/'+k:list(v.shape) for group in ['actor_state_dict','critic_state_dict'] for k,v in value[group].items() if isinstance(v,torch.Tensor)}
audits={}
for name,entry in state.items():
    p=Path(entry['checkpoint']);model=torch.load(p,map_location='cpu',weights_only=False)
    baseline=inputs[entry['variant']+'_s1']['checkpoint'];old=torch.load(baseline,map_location='cpu',weights_only=False)
    assert shapes(model)==shapes(old)
    count=len(re.findall(r'Learning iteration \d+/\d+',(ROOT/f'{name}_train.log').read_text()));assert count==1500 and model['iter']==8995
    d=DEST/'models'/name;d.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d/p.name);shutil.copytree(Path(entry['run_dir'])/'params',d/'params',dirs_exist_ok=True)
    for log in Path(entry['run_dir']).glob('events*'):shutil.copy2(log,d/log.name)
    a=EventAccumulator(entry['run_dir'],size_guidance={'scalars':0});a.Reload();metrics={}
    for tag in a.Tags()['scalars']:
        values=a.Scalars(tag);assert all(math.isfinite(v.value) for v in values);tail=values[-100:]
        metrics[tag]={'first_step':values[0].step,'last_step':values[-1].step,'last':values[-1].value,'last100_mean':sum(v.value for v in tail)/len(tail)}
    (d/'training_metrics.json').write_text(json.dumps(metrics,indent=2))
    audits[name]={'variant':entry['variant'],'training_seed':entry['seed'],'logged_iterations':count,'iteration':model['iter'],'finite_tensor_count':finite(model),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'parent_sha256':entry['parent_sha256'],'architecture_shapes_equal_baseline':True}
(DEST/'checkpoint_audit.json').write_text(json.dumps(audits,indent=2))
def rows(name):return list(csv.DictReader((ROOT/name).open()))
rates=rows('training_seed_rates.csv');assert len(rates)==30
def row(variant,seed,command):return next(x for x in rates if x['variant']==variant and int(x['training_seed'])==seed and x['command']==command)
commands=['forward_03','forward_05','forward_08']
for c in commands:assert statistics[c]['training_pairs']==5 and statistics[c]['exact_resamples']==3125
labels={'supports_lower_applied_rate':'conditional bootstrap evidence supports a lower applied fall rate','supports_higher_applied_rate':'conditional bootstrap evidence supports a higher applied fall rate (against applied)','cannot_distinguish_at_current_seed_count':'cannot distinguish requested and applied at the current training-seed count'}
lines=['# Issue #9: requested vs applied, five training seeds per variant','','Issue: https://github.com/weiguang52/HUMANOID_TW/issues/9 . Base6d691ae. No configuration or default changes.','','## Cohort and provenance','','Existing requested seeds1/2 are issue6 bounded-strengthen checkpoints evaluated under issue7; existing applied seeds1/2 are issue8. Fixed new training seeds3/4/5 were run in order requested3,applied3,requested4,applied4,requested5,applied5. All ten policies share the exact robust05 seed1 parent SHA256 c200e355d29fda60225761550ea80642bb222778ce50c00ae3deb3d95fa7a38d.','','Each of six new runs uses4096 environments,1500updates, full optimizer resume, no std reset or checkpoint selection. Bounded mass*U(0.5,1.5), training pushes+/-0.45m/s, original friction/noise/random0/20ms delay, architecture/rewards/critic unchanged. Requested vs applied retains only the previously defined policy last_action difference. Six saved-config and three matched-pair audits verify allowed seed/name changes and the one existing observation difference. Training/configuration source bytes match6d691ae.','','All policies use the unchanged issue7 evaluation:bounded combined,1024 environments,3evaluation seeds,0.3/0.5/0.8m/s,20s, deterministic policy and paired resets. Evaluation pushes remain+/-0.3m/s. New18jobs/55296trials plus archived36864trials give92160 total observed trials, but the statistical sample size is FIVE training pairs, not92160independent training samples.','','## Five-vs-five individual training-seed rates','','Each rate pools exactly3072evaluation trials within one trained policy and speed.','','| Command | Training seed | Requested falls/rate | Applied falls/rate | Applied-requested (pp) |','|---|---:|---|---|---:|']
for c in commands:
    for seed in [1,2,3,4,5]:
        r=row('requested',seed,c);a=row('applied',seed,c)
        lines.append(f"| {c} | {seed} | {r['falls']}/3072; {float(r['fall_rate']):.3%} | {a['falls']}/3072; {float(a['fall_rate']):.3%} | {100*(float(a['fall_rate'])-float(r['fall_rate'])):+.3f} |")
lines+=['','## Training-seed group means and sample standard deviations','','| Command | Variant | Mean | Sample SD (pp,ddof1) | Seed-bootstrap95% mean interval |','|---|---|---:|---:|---|']
for c in commands:
    for variant in ['requested','applied']:
        s=statistics[c][variant];lo,hi=s['bootstrap95']
        lines.append(f"| {c} | {variant} | {s['mean']:.3%} | {100*s['sample_sd']:.3f} | [{lo:.3%}, {hi:.3%}] |")
lines+=['','## Matched-seed differences and frozen conclusion','','Exact empirical bootstrap enumerates all5^5=3125with-replacement training-index vectors. The same indices resample requested/applied together. Linear2.5/97.5percentiles define intervals. Numerical values within1e-12rate units of zero/threshold equality are treated as equality to avoid floating-point false exclusions.3125resamples are not3125new training seeds. Group intervals are NOT binomial intervals obtained by pooling trials. With equal denominators, means happen to equal pooled point rates, but uncertainty is calculated from five seed rates. The observed between-policy SD also contains finite evaluation error under the fixed3072-trial protocol; it is not a decomposition of pure optimizer variance. The intervals are conditional on the chosen evaluation streams and shared parent.','','| Command | Mean applied-requested (pp) | Paired-difference SD (pp) | Paired seed-bootstrap95% (pp) | Excludes0 | Interpretation |','|---|---:|---:|---|---|---|']
for c in commands:
    s=statistics[c];d=s['paired_applied_minus_requested'];lo,hi=d['bootstrap95']
    lines.append(f"| {c} | {100*d['mean']:+.3f} | {100*d['sample_sd']:.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] | {s['difference_ci_excludes_zero']} | {labels[s['effect_label']]} |")
primary=accept['primary_effect_label']
lines+=['','**Prespecified primary0.5m/s conclusion: '+labels[primary]+'.** Other speeds retain their separate conclusions and are not promoted into a universal all-speed claim. The three intervals are not a simultaneous familywise95%guarantee. With only five observed pairs, empirical percentile tail/coverage uncertainty remains substantial; results are conditional on this shared parent.','','![Five matched seed rates](figures/five_seed_paired_rates.png)','','![Paired differences](figures/paired_difference_intervals.png)','','## Low/mid-speed group-mean threshold','','| Variant | 0.3m/s mean95% upper | 0.5m/s mean95% upper | BOTH strictly<1% |','|---|---:|---:|---|']
for variant,v in accept['group_mean_thresholds'].items():
    lines.append(f"| {variant} | {v['low_mid_mean_ci_upper']['forward_03']:.3%} | {v['low_mid_mean_ci_upper']['forward_05']:.3%} | {v['both_upper_below1pct']} |")
lines+=['','Any group meets both thresholds: '+str(accept['any_group_both_low_mid_upper_below1pct'])+'. Defaults promoted: NO, regardless of result.','','## Training-seed count to detect an absolute3pp effect','','Planning uses a two-sided alpha0.05,80%power noncentral-t model. Main design is matched training pairs, using observed sampleSD of applied-requested. N pairs means N seeds PER configuration (2Ntraining runs). Independent equal-size group planning is a sensitivity estimate using pooled groupSD. This is planning for a future separately authorized study, not additional training launched here.','','| Command | Pilot paired SD (pp) | Estimated matched pairs needed | Pair-count sensitivity from SD95% bounds | Additional pairs beyond current5 | Independent seeds per arm |','|---|---:|---:|---|---:|---:|']
for c in commands:
    p=planning['per_speed'][c];bounds=p['paired_n_using_sd95_bounds']
    lines.append(f"| {c} | {100*p['paired_difference_sample_sd']:.3f} | {p['paired_training_pairs_needed']} | {bounds} | {p['additional_pairs_beyond_pilot']} | {p['unpaired_training_seeds_per_arm']} |")
lines+=['','The SD range uses the normal-theory chi-square interval from only five pilot differences. Its pair-count range is a sensitivity calculation, not a calibrated confidence interval for required N. Normality/constant variance, near-boundary fall rates, sparse pilot tails and the shared parent can all invalidate precise planning claims. A zero pilot SD is reported as not reliably estimable. The absolute effect0.03is three percentage points, not a3%relative change.','','![Illustrative primary-speed power](figures/three_pp_power_planning.png)','','## Independent-parent check','','Only three byte-identical copies of the seed1 robust05 final checkpoint were found in accessible training/archive roots. No different-source robust05 training-seed checkpoint is available. Therefore the optional parent-source comparison is MISSING, not silently replaced with a robust08 child or another curriculum stage. See parent_availability_audit.json for roots, explicit run-name criteria, highest numeric checkpoints and hashes. Five seeds conditional on one parent do not establish independent full-pipeline replication or parent-robust effects.','','## Validation and artifacts','','Five statistics tests cover joint paired resampling, exact resample count, sampleSD, identical-group CI, two-sided null power and planning behavior. All six new checkpoints have1500logged updates, final iteration8995, finite tensors and baseline-compatible actor/critic shapes. All ten input checkpoint hashes and baseline CSV manifests are checked. Raw first-failure records, pre-reset states, actual push bounds/delay labels and video audits accompany the study; these diagnostics do not change the training-seed statistical unit.','','## Colored worst-command front/side videos','','Six new models each have20s colored front/side diagnostics at their highest pooled-fall command, tie0.3/0.5/0.8. Videos use one environment and do not replace1024-env acceptance. Cameras follow translation and heading; exact precolored USD preserves original physics/non-material relationships, with no runtime material binding. Joint legend: joint_colors.json. Velocity commands have no sourceHumanML3D NPY.']
for p in sorted((DEST/'videos').glob('*.mp4')):lines.append(f'- [{p.stem}](videos/{p.name})')
assert len(list((DEST/'videos').glob('*.mp4')))==12
lines+=['','Existing seeds1/2 reference videos remain in their original versioned bundles:']
links=[]
for variant,bundle,prefix in [('requested','stage1_issue6_20261109','bounded_strengthen'),('applied','stage1_issue8_20261109','applied')]:
    for seed in [1,2]:
        for view in ['front','side']:
            found=list((Path('validation_artifacts')/bundle/'videos').glob(f'{prefix}_s{seed}_worst_{view}_*.mp4'));assert len(found)==1
            target='../'+bundle+'/videos/'+found[0].name;lines.append(f'- [{variant} seed{seed} {view}]({target})');links.append({'variant':variant,'training_seed':seed,'view':view,'relative_link':target,'sha256':hashlib.sha256(found[0].read_bytes()).hexdigest()})
(DEST/'existing_video_links.json').write_text(json.dumps(links,indent=2))
lines+=['','## Tracking','','All artifacts were generated remotely. Raw original logs remain on the data disk; archive logs strip ANSI/trailing whitespace. No mounted files are copied to local storage. Notion replaces Linear: https://www.notion.so/403325d8231d8172995bf19510b8d760 . Issue reply reports the primary/secondary seed-level conclusions and illustrative3pp planning estimate. No default configuration is promoted.']
(DEST/'README.md').write_text('\n'.join(lines)+'\n')
for log in DEST.rglob('*.log'):
    text=re.sub(r'\x1b\[[0-9;?]*[A-Za-z]','',log.read_text(errors='replace'))
    log.write_text('\n'.join(line.rstrip() for line in text.splitlines()).rstrip()+'\n')
manifest={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file() and p.name!='sha256.json'}
(DEST/'sha256.json').write_text(json.dumps(manifest,indent=2))
print('ARCHIVED',len(manifest),'files; PRIMARY',primary,'PLANNED_PAIRS',planning['per_speed']['forward_05']['paired_training_pairs_needed'],flush=True)
