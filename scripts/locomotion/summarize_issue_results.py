"""Build an auditable report from completed remote issue experiments."""
import hashlib,json,shutil,subprocess
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue_runs_20261108')
DEST=Path('validation_artifacts/stage1_issues_20261108')
assert (ROOT/'all_training_evaluation_complete').exists(), 'Training, evaluation or video work remains'
assert (ROOT/'baseline_supplement_complete').exists(), 'Baseline challenge/symmetry supplement remains'
DEST.mkdir(exist_ok=True)
state=json.loads((ROOT/'training_state.json').read_text())
for p in ROOT.glob('*.json'): shutil.copy2(p,DEST/p.name)
shutil.copy2(ROOT/'historical_ft5_invocation.sh',DEST/'historical_ft5_invocation.sh')
for name,entry in state.items():
    d=DEST/'models'/name; d.mkdir(parents=True,exist_ok=True)
    shutil.copy2(entry['checkpoint'],d/Path(entry['checkpoint']).name)
    shutil.copytree(Path(entry['run_dir'])/'params',d/'params',dirs_exist_ok=True)
    for e in Path(entry['run_dir']).glob('events*'): shutil.copy2(e,d/e.name)
    accumulator=EventAccumulator(entry['run_dir'],size_guidance={'scalars':0});accumulator.Reload()
    scalar_summary={}
    for tag in accumulator.Tags()['scalars']:
        values=accumulator.Scalars(tag); tail=values[-100:]
        scalar_summary[tag]={'first_step':values[0].step,'last_step':values[-1].step,'first':values[0].value,'last':values[-1].value,'last100_mean':sum(x.value for x in tail)/len(tail)}
    (d/'training_metrics.json').write_text(json.dumps(scalar_summary,indent=2))
parent=Path('logs/rsl_rl/tw_stage1/2026-11-07_16-32-16_tw_stage1_s1_ft4')
d=DEST/'models'/'ft4_parent';d.mkdir(parents=True,exist_ok=True)
shutil.copy2(parent/'model_4498.pt',d/'model_4498.pt')
shutil.copytree(parent/'params',d/'params',dirs_exist_ok=True)
(DEST/'videos').mkdir(exist_ok=True)
for video in (ROOT/'videos').glob('*.mp4'):
    if video.name.startswith(('ft5_video_','robust08_video_')): shutil.copy2(video,DEST/'videos'/video.name)
colors=Path('/root/gpufree-data/datasets/practice9/custom_robot/usd/urdf0711_training_30dof_tw123_yaw_waist_v3_colored.colors.json')
shutil.copy2(colors,DEST/'joint_colors.json')
lines=['# TW stage1 GitHub issues #1–4 results','','SuperMimic remains stopped: historical control pilots passed 0/7 fixed motions. These stage1 velocity-control experiments do not validate SuperMimic.','','## FT5 independent baseline','','128 environments per command per seed, deterministic inference, 20 s. Observation noise and startup mass/friction randomization remain active; external pushes disabled. Fall rate counts trials with any non-timeout termination, not resets per step.','','| Seed | Command | Failures / 128 | XY error m/s | Yaw error rad/s | Foot contact slide m/s | Joint acceleration RMS rad/s² |','|---|---|---:|---:|---:|---:|---:|']
baseline=[]
for seed in [1,2,3]:
    data=json.loads((ROOT/f'ft5_seed{seed}.json').read_text()); baseline+=data['results']
    for r in data['results']:
        m=r['metrics'];lines.append(f"| {seed} | {r['command']} | {r['failed_trials']} | {m['xy_error']:.5f} | {m['yaw_error']:.5f} | {m['foot_contact_slide']:.5f} | {m['joint_acc_rms']:.3f} |")
lines+=['',f"Issue #1 empirical fall-rate gate (<1% for every command and seed): {'PASS' if all(r['fall_rate']<.01 for r in baseline) else 'FAIL'}.",'','Training logs previously reported roughly 0.067 XY and 0.21 yaw error. Training episode statistics and independent fixed-command trial statistics have different distributions and denominators; training termination proportions are not independent trial fall rates.','','## Matched final-checkpoint comparisons','','| Model | Protocol | Worst command/seed fall rate | Mean +/-0.3 yaw error | Mean old-range XY error |','|---|---|---:|---:|---:|']
summary={}
old={'stand','forward_01','forward_02','forward_03','yaw_positive','yaw_negative'}
for name in ['ft5',*state]:
 for protocol in ['nominal','challenge']:
    files=[ROOT/f'{name}_{protocol}_seed{s}.json' for s in [1,2,3]]
    if not all(p.exists() for p in files):continue
    rs=[r for p in files for r in json.loads(p.read_text())['results']]
    yaw=[r['metrics']['yaw_error'] for r in rs if r['command'] in {'yaw_positive','yaw_negative'}]
    xy=[r['metrics']['xy_error'] for r in rs if r['command'] in old]
    result={'worst_fall_rate':max(r['fall_rate'] for r in rs),'mean_yaw_error':sum(yaw)/len(yaw),'old_range_xy_error':sum(xy)/len(xy)}
    summary[name+'_'+protocol]=result
    lines.append(f"| {name} | {protocol} | {result['worst_fall_rate']:.2%} | {result['mean_yaw_error']:.5f} | {result['old_range_xy_error']:.5f} |")
lines+=['','The complete per-command/per-seed records are in the JSON files; pooled means do not replace individual acceptance checks. Range/robustness comparisons use a combined randomization package, so individual randomizer effects are not identified. No wider-range configuration is automatically promoted to production defaults.','','## Videos','','Camera follows translation and heading. Front is along the robot heading; side is perpendicular. Both use the separately precolored exact experiment USD. Joint-to-color legend: [joint_colors.json](joint_colors.json). Stage1 has no HumanML3D source motion; skeleton-reference comparison is not applicable.','']
probes={}
for p in sorted((DEST/'videos').glob('*.mp4')):
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate,duration','-of','json',str(p)]))
    stream=data['streams'][0]
    assert stream['codec_name']=='h264' and int(stream['nb_frames'])==1000, (p,stream)
    assert (stream['width'],stream['height'])==(960,720), (p,stream)
    probes[p.name]=data
    lines.append(f'- [{p.stem}](videos/{p.name})')
assert len(probes)==32, f'Expected 32 full-duration front/side videos, got {len(probes)}'
(DEST/'video_probe.json').write_text(json.dumps(probes,indent=2))
(DEST/'comparison_summary.json').write_text(json.dumps(summary,indent=2))
lines+=['','## Execution evidence','','Original console logs and intermediate checkpoints remain on the remote data disk. Diagnostic mixed-environment preliminary results were isolated and excluded. Formal runs use the original Isaac Lab environment; the launcher clears inherited PYTHONPATH. Per the user decision on 2026-11-08, Notion replaces Linear synchronization. The canonical project record is https://www.notion.so/403325d8231d8172995bf19510b8d760; AGENTS.md and GitHub issue #4 record this decision.','']

lines+=['','## Interpretation and acceptance','','Exploration bonus 0.001 raises the final-100-iteration policy mean std from about 0.023–0.024 (control) to 0.059 for both training seeds. This is evidence against continued std collapse in those arms; it does not by itself prove robust control. Increasing yaw reward further lowers independent turn error. Training command distributions and action noise differ from deterministic fixed-command evaluation, so their error values are not interchangeable.','','| Training seed | Entropy yaw-error reduction vs control | Yaw-reward reduction vs control |','|---|---:|---:|']
for seed in [1,2]:
    c=summary[f'control_s{seed}_nominal']['mean_yaw_error']
    e=summary[f'entropy_s{seed}_nominal']['mean_yaw_error']
    y=summary[f'yaw_s{seed}_nominal']['mean_yaw_error']
    lines.append(f'| {seed} | {(1-e/c):.2%} | {(1-y/c):.2%} |')
lines+=['','All entropy/yaw arms still fail the strict per-command/per-seed <1% fall gate. Historical FT5 re-inflated std to 0.3 on resume; the matched new arms deliberately preserve FT4 std and optimizer. Historical FT5 comparisons therefore include exploration-initialization differences; causal reward comparisons use the matched new arms.','','## Range and robustness outcomes','','| Model | Protocol | 0.3 m/s falls (seeds 1/2/3) | 0.5 m/s falls | 0.8 m/s falls |','|---|---|---|---|---|']
for name in ['ft5','range05','robust05','range08','robust08']:
    for protocol in ['nominal','challenge']:
        rows=[x for seed in [1,2,3] for x in json.loads((ROOT/f'{name}_{protocol}_seed{seed}.json').read_text())['results']]
        cells=['/'.join(str(x['failed_trials']) for x in rows if x['command']==cmd) for cmd in ['forward_03','forward_05','forward_08']]
        lines.append(f'| {name} | {protocol} | '+ ' | '.join(cells)+' |')
lines+=['','Each denominator is 128 trials. Wider-range training substantially improves deterministic nominal tracking, and the combined robustness package improves the matched challenge outcomes. The final robust08 policy still has material challenge failures; wider command coverage is not a claim of perturbation acceptance. Defaults remain unchanged. The challenge protocol combines startup friction/mass changes, 0/20 ms action delay, observation noise and repeated +/-0.3 m/s velocity pushes. This experiment does not isolate which individual randomizer or mechanical factor causes a failure.','','## Left/right gait diagnostics','','| Model | Command | Mean left contact fraction | Mean right contact fraction | Left acceleration RMS | Right acceleration RMS |','|---|---|---:|---:|---:|---:|']
for name in ['ft5','control_s1','entropy_s1','yaw_s1','robust08']:
    rows=[x for seed in [1,2,3] for x in json.loads((ROOT/f'{name}_nominal_seed{seed}.json').read_text())['results']]
    for cmd in ['forward_02','forward_03']:
        ms=[x['metrics'] for x in rows if x['command']==cmd]
        vals=[sum(m[k] for m in ms)/len(ms) for k in ['left_contact_duty','right_contact_duty','left_joint_acc_rms','right_joint_acc_rms']]
        lines.append(f'| {name} | {cmd} | '+' | '.join(f'{v:.5f}' for v in vals)+' |')
lines+=['','Left/right contact duty and acceleration differences identify observable gait asymmetry; they do not establish a hardware or reward-causality diagnosis. Turn commands naturally require asymmetric motion. Full per-foot height, root-height, contact-slide and joint-acceleration metrics remain in the JSON records.','','## Visual validation','','32 videos passed H.264, 960x720, 50fps and 1000-frame checks. Five timestamps per video (160 sampled frames) passed nonblack and colored-body framing checks. A 3x3 opening removes isolated render noise from the colored-pixel mask; framing checks do not replace visual inspection. Contact sheets of all baseline/final commands were inspected for colors, full body and attached visible parts. Camera follows translation and heading. No runtime material binding is used. Initial black-frame videos remain excluded in the remote preliminary directory.']
for n in ['ft5_video_contact_sheet.png','robust08_video_contact_sheet.png']:
    shutil.copy2(ROOT/n,DEST/n)

(DEST/'README.md').write_text('\n'.join(lines)+'\n')
manifest={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file() and p.name!='sha256.json'}
(DEST/'sha256.json').write_text(json.dumps(manifest,indent=2))
print('Archived',len(manifest),'files; review README and videos before commit')
