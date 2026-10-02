"""Package final candidates and physically measured comparison, on the server."""
import json, shutil, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_s2_joint_commands import read, highpass
ROOT=Path('/root/gpufree-data/datasets/practice9/s2_control_fix')
OUT=Path('validation_artifacts/s2_control_fix'); OUT.mkdir(parents=True,exist_ok=True)
IDS=['000801','006680','010407','010782']
all_data={}
for name,path in [('before',Path('/root/gpufree-data/datasets/practice9/s2_trials/smooth')),('after',ROOT/'acc_train')]:
    all_data[name]=[read(path/(i+'.telemetry.npz')) for i in IDS]
names=all_data['before'][0]['names']; metrics={}
for name,ds in all_data.items():
    metrics[name]={}
    for key in ['band','actual_band']:
        a=np.concatenate([d[key] for d in ds]);metrics[name][key]=np.sqrt(np.nanmean(a*a,axis=0))
    metrics[name]['overall_actual']=float(np.sqrt(np.nanmean(np.concatenate([d['actual_band'] for d in ds])**2)))
order=np.argsort(-metrics['before']['actual_band']); fig,axes=plt.subplots(1,2,figsize=(17,12),layout='constrained')
for ax,key,title in zip(axes,['band','actual_band'],['Command target','Physical joint angle']):
    y=np.arange(len(names))
    for offset,name,color in [(-.18,'before','tab:orange'),(.18,'after','tab:blue')]:
        ax.barh(y+offset,metrics[name][key][order],height=.35,label=name,color=color)
    ax.set_yticks(y,[names[j] for j in order]);ax.invert_yaxis();ax.set_xlabel('2-8 Hz RMS (degrees)');ax.set_title(title);ax.legend();ax.grid(axis='x',alpha=.2)
fig.suptitle('Walking: 4 clips, seed42; reset/filter-edge samples excluded');fig.savefig(OUT/'joint_comparison.png',dpi=130);plt.close(fig)
fig,axes=plt.subplots(4,1,figsize=(13,10),layout='constrained')
for ax,j in zip(axes,order[:4]):
    for name in ['before','after']:
        d=all_data[name][0];t=np.arange(len(d['target']))/d['rate'];mask=(t>=2)&(t<=5)
        ax.plot(t[mask],d['band'][mask,j],label=name,lw=.9)
    ax.set_title(names[j]);ax.set_ylabel('2-8Hz command deg');ax.grid(alpha=.2)
axes[0].legend();axes[-1].set_xlabel('seconds / clip000801');fig.savefig(OUT/'main_joint_commands.png',dpi=140);plt.close(fig)
summary={'selected':'acc_train','walking':{n:{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in m.items()} for n,m in metrics.items()},'joint_names':names,'camera':{}}
for v in ['rate_train','acc_train']:
    src=ROOT/v;dst=OUT/v;dst.mkdir(exist_ok=True);checkpoint=Path((src/'final_checkpoint').read_text().strip())
    shutil.copy2(checkpoint,dst/checkpoint.name)
    shutil.copytree(checkpoint.parent/'params',dst/'params',dirs_exist_ok=True)
    for pattern in ['*.evaluation.json','*.summary.json','side.encoding.json','side.evaluation.json.camera.npz','walk_side_stable_camera.mp4','implementation_commit']:
        for p in src.glob(pattern):shutil.copy2(p,dst/p.name)
    c=np.load(src/'side.evaluation.json.camera.npz');valid=~c['reset'];valid[0]=False
    for shift in (-1,1):valid &= ~np.roll(c['reset'],shift)
    cam=highpass(c['camera_origin'],valid,1/float(c['dt']),[2,8]);body=highpass(c['robot_origin'],valid,1/float(c['dt']),[2,8])
    boundaries=np.r_[0,np.flatnonzero(c['reset']),len(valid)];zmax=max([float(np.ptp(c['camera_origin'][a:b,2])) for a,b in zip(boundaries[:-1],boundaries[1:]) if b>a])
    summary['camera'][v]={'max_segment_vertical_range_m':zmax,'camera_xy_2_8_rms_m':float(np.sqrt(np.nanmean(cam[:,:2]**2))),'robot_xy_2_8_rms_m':float(np.sqrt(np.nanmean(body[:,:2]**2))),'resets':int(c['reset'].sum())}
shutil.copy2(ROOT/'comparison.json',OUT/'comparison.json')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
report=json.load(open(ROOT/'comparison.json'))
lines=['# S2 control fix final candidates','','Selected candidate: **acc_train**. Both trials warm-start S2 smooth, 600 additional PPO iterations, seed42, 1024 environments. Joint target velocity is capped at 50% actuator limits, with unchanged PD gains and no target low-pass. acc_train uses acceleration penalty -2e-6; rate_train uses -1e-7.','', 'All 7 acc_train clips completed without abnormal termination in deterministic seed42 evaluation. rate_train had two end-effector termination events on weight-shift and is not selected. Multi-seed/noise robustness is not yet accepted.','', '|Walking physical 2-8 Hz RMS|Before|After|Reduction|','|---|---:|---:|---:|']
b=metrics['before']['overall_actual'];a=metrics['after']['overall_actual'];lines.append(f'|All joints pooled, degrees|{b:.3f}|{a:.3f}|{100*(1-a/b):.1f}%|')
for j in order[:6]:
    b=metrics['before']['actual_band'][j];a=metrics['after']['actual_band'][j];lines.append(f'|{names[j]}|{b:.3f}|{a:.3f}|{100*(1-a/b):.1f}%|')
lines+=['','Walking body-position tracking error (unweighted mean over 4 clip metrics): '+', '.join(f'{v}={np.mean([r["body_error_m"] for r in report[v][:4]]):.5f} m' for v in ['reference_smooth','acc_train'])+'.','', 'Limitations: oscillation is reduced, not eliminated. Some ankle/yaw joints worsen; see all-joint plot and JSON. Stand RMS increases from .084 to .246 degrees, weight-shift .263 to .468, wave .288 to .276. Complete motion-end alone does not establish faithful imitation or motor feasibility.','', 'Camera follows low-pass horizontal translation only (tau=.5s), with fixed height and direction. All continuous segments have zero camera vertical range. Camera traces are included; reset cuts can occur. Videos: H264, 1280x720, 50fps, 999 frames (~20s). Physical jitter above was computed from telemetry, independently of rendering.','', 'Keep each final model_11597.pt alongside its params/control_runtime.json; playback restores the target-rate behavior. External deployment must implement the same rate limiter. Intermediate checkpoints are intentionally excluded.','', 'Plots exclude resets and bandpass edge samples; actual angles sampled at200Hz, targets at50Hz. Band 2-8Hz captures the observed ~4.3Hz oscillation and is not a universal perceptual threshold.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
(OUT/'sha256.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='sha256.json'},indent=2)+'\n')
print('\n'.join(lines));print(summary['camera'])
