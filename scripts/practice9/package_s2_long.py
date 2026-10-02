"""Archive final long-run candidates and explicit robustness failures."""
from pathlib import Path
import json,shutil,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_s2_joint_commands import read,highpass
from summarize_s2_control_fix import summarize,IDS
ROOT=Path('/root/gpufree-data/datasets/practice9/s2_control_fix')
OUT=Path('validation_artifacts/s2_long');OUT.mkdir(parents=True,exist_ok=True)
paths={'smooth':Path('/root/gpufree-data/datasets/practice9/s2_trials/smooth'),'acc_train':ROOT/'acc_train','long_control':ROOT/'long_control','long_velocity':ROOT/'long_velocity'}
report={};pooled={};camera={}
for v,p in paths.items():
 report[v]={'seed42':summarize(p,IDS)}
 ds=[read(p/(i+'.telemetry.npz')) for i in IDS[:4]];names=ds[0]['names']
 pooled[v]={k:np.sqrt(np.nanmean(np.concatenate([d[k] for d in ds])**2,axis=0)).tolist() for k in ['band','actual_band']}
 pooled[v]['all_joint_actual_rms_deg']=float(np.sqrt(np.nanmean(np.concatenate([d['actual_band'] for d in ds])**2)))
 if not v.startswith('long_'):continue
 assert (p/'status').read_text().strip()=='completed',v
 for mode in ['seed123','seed2026','noise123']:
  report[v][mode]=summarize(p/mode,IDS);assert len(report[v][mode])==7
 dst=OUT/v;dst.mkdir(exist_ok=True);ck=Path((p/'final_checkpoint').read_text().strip());shutil.copy2(ck,dst/ck.name);shutil.copytree(ck.parent/'params',dst/'params',dirs_exist_ok=True)
 for f in p.rglob('*.evaluation.json'):
  rel=f.relative_to(p);(dst/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst/rel)
 for name in ['walk_side_stable_camera.mp4','side.encoding.json','side.evaluation.json.camera.npz','source_commit']:
  shutil.copy2(p/name,dst/name)
 c=np.load(p/'side.evaluation.json.camera.npz');valid=~c['reset'];valid[0]=False
 for shift in [-1,1]:valid &= ~np.roll(c['reset'],shift)
 edges=np.r_[0,np.flatnonzero(c['reset']),len(valid)]
 camera[v]={'max_continuous_z_range_m':max(float(np.ptp(c['camera_origin'][a:b,2])) for a,b in zip(edges[:-1],edges[1:]) if b>a)}
 for key in ['camera_origin','robot_origin']:
  a=highpass(c[key],valid,1/float(c['dt']),[2,8]);camera[v][key+'_xy_rms_m']=float(np.sqrt(np.nanmean(a[:,:2]**2)))
order=np.argsort(-np.array(pooled['smooth']['actual_band']));y=np.arange(len(names));fig,axes=plt.subplots(1,2,figsize=(18,13),layout='constrained')
for ax,key,title in zip(axes,['band','actual_band'],['Target angle','Physical joint angle']):
 for i,v in enumerate(paths):ax.barh(y+(i-1.5)*.2,np.array(pooled[v][key])[order],height=.19,label=v)
 ax.set_yticks(y,[names[j] for j in order]);ax.invert_yaxis();ax.set_xlabel('2-8Hz RMS / degrees');ax.set_title(title);ax.legend();ax.grid(axis='x',alpha=.2)
fig.suptitle('Four walking clips / seed42 / valid samples pooled');fig.savefig(OUT/'joint_comparison.png',dpi=130);plt.close(fig)
failures={}
for v in ['long_control','long_velocity']:
 failures[v]=[]
 for mode,rows in report[v].items():
  for r in rows:
   bad={k:n for k,n in r['termination'].items() if k not in ['motion_end','time_out'] and n}
   if bad:failures[v].append(dict(mode=mode,clip=r['clip'],failures=bad))
summary=dict(pooled_walking=pooled,joint_names=names,camera=camera,failures=failures,acceptance='Neither candidate passes all robustness replays; keep previous weights available.')
(OUT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# S2: 4000-iteration controlled continuation','','Both trials start from acc_train/model_11597.pt, use seed42 and1024env. long_control preserves the objective; long_velocity adds mean squared reference-joint velocity error with weight -0.2. PD, target rate (.5 actuator speed), acceleration and action-rate penalties remain unchanged. Each final checkpoint is model_15596.pt.','', '|Candidate|Walking physical 2-8Hz RMS (deg)|Mean body-position error (m)|','|---|---:|---:|']
for v in paths:lines.append(f'|{v}|{pooled[v]["all_joint_actual_rms_deg"]:.3f}|{np.mean([r["body_error_m"] for r in report[v]["seed42"][:4]]):.5f}|')
lines+=['','All 30 joint comparisons are in joint_comparison.png. Lower average oscillation does not prove all joints improved or faithful imitation.','', '## Robustness failures', '', 'Each candidate has 28 replays: seven clips times seed42/123/2026 without observation noise, plus seed123 with noise. These are replay seeds, not independent training seeds. Each replay lasts at least1000steps or one motion length; reset segments may follow failures. Completion after reset does not erase a failure.','']
for v in failures:
 lines.append(f'{v}: {len(failures[v])}/28 replays contain abnormal termination.')
 for r in failures[v]:lines.append(f'- {r["mode"]}/{r["clip"]}: {r["failures"]}')
lines+=['','Neither candidate is promoted as fully accepted. long_velocity is the smoother candidate; full stability remains unresolved. Previous weights are preserved. No motor specification is inferred from these results.','', '## Camera and artifacts','','Camera height/direction fixed, horizontal tracking tau=.5s. Within-segment height range is zero (see summary.json); episode resets can cut. Side videos are H2641280x720/50fps999frames. Joint oscillation is measured from physical telemetry independently of camera motion.','', 'Keep final weights beside params/control_runtime.json. Replay restores target shaping; deployment must implement the same rate limiter. No intermediate checkpoints included. Server shutdown interrupted only robustness evaluation; resume_s2_long_validation.sh verifies existing JSON/telemetry and completes missing outputs.','']
(OUT/'README.md').write_text('\n'.join(lines).rstrip()+'\n')
(OUT/'sha256.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='sha256.json'},indent=2)+'\n')
print('\n'.join(lines));print(camera)
