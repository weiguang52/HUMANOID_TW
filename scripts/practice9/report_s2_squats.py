"""Report diagnostic squat replay without masking reset failures."""
from pathlib import Path
import json,hashlib,shutil,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SRC=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/root/gpufree-data/datasets/practice9/s2_squat_test');OUT=Path(sys.argv[2]) if len(sys.argv)>2 else Path('validation_artifacts/s2_squats');OUT.mkdir(parents=True,exist_ok=True)
rows=[];fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for row,clip in enumerate(['000890','001240']):
 e=json.load(open(SRC/(clip+'.evaluation.json')));d=np.load(SRC/(clip+'.telemetry.npz'));ends=np.r_[np.flatnonzero(np.diff(d['control_step'])),len(d['control_step'])-1];events=np.flatnonzero(d['terminal']);n=int(events[0]) if len(events) else len(d['action']);dt=float(d['control_dt']);t=np.arange(n)*dt
 actual=d['q'][ends][:n];ref=d['reference_q'][:n];names=d['joint_names'].tolist()
 for jname,color in [('left_knee_pitch_joint','tab:blue'),('right_knee_pitch_joint','tab:orange')]:
  j=names.index(jname);axes[row,0].plot(t,np.rad2deg(actual[:,j]),color=color,label=jname+' actual');axes[row,0].plot(t,np.rad2deg(ref[:,j]),'--',color=color,label='reference '+jname)
 h=d['root_height'][ends][:n];axes[row,1].plot(t,h);axes[row,1].set_ylabel('actual root height / m')
 for ax in axes[row]:ax.set_title(clip+' / before first termination');ax.set_xlabel('seconds');ax.grid(alpha=.2)
 axes[row,0].set_ylabel('knee angle / deg');axes[row,0].legend(fontsize=7)
 m=json.load(open(SRC/(clip+'.manifest.json')))['motions'][0]
 rows.append(dict(clip=clip,steps=e['steps'],first_termination_s=float((n+1)*dt) if len(events) else None,termination=e['termination_counts'],root_height_start_m=float(h[0]),root_height_min_before_termination_m=float(h.min()),quality_pass=m['quality_pass'],quality_reasons=m['retarget_quality']['reasons'],checkpoint=e['checkpoint'],body_position_error_m=e['motion_metrics']['error_body_pos']))
 for suffix in ['evaluation.json','encoding.json','manifest.json','side.mp4']:
  shutil.copy2(SRC/(clip+'.'+suffix),OUT/(clip+'.'+suffix))
fig.savefig(OUT/'first_attempt.png',dpi=140);plt.close(fig)
(OUT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['# Standing-to-squat diagnostic test','','Latest long_velocity/model_15596.pt; physical contact + PD, seed42, observation noise disabled. Source HumanML3D000890 (squat then stand) and001240 (squat then stand). Stable side camera with fixed height and smoothed XY.','', 'Both references preserve tw44_table_v2 and pass FK conversion, but fail retarget quality due to prolonged joint-limit proximity. Both knees reach -90 degrees. Explicit unsafe-motion flag is confined to diagnostic replay; no training manifest or joint limit is changed. These results test the current policy plus imperfect reference, not policy capability in isolation.','', '|Clip|First abnormal termination (s)|Termination counts|','|---|---:|---|']
for r in rows:lines.append(f'|{r["clip"]}|{r["first_termination_s"]}|{r["termination"]}|')
lines+=['','Neither motion completes successfully. Videos include automatic episode resets; discontinuities at reset must not be interpreted as recovery or camera jitter. first_attempt.png stops before the first termination. No sitting-on-floor capability is claimed. The model was trained on walking, stand, weight-shift and wave, not these squat motions.','', 'Next step: build a feasible shallow-squat reference within joint limits and verified foot support, train a stand-to-shallow-squat curriculum before increasing depth. Ground sitting needs its own contact model/reward/termination audit and cannot be inferred from this test.']
lines += ['', 'Grounding caveat: original references start with soles 4.4cm/10.9cm above ground. Original replay cannot isolate controller capability. The grounded subdirectory is a derived diagnostic with each frame lowest sole at1mm and body linear velocities recomputed; joint angles unchanged. This removes initial drop but does not prove foot locking or full reference feasibility.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
(OUT/'sha256.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='sha256.json'},indent=2)+'\n')
print(json.dumps(rows,indent=2))
