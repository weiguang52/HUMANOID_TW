"""Post-training review: first episode, exclude 1s, no contact relabeling."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/root/gpufree-data/datasets/practice9')
OUT=Path('validation_artifacts/tw154_review');OUT.mkdir(exist_ok=True)
rows=[]
skipped=[]
for variant in ['path','rate']:
 report=json.loads(Path(f'validation_artifacts/tw154_flat_walk/{variant}/comparison.json').read_text())
 for r in report['rows']:
  a=np.load(ROOT/'tw154_v1'/'validation'/variant/(r['replay']+'.npz'))
  end=int(np.flatnonzero(a['terminal'])[0]) if a['terminal'].any() else len(a['terminal'])
  cv=np.arange(50,end)
  ends=np.r_[np.flatnonzero(np.diff(a['control_step'])),len(a['control_step'])-1]
  names=a['joint_names'].tolist();ji=[i for i,n in enumerate(names) if 'hip_pitch' in n or 'knee_pitch' in n]
  q=a['q'][ends];target=a['target_q'][ends];ref=a['reference_q'];dt=float(a['control_dt'])
  vel=np.diff(target,axis=0)/dt;cap=1.5 if variant=='path' else 2.25
  ar=a['anchor_reference'][cv,:2];ac=a['anchor_actual'][cv,:2]
  d=r['step_diagnostics']
  if d.get('insufficient_samples'):
   skipped.append(dict(variant=variant,replay=r['replay'],reason='first episode <= 1s'));continue
  rows.append(dict(variant=variant,replay=r['replay'],category=r['category'],clean=r['clean_motion_end'],
   knee_ratio=float(np.mean(d['knee_actual_span_deg'])/np.mean(d['knee_reference_span_deg'])),
   foot_z_ratio=float(np.mean(d['foot_actual_z_span_mm'])/np.mean(d['foot_reference_z_span_mm'])),
   swing_contact=d['known_swing_contact_fraction'],
   path_rmse_m=float(np.sqrt(np.mean(np.sum((ar-ac)**2,axis=1)))),
   net_vector_error_m=float(np.linalg.norm((ac[-1]-ac[0])-(ar[-1]-ar[0]))),
   cap_fraction=float(np.mean(np.abs(vel[cv[:-1]][:,ji])>=cap*.99)),
   leg_pitch_reference_rms_deg=float(np.rad2deg(np.sqrt(np.mean((q[cv][:,ji]-ref[cv][:,ji])**2)))),
   leg_pitch_pd_rms_deg=float(np.rad2deg(np.sqrt(np.mean((target[cv][:,ji]-q[cv][:,ji])**2)))),
   knee_left_deg=d['knee_actual_span_deg'][0],knee_right_deg=d['knee_actual_span_deg'][1]))
  if r['replay']=='000832.seed42':
   fig,ax=plt.subplots(4,1,figsize=(12,11),constrained_layout=True)
   t=np.arange(end)*dt
   for side,color in [('left','tab:blue'),('right','tab:orange')]:
    j=names.index(side+'_knee_pitch_joint')
    ax[0].plot(t,np.rad2deg(ref[:end,j]),'--',color=color,label=side+' reference')
    ax[0].plot(t,np.rad2deg(q[:end,j]),color=color,label=side+' actual')
    ax[1].plot(t,np.rad2deg(target[:end,j]-q[:end,j]),color=color,label=side)
   for f,color in enumerate(['tab:blue','tab:orange']):
    ax[2].plot(t,a['foot_position'][ends][:end,f,2]*1000,color=color,label=['left','right'][f])
   ax[3].plot(t,np.linalg.norm(a['anchor_actual'][:end,:2]-a['anchor_reference'][:end,:2],axis=1)*100)
   for axy,label in zip(ax,['Knee angle (deg)','PD target - actual (deg)','Foot-link world Z (mm)','Horizontal path error (cm)']):axy.set_ylabel(label);axy.grid(alpha=.3)
   ax[0].legend(ncol=4);ax[2].legend();ax[3].set_xlabel('Simulation time (s)')
   fig.suptitle(variant+' / 000832 / seed42 / first episode')
   fig.savefig(OUT/f'{variant}_000832.png',dpi=130);plt.close(fig)
summary={}
for v in ['path','rate']:
 summary[v]={}
 for cat in sorted({r['category'] for r in rows}):
  rr=[r for r in rows if r['variant']==v and r['category']==cat]
  keys=['path_rmse_m','net_vector_error_m','cap_fraction','knee_left_deg','knee_right_deg','leg_pitch_reference_rms_deg','leg_pitch_pd_rms_deg']
  if cat=='flat_forward':keys+=['knee_ratio','foot_z_ratio','swing_contact']
  summary[v][cat]={k:float(np.mean([r[k] for r in rr if r[k] is not None])) for k in keys}
old=ROOT/'tw151_v1'/'progress_metrics.json'
result=dict(summary=summary,rows=rows,skipped=skipped,previous_tw151=json.loads(old.read_text())['summary'],
 limitations=['Single training seed; 3 evaluation seeds are not 3 training seeds.',
 'First episode excluding 1s. Amplitude ratios use P95-P5; standing ratios suppressed due tiny reference amplitude.',
 'Contact known mask incomplete. Foot-link speed includes rolling; Z amplitude is not sole clearance.',
 '2-8Hz includes intentional motion. Rate occupancy does not establish causality.',
 'Video contact sheets are temporal samples, not exhaustive frame-by-frame visual inspection. Source and robot are not synchronized.'])
(OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
