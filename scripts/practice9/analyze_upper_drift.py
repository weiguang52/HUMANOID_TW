"""Post-hoc phase-aligned diagnostics; associations are not causal evidence."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/root/gpufree-data/datasets/practice9/perturbation_20261025')
O=Path('/root/gpufree-data/datasets/practice9/upper_drift_analysis_20261025')
O.mkdir(exist_ok=True)
def yaw(q):
 w,x,y,z=q.T;return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
def analyze(path):
 a=np.load(path);n=len(a['anchor_actual']);dt=float(a['control_dt']);t=np.arange(n)*dt;step=a['control_step']
 names=a['joint_names'].tolist();arms=[i for i,s in enumerate(names) if any(k in s for k in ['shoulder','elbow','wrist'])]
 contact=np.linalg.norm(a['force'],axis=-1)>1.
 footspeed=np.linalg.norm(a['foot_velocity'][...,:2],axis=-1)
 def agg(x):return np.array([x[step==i].mean() for i in range(n)])
 contact_speed=agg(np.where(contact,footspeed,0).sum(-1)/np.maximum(contact.sum(-1),1))
 arm_speed=np.sqrt((a['reference_qd'][:,arms]**2).mean(-1))
 arm_error=agg(np.sqrt(((a['q'][:,arms]-a['reference_q'][step][:,arms])**2).mean(-1)))
 target_error=agg(np.sqrt(((a['q'][:,arms]-a['target_q'][:,arms])**2).mean(-1)))
 xy=a['anchor_actual'][:,:2];disp=(xy-xy[50])*1000
 upper=np.linalg.norm(a['upper_actual']-a['upper_reference'],axis=-1).mean(-1)*1000
 heading=np.rad2deg(np.arctan2(np.sin(yaw(a['pelvis_actual_quat'])-yaw(a['pelvis_reference_quat'])),np.cos(yaw(a['pelvis_actual_quat'])-yaw(a['pelvis_reference_quat']))))
 valid=t>=1
 sat=np.abs(a['torque'])>=.99*a['effort_limits'];pvalid=t[step]>=1
 windows=[]
 for start,end in [(1,5),(5,10),(10,15),(15,20)]:
  mask=(t>=start)&(t<end);pm=mask[step];ii=np.flatnonzero(mask)
  if not len(ii):continue
  windows.append({'start':start,'end':end,'xy_increment_mm':(xy[ii[-1]]-xy[ii[0]]).tolist(),'reference_arm_speed_rms':float(arm_speed[mask].mean()),'upper_relative_error_mm':float(upper[mask].mean()),'heading_abs_mean_deg':float(abs(heading[mask]).mean()),'contact_link_speed_mm_s':float(footspeed[pm][contact[pm]].mean()*1000),'double_contact_fraction':float(contact[pm].all(-1).mean()),'individual_contact_fraction':contact[pm].mean(0).tolist(),'foot_height_range_mm':(np.ptp(a['foot_position'][pm,:,2],axis=0)*1000).tolist(),'foot_xy_displacement_mm':((a['foot_position'][pm][-1,:,:2]-a['foot_position'][pm][0,:,:2])*1000).tolist()})
  windows[-1]['xy_increment_mm']=(1000*np.array(windows[-1]['xy_increment_mm'])).tolist()
 report={'net_xy_mm':float(np.linalg.norm(disp[-1])),'upper_relative_error_mm':float(upper[valid].mean()),'reference_arm_speed_rms':float(arm_speed[valid].mean()),'arm_reference_error_rms_rad':float(arm_error[valid].mean()),'arm_target_error_rms_rad':float(target_error[valid].mean()),'arm_target_reference_rms_rad':float(np.sqrt(((a['target_q'][pvalid][:,arms]-a['reference_q'][step[pvalid]][:,arms])**2).mean(-1)).mean()),'contact_link_speed_mm_s':float(footspeed[pvalid][contact[pvalid]].mean()*1000),'arm_torque_saturation_fraction':float(sat[pvalid][:,arms].mean()),'all_torque_saturation_fraction':float(sat[pvalid].mean()),'windows':windows}
 return report,{'t':t,'disp':disp,'arm_speed':arm_speed,'upper':upper,'contact_speed':contact_speed*1000,'heading':heading}
results={};series={}
for model in ['original','baseline']:
 for mid in [2,3]:
  key=f'{model}/motion_{mid}'
  results[key],series[key]=analyze(R/model/'batch_42'/f'motion_{mid}_v00.npz')
# Nonzero cases are paired sensitivity checks, not independent population samples.
perturbed={}
for model in ['original','baseline']:
 for seed in [42,123,2026]:
  for speed in [5,10]:
   k=f'{model}/{seed}/v{speed:02d}';perturbed[k],_=analyze(R/model/f'batch_{seed}'/f'motion_3_v{speed:02d}.npz')
(O/'analysis.json').write_text(json.dumps({'zero':results,'perturbed_upper':perturbed,'limits':['Foot-link velocity is not contact-point slip; rolling cannot be excluded.','Upper position error measured in each torso heading frame, not world.','Reference captured before step, joint physics samples within step; joint errors have up to20ms alignment offset.','No COM/support polygon or ground-truth contact-point telemetry.','One synthetic upper trajectory and two checkpoints; no causal conclusion.']},indent=2))
fig,axes=plt.subplots(5,1,figsize=(11,12),sharex=True)
for model in ['original','baseline']:
 d=series[f'{model}/motion_3'];t=d['t']
 axes[0].plot(t,np.linalg.norm(d['disp'],axis=-1),label=model)
 axes[1].plot(t,d['arm_speed'],label=model)
 axes[2].plot(t,d['upper'],label=model)
 axes[3].plot(t,d['contact_speed'],label=model)
 axes[4].plot(t,d['heading'],label=model)
for ax,label in zip(axes,['XY displacement from1s (mm)','Reference arm speed RMS (rad/s)','Torso-relative upper error (mm)','Contact foot-link speed (mm/s)','Pelvis heading error (deg)']):
 ax.set_ylabel(label);ax.grid(alpha=.3);ax.legend();ax.set_xlim(1,20)
axes[-1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(O/'upper_timeline.png',dpi=150);fig.savefig(O/'upper_timeline.jpg',dpi=90);plt.close(fig)
print(json.dumps({k:{a:b for a,b in v.items() if a!='windows'} for k,v in results.items()},indent=2))
