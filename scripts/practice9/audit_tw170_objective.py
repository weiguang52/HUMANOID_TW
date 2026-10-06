"""Offline explanatory audit; no phase edits or simulator reward claims."""
from pathlib import Path
import json,numpy as np
root=Path('/root/gpufree-data/datasets/practice9/tw154_v1/validation/rate')
rows=[]
for p in root.glob('*.seed*.npz'):
 a=np.load(p);end=int(np.flatnonzero(a['terminal'])[0]) if a['terminal'].any() else len(a['terminal'])
 cv=np.arange(50,end)
 if len(cv)<100:continue
 ends=np.r_[np.flatnonzero(np.diff(a['control_step'])),len(a['control_step'])-1]
 names=a['joint_names'].tolist();knees=[names.index(s+'_knee_pitch_joint') for s in ['left','right']]
 legs=[i for i,n in enumerate(names) if any(t in n for t in ['hip_','knee_','ankle_'])]
 q=a['q'][ends][cv];target=a['target_q'][ends][cv];ref=a['reference_q'][cv]
 known=a['reference_known'][cv].astype(bool);contact=a['reference_contact'][cv].astype(bool);mask=known&~contact
 e=ref[:,knees]-q[:,knees]
 new=((np.sqrt(1+(e/.35)**2)-1)*mask).mean(1)
 old=np.exp(-np.mean((ref[:,legs]-q[:,legs])**2,axis=1)/.35**2)
 delta=np.diff(target[:,knees],axis=0)
 flips=((delta[1:]*delta[:-1])<0)&(np.abs(delta[1:])>.001)&(np.abs(delta[:-1])>.001)
 def band(x):
  x=x[:,knees];t=np.arange(len(x));x=x-np.stack([np.polyval(np.polyfit(t,x[:,j],1),t) for j in range(2)],1)
  f=np.fft.rfftfreq(len(x),.02);z=np.fft.rfft(x,axis=0);weights=np.full(len(f),2.);weights[0]=1
  if len(x)%2==0:weights[-1]=1
  return np.rad2deg(np.sqrt(np.sum(weights[(f>=8)&(f<=25),None]*np.abs(z[(f>=8)&(f<=25)])**2,axis=0))/len(x)).tolist()
 rows.append(dict(replay=p.stem,leg_exp_mean=float(old.mean()),swing_known_fraction=float(mask.mean()),
  added_cost_mean=float(new.mean()),added_weighted_cost_mean=float(-.5*new.mean()),
  knee_target_reversal_fraction=float(flips.mean()),actual_8_25hz_deg=band(q),target_8_25hz_deg=band(target)))
out=dict(rows=rows,note='Control-step last substep samples, first episode after 1s, >=2s window. Offline reward proxies exclude dt. 8-25Hz includes intentional motion; not proof of instability. Limited known labels preserved.')
Path('validation_artifacts/tw170_setup/objective_audit.json').write_text(json.dumps(out,indent=2))
print('audit replays',len(rows),'old leg exp mean',np.mean([r['leg_exp_mean'] for r in rows]),'added weighted',np.mean([r['added_weighted_cost_mean'] for r in rows]))
print('target reversals',np.mean([r['knee_target_reversal_fraction'] for r in rows]),'actual HF',np.mean([r['actual_8_25hz_deg'] for r in rows]),'target HF',np.mean([r['target_8_25hz_deg'] for r in rows]))
