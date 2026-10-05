"""Partial results are explicitly labelled; no completion implied by no resets."""
import json,sys
from pathlib import Path
import numpy as np
from step_diagnostics import diagnose
root=Path('/root/gpufree-data/datasets/practice9/tw151_v1')
m=json.load(open(root/'validation/manifest.json'))['motions'];rows=[]
for variant in ('size','residual'):
 for p in sorted((root/'validation'/variant).glob('*.seed*.json')):
  if p.name.endswith('.summary.json'):continue
  r=json.load(open(p));i=next(i for i,x in enumerate(m) if x['id']==p.name.split('.')[0])
  d=diagnose(p,m[i],sum(x['frames'] for x in m[:i]))
  a=np.load(p.with_suffix('.npz'));ji=[j for j,n in enumerate(a['joint_names']) if 'hip_pitch' in n or 'knee_pitch' in n]
  st=a['control_step'];ends=np.r_[np.flatnonzero(st[1:]!=st[:-1]),len(st)-1]
  velocity=np.diff(a['target_q'][ends][:,ji],axis=0)/float(a['control_dt'])
  valid=~a['terminal'][1:]&~a['terminal'][:-1]
  end=int(np.flatnonzero(a['terminal'])[0]) if a['terminal'].any() else len(a['terminal'])
  cv=np.arange(50,end)
  if len(cv):
   ar=a['anchor_reference'][cv[-1],:2]-a['anchor_reference'][cv[0],:2]
   ac=a['anchor_actual'][cv[-1],:2]-a['anchor_actual'][cv[0],:2]
   d['horizontal_net_vector_error_m']=float(np.linalg.norm(ac-ar))
  counts=r['termination_counts']
  rows.append(dict(variant=variant,id=m[i]['id'],seed=r['seed'],
   clean=counts.get('motion_end',0)>0 and not any(v for k,v in counts.items() if k not in ('motion_end','time_out')),
   target_rate_cap_fraction=float((np.abs(velocity[valid])>=1.5*.99).mean()),
   reference_over_target_rate_cap_fraction=float((np.abs(a['reference_qd'][~a['terminal']][:,ji])>1.5).mean()),**d))
summary={}
for v in ('size','residual'):
 rr=[r for r in rows if r['variant']==v]
 summary[v]=dict(replays=len(rr),clean=sum(r['clean'] for r in rr),
  mean_knee_amplitude_ratio=float(np.mean([np.mean(r['knee_actual_span_deg'])/np.mean(r['knee_reference_span_deg']) for r in rr])),
  mean_foot_z_amplitude_ratio=float(np.mean([np.mean(r['foot_actual_z_span_mm'])/np.mean(r['foot_reference_z_span_mm']) for r in rr])),
  mean_horizontal_net_vector_error_m=float(np.mean([r['horizontal_net_vector_error_m'] for r in rr])),
  target_rate_cap_fraction=float(np.mean([r['target_rate_cap_fraction'] for r in rr])))
result=dict(expected_replays=48,completed_replays=len(rows),summary=summary,rows=rows,
 note='Original source labels and geometric screening only. All KIT. Joint-origin Z, not sole clearance. Cap is actuator 3rad/s x runtime0.5; occupancy does not establish causality.')
(root/'progress_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
