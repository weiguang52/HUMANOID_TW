"""Describe all original amplitude groups without fitting or changing gates."""
import argparse,hashlib,json,math
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir()
r=a.root;source=[];reports=[]
for name,relative in [('parent','standing_wm_calibration_v1/after.json'),('control','dynamic_h1_mixture_fit_v1/control/after.json'),('candidate','dynamic_h1_mixture_fit_v1/candidate/after.json')]:
 path=r/relative;d=json.loads(path.read_text());source.append(dict(file=relative,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 assert len(d['rows'])==16
 for split in ['fit','validation']:
  for amplitude in ['0.00','0.02','0.05','0.10']:
   rows=[x for x in d['rows'] if x['split']==split and x['mode']=='amplitude_'+amplitude]
   assert len(rows)==2
   for h in ['1','8','32']:
    metrics=[x['world_model'][h] for x in rows]
    assert all(x['windows']>0 and x['finite_fraction']==1 for x in metrics)
    n=sum(x['windows'] for x in metrics)
    entry=dict(model=name,split=split,amplitude=amplitude,horizon=int(h),windows=n,traces=[x['trace'] for x in rows],original_termination_reasons=[x['termination_reasons'] for x in rows])
    for key in ['root_linear_velocity_rmse_m_s','root_angular_velocity_rmse_rad_s','joint_velocity_rmse_rad_s']:
     entry[key]=math.sqrt(sum(x['windows']*x[key]**2 for x in metrics)/n)
    reports.append(entry)
out=dict(sources=source,reports=reports,scope='All16 original traces and all4 original forcing amplitudes, exact window-weighted RMSE from frozen reports. Unequal episode lengths and original terminations retained. Descriptive only, no fitting, new gates, filtering, checkpoint choice or policy acceptance.')
(a.output/'audit.json').write_text(json.dumps(out,indent=2))
for x in reports:
 if x['split']=='validation' and x['horizon']==1:print(x['model'],x['amplitude'],x['windows'],x['root_angular_velocity_rmse_rad_s'])
