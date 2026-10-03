"""Summarize matched holdout replays; completed jobs only, never imply acceptance."""
from pathlib import Path
import json
import numpy as np
ROOT=Path('/root/gpufree-data/datasets/practice9')
STATE=ROOT/'tw75_contact_v1'

def summarize(directory):
    rows=[]
    for file in sorted(directory.glob('*.seed*.json')):
        if file.name.endswith('.summary.json'):continue
        report=json.loads(file.read_text());path=file.with_suffix('.npz')
        if not path.exists():continue
        a=np.load(path);step=a['control_step'];dt=float(a['physics_dt'])
        valid=step>=50
        for t in np.flatnonzero(a['terminal']):valid &= np.abs(step-t)>25
        if not valid.any():continue
        f=a['force'][:,:,2];v=np.linalg.norm(a['foot_velocity'][:,:,:2],axis=-1)
        contact=f>1.;pair=valid[1:]&valid[:-1]
        counts=report['termination_counts']
        row={'replay':file.stem,'termination_counts':counts,
             'clean_motion_end':counts.get('motion_end',0)>0 and sum(n for k,n in counts.items() if k not in ['motion_end','time_out'])==0,
             'body_error_m':report['motion_metrics']['error_body_pos'],
             'raw_switches_per_s':((np.diff(contact.astype(int),axis=0)!=0)*pair[:,None]).sum(0).astype(float).tolist(),
             'contact_slip_mean_m_s':[float(v[valid&contact[:,i],i].mean()) if np.any(valid&contact[:,i]) else None for i in range(2)]}
        row['raw_switches_per_s']=[x/(valid.sum()*dt) for x in row['raw_switches_per_s']]
        rows.append(row)
    return rows

if __name__=='__main__':
    out={'note':'Limited 6 holdout clips x 3 seeds; raw Fz threshold 1N, reset neighborhoods excluded. Not full acceptance.',
         'baseline_seed123':summarize(ROOT/'tw75_train_v1/seed123')}
    for name in ['control','contact']:
        p=STATE/name
        if (p/'status').exists() and (p/'status').read_text().strip()=='completed':out[name]=summarize(p)
    (STATE/'comparison.json').write_text(json.dumps(out,indent=2))
    print({k:len(v) for k,v in out.items() if isinstance(v,list)})
