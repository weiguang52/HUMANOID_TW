"""Audit paired standing evaluations without extending terminated episodes."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--before',type=Path,required=True)
p.add_argument('--after',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert not a.output.exists(), 'Preserve previous evidence'
datasets=[]
for root in [a.before,a.after]:
    d=json.loads((root/'results.json').read_text())
    datasets.append(d)
assert datasets[0]['manifest_sha256']==datasets[1]['manifest_sha256']
assert datasets[0]['seed']==datasets[1]['seed']
assert datasets[0]['max_steps']==datasets[1]['max_steps']
paired=[]
rows=[{(x['id'],x['agent']):x for x in d['rows']} for d in datasets]
assert rows[0].keys()==rows[1].keys()
assert all(len(x)==len(d['rows']) for x,d in zip(rows,datasets))
for key in rows[0]:
    traces=[]
    for root,table in zip([a.before,a.after],rows):
        with np.load(root/table[key]['transition_file']) as z:
            t={k:z[k].copy() for k in z.files}
        assert len(t['state'])==table[key]['steps']
        assert all(np.isfinite(v).all() for v in t.values())
        assert np.array_equal(t['next_state'][:-1],t['state'][1:])
        traces.append(t)
    n=min(len(t['state']) for t in traces)
    summary={'id':key[0],'agent':key[1],'common_steps':n,'common_seconds':n*.02}
    for label,t,table in zip(['before','after'],traces,rows):
        s=t['next_state'][:n]
        q=s[:,3:7];q=q/np.linalg.norm(q,axis=1,keepdims=True)
        delta=s[:,:2]-t['state'][0,:2]
        summary[label]={
            'actual_seconds':len(t['state'])*.02,
            'termination_reasons':table[key]['termination_reasons'],
            'completed_requested_window':table[key]['completed_requested_window'],
            'common_action_abs_max':float(np.abs(t['action'][:n]).max()),
            'common_action_abs_p95':float(np.percentile(np.abs(t['action'][:n]),95)),
            'common_root_xy_max_m':float(np.linalg.norm(delta,axis=1).max()),
            'common_root_tilt_max_deg':float(np.degrees(np.arccos(np.clip(
                1-2*(q[:,1]**2+q[:,2]**2),-1,1))).max()),
            'world_model_full_actual_episode':table[key].get('world_model_rollouts')}
    paired.append(summary)
a.output.write_text(json.dumps({'scope':'Common actual duration; full-episode WM windows differ and are descriptive, not a paired improvement claim','rows':paired},indent=2))
print(json.dumps(paired,indent=2))
