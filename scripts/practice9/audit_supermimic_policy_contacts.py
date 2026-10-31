"""Audit all frozen policy/PD native contacts, correcting the explicit terrain label."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True);p.add_argument('--original-root',type=Path,required=True)
a=p.parse_args();assert not (a.state/'contact_audit.json').exists()
smoke=json.loads((a.state/'classifier_smoke/results.json').read_text());assert smoke['rows'][0]['steps']==1
rows=[]
for case in ['standing_retention_zero_readout_v1','standing_policy_anchor_v1']:
 root=a.state/case;original=a.original_root/case/'evaluation'
 new=json.loads((root/'results.json').read_text());old=json.loads((original/'results.json').read_text())
 for k in ['checkpoint_sha256','manifest_sha256','seed','max_steps']:assert old[k]==new[k]
 original_rows={(r['id'],r['agent']):r for r in old['rows']}
 assert len(original_rows)==len(new['rows'])==2
 for r in new['rows']:
  oldr=original_rows[(r['id'],r['agent'])];assert r['steps']==oldr['steps'] and r['termination_reasons']==oldr['termination_reasons']
  rawpath=root/r['native_contact_file'];raw=json.loads(rawpath.read_text());assert raw['state_read_unchanged']
  with np.load(root/r['transition_file']) as z:data={k:z[k].copy() for k in z.files}
  with np.load(original/oldr['transition_file']) as z:olddata={k:z[k].copy() for k in z.files}
  for key,width in [('state',73),('next_state',73),('action',30),('context',34)]:assert data[key].shape==(r['steps'],width) and np.isfinite(data[key]).all()
  assert np.array_equal(data['next_state'][:-1],data['state'][1:])
  replay={k:float(np.max(np.abs(data[k]-olddata[k]))) for k in data}
  series=raw['samples'];assert len(series)==r['steps'] and [x['step'] for x in series]==list(range(1,r['steps']+1))
  trajectory=[]
  for i,frame in enumerate(series):
   feet={}
   for side in ['left','right']:
    f=frame['feet'][side];ground=[c for c in f['contacts'] if 'terrain' in c['geoms'] and c['normal_force_n']>1e-4]
    assert f['geom']=='robot/'+side+'_foot_collision'
    matrix=np.array(f['rotation_w']);positions=np.array([c['pos_w'] for c in ground]).reshape(-1,3)
    span=np.ptp((positions-np.array(f['position_w']))@matrix,axis=0) if len(positions)>0 else np.zeros(3)
    feet[side]=dict(loaded_points=len(ground),normal_force_n=sum(c['normal_force_n'] for c in ground),span_local_m=span.tolist(),tilt_deg=f['tilt_deg'])
   state=data['next_state'][i];q=state[3:7]/np.linalg.norm(state[3:7])
   trajectory.append(dict(step=i+1,seconds=(i+1)*.02,root_xy_m=float(np.linalg.norm(state[:2]-data['state'][0,:2])),root_tilt_deg=float(np.degrees(np.arccos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))) ,feet=feet))
  name=f'{case}_{r["agent"]}_contact_series.json';(a.state/name).write_text(json.dumps(trajectory))
  fixed=[1,10,20,40,60,r['steps']];selected=[trajectory[i-1] for i in sorted(set(fixed)) if i<=r['steps']]
  rows.append(dict(case=case,agent=r['agent'],steps=r['steps'],termination_reasons=r['termination_reasons'],raw_sha256=hashlib.sha256(rawpath.read_bytes()).hexdigest(),state_read_exactly_unchanged=True,replay_max_abs_difference=replay,replay_identical=all(v==0 for v in replay.values()),series=name,fixed_time_samples=selected))
report=dict(scope='Actual GPU contact forces, sampled after each control step. Terrain is the explicitly verified plane named terrain; original v1 world-body-only ground summaries are invalid and preserved. Contact/lean timing is descriptive, not causal proof. Replays retain original durations/terminations but are not claimed bitwise identical.',rows=rows,promoted=False)
(a.state/'contact_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps([dict(case=r['case'],agent=r['agent'],steps=r['steps'],replay_max_abs_difference=r['replay_max_abs_difference'],last=r['fixed_time_samples'][-1]) for r in rows]))
