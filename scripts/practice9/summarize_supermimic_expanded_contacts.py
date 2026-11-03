"""Describe all fixed expanded native traces, including original failures."""
import argparse,json,collections,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
assert json.loads((r/'full_independent_audit.json').read_text())['passed']
plan=json.loads((r/'collection_plan.json').read_text());rows=json.loads((r/'native_full/results.json').read_text())['rows'];assert len(rows)==96
reasons=collections.Counter();contact_masks=collections.Counter();loaded_counts=collections.Counter();source=[];durations=[];windows=collections.Counter();traces=[]
for motion in plan['motions']:
 selected=[x for x in rows if x['id']==motion['id']];assert len(selected)==4
 source.append(dict(id=motion['id'],source_family=motion['source_family'],corpus=motion['corpus'],episodes=4,steps=sum(x['steps'] for x in selected),terminated=sum(bool(x['termination_reasons']) for x in selected)))
for row in rows:
 path=r/'native_full'/row['trace'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'];d=np.load(path);n=len(d['state']);durations.append(n*.02)
 for h in [1,8,32]:windows[h]+=max(n-h+1,0)
 reasons.update(row['termination_reasons']);contact=json.loads((r/'native_full'/row['native_contact_file']).read_text());assert len(contact['samples'])==n
 for sample in contact['samples']:
  mask=''.join('1' if sample['feet'][side]['loaded_ground_points'] else '0' for side in ['left','right']);contact_masks[mask]+=1
  for foot in sample['feet'].values():loaded_counts[foot['loaded_ground_points']]+=1
 traces.append(dict(trace=row['trace'],sha256=row['sha256'],steps=n,terminated=bool(row['termination_reasons'])))
out=dict(episodes=96,steps=sum(x['steps'] for x in rows),terminated_episodes=sum(bool(x['termination_reasons']) for x in rows),termination_reasons=dict(reasons),duration_seconds=dict(min=min(durations),median=float(np.median(durations)),max=max(durations)),available_all_windows=dict(windows),post_step_contact_masks=dict(contact_masks),loaded_points_per_foot_histogram=dict(loaded_counts),sources=source,traces=traces,policy_acceptance=False,training_export_approved=False,scope='Original reference-target PD collection, maximum2s per phase; neither learned policy nor20s success. Contacts are post-step observations; loaded point counts do not certify noncollinear planar support. All original terminations retained, no quality filtering.')
(r/'collection_review.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['sources','traces']}))
