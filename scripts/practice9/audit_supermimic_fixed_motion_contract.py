"""Freeze existing seven-motion evidence without changing samples or evaluation windows."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();r=a.data_root
manifest=r/'independent_holdout_20261025/test.json';train=r/'training_speed_blend_20261026/train.json'
baseline=r/'supermimic_tw_20261031/physics_v3_1000/evaluation/results.json'
m=json.loads(manifest.read_text());t=json.loads(train.read_text());b=json.loads(baseline.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(manifest)==b['manifest_sha256']
ids=['003245','012584','003363','008312','005048','004518','010195']
assert [x['id'] for x in m['motions']]==ids
assert not set(ids)&{x['id'] for x in t['motions']}
train_families={x['source_family'] for x in t['motions']}
assert not {x['source_family'] for x in m['motions']}&train_families
previous={x['id']:x for x in b['rows'] if x['agent']=='trained'}
rows=[]
for x in m['motions']:
 path=Path(x['file'])
 with np.load(path,allow_pickle=False) as z:
  frames=len(z['joint_pos']);fps=float(z['fps'].item())
 assert frames==x['frames'] and fps==50
 steps=previous[x['id']]['planned_steps'];assert steps==min(frames-1,b['max_steps'])
 text=r.parent/'HumanML3D-official/HumanML3D/texts'/ (x['id']+'.txt')
 assert text.exists()
 rows.append(dict(id=x['id'],source_family=x['source_family'],file=str(path),sha256=sha(path),frames=frames,fps=fps,
  existing_required_steps=steps,existing_required_duration_s=steps/fps,full_reference_transition_duration_s=(frames-1)/fps,
  existing_window_covers_full_clip=steps==frames-1,category=x['category'],
  caption_file=str(text),caption_sha256=sha(text),caption=text.read_text().splitlines()[0],
  quality_pass=x['quality_pass'],strict_retarget_quality_pass=x['strict_retarget_quality_pass'],existing_acceptance_policy=x['acceptance_policy'],
  baseline_steps=previous[x['id']]['steps'],baseline_termination_reasons=previous[x['id']]['termination_reasons']))
out=dict(manifest=str(manifest),manifest_sha256=sha(manifest),training_manifest_sha256=sha(train),baseline_results_sha256=sha(baseline),
 id_and_source_family_overlap=[],motions=rows,passed_source_isolation=True,
 scope='Existing fixed samples/windows unchanged. Category walking includes other motions. Passing a capped20s window does not prove completion of longer full retargeted reference clips. Advisory near-limit acceptance inherited, not newly relaxed. Not a policy/multi-seed acceptance result.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='motions'}));print(json.dumps([{k:x[k] for k in ['id','existing_required_duration_s','full_reference_transition_duration_s','strict_retarget_quality_pass']} for x in rows]))
