"""Build split/category-balanced manifests from audited FK outputs only."""
import argparse,json
from pathlib import Path
from tw75_sampling import balance
from joint_coordinates import CONTRACT_VERSION
p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--fk-manifests',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
lookup={r['id']:r for r in json.load(open(a.selection))};accepted={};template=None
for f in a.fk_manifests:
 d=json.load(open(f))
 if d.get('joint_coordinate_contract')!=CONTRACT_VERSION or float(d['target_fps'])!=50:raise ValueError('Wrong coordinate/FPS contract')
 if template is not None and d['joint_names']!=template['joint_names']:raise ValueError('Joint order mismatch')
 template=d
 for m in d['motions']:
  if m.get('quality_pass') is not True or m.get('fk_quality_pass') is not True:continue
  if m['id'] not in lookup:raise ValueError('Unknown source ID')
  if m['id'] in accepted:raise ValueError('Duplicate source ID')
  source=lookup[m['id']];accepted[m['id']]=dict(m,category=source['category'],split=source['split'],source_family=source['source_family'])
a.output.mkdir(parents=True,exist_ok=True)
summary={}
for split in ['train','val','test']:
 rows=[r for r in accepted.values() if r['split']==split];weighted=balance(rows)
 payload={k:template[k] for k in ['joint_coordinate_contract','schema_version','target_fps','robot','joint_names']};payload.update(backend={'backend':'s2_curriculum_v1','components':[str(f) for f in a.fk_manifests]},motions=weighted)
 (a.output/(split+'.json')).write_text(json.dumps(payload,indent=2)+'\n')
 summary[split]={'counts':{c:sum(r['category']==c for r in rows) for c in ['walking','standing_upper']},'frames':sum(r['frames'] for r in rows),'note':'50/50 category PRIOR and equal clip PRIOR. To preserve exact category mass use uniform_ratio=1, or implement within-category failure sampling. Global failure mixing can change class mass. Counts and coverage need review before training.'}
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
