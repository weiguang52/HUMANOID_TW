import json
from pathlib import Path
from diagnose_sole_contacts import diagnose
s=Path('/root/gpufree-data/datasets/practice9/tw198_v1');m=json.loads((s/'manifest.json').read_text());rows=[]
for p in sorted(s.glob('*/*/*.npz')):
 mid=p.stem.removesuffix('_reference_pd').removesuffix('_policy');ix=next(i for i,r in enumerate(m['motions']) if r['id']==mid)
 d=diagnose(p,m['motions'][ix],sum(r['frames'] for r in m['motions'][:ix]));d.update(model=p.parts[-3],rate=p.parts[-2],case=p.stem);rows.append(d)
p=Path('validation_artifacts/tw198_standing_diagnostic/sole_metrics.json');p.write_text(json.dumps(rows,indent=2));print('sole metrics',len(rows))
