from pathlib import Path
import json
scripts=Path('scripts/practice9')
state=Path('/root/gpufree-data/datasets/practice9/tw176_v1');state.mkdir(exist_ok=True)
old=state.parent/'tw154_v1'
(state/'validation').mkdir(exist_ok=True)
# Immutable inputs reused in place; no regeneration, relabeling, or local copies.
for dest,src in [(state/'train.json',old/'train.json'),(state/'contact',old/'contact'),(state/'validation/manifest.json',old/'validation/manifest.json')]:
 if not dest.exists():dest.symlink_to(src)
manifest=json.loads((old/'validation/manifest.json').read_text())
for v in ['control','sole']:
 out=state/'validation'/v;out.mkdir(exist_ok=True)
 for seed in [42,123,2026]:
  jobs=[dict(motion_id=i,steps=r['frames']+1,output=str(out/f'{r["id"]}.seed{seed}.json')) for i,r in enumerate(manifest['motions'])]
  (out/f'jobs{seed}.json').write_text(json.dumps(jobs,indent=2))
