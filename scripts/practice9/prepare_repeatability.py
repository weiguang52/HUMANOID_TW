"""Freeze diagnostic repeatability matrix; no training or checkpoint selection."""
from pathlib import Path
import json,hashlib
R=Path('/root/gpufree-data/datasets/practice9/repeatability_20261025')
R.mkdir(exist_ok=True)
models={'original':Path('/root/gpufree-data/datasets/practice9/head_locked_20261024/final_checkpoint').read_text().strip(), 'baseline':Path('/root/gpufree-data/datasets/practice9/resume_ablation_20261024/baseline_recovery/final_checkpoint').read_text().strip()}
plan=[]
for seed in [42,123,2026]:
 plan.append((f'batch_{seed}',seed,[2,3],False))
 for mid in [2,3]:plan.append((f'single_{seed}_{mid}',seed,[mid],False))
plan.extend([('repeat_42',42,[2,3],False),('reverse_42',42,[3,2],False),('front_42',42,[2,3],True)])
for model,checkpoint in models.items():
 p=R/model;p.mkdir(exist_ok=True)
 (p/'checkpoint').write_text(checkpoint+'\n')
 rows=[]
 for name,seed,ids,front in plan:
  out=p/name
  if out.exists():raise FileExistsError(out)
  out.mkdir()
  jobs=[{'motion_id':mid,'steps':1000,'controller':'policy','output':str(out/f'motion_{mid}.json')} for mid in ids]
  (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
  rows.append({'name':name,'seed':seed,'front':front,'jobs':str(out/'jobs.json')})
 (p/'plan.json').write_text(json.dumps(rows,indent=2))
audit={'checkpoints':{k:{'path':v,'sha256':hashlib.sha256(Path(v).read_bytes()).hexdigest()} for k,v in models.items()},'scope':'Two static diagnostic motions only; no general locomotion acceptance','manifest':'/root/gpufree-data/datasets/practice9/tw198_v1/manifest.json','robot_usd':'/root/gpufree-data/datasets/practice9/head_locked_20261024/robot_colored.usd'}
for key in ['manifest','robot_usd']:audit[key+'_sha256']=hashlib.sha256(Path(audit[key]).read_bytes()).hexdigest()
(R/'audit.json').write_text(json.dumps(audit,indent=2))
