from pathlib import Path
import json,hashlib
R=Path('/root/gpufree-data/datasets/practice9/perturbation_20261025');R.mkdir(exist_ok=True)
audit=json.loads(Path('/root/gpufree-data/datasets/practice9/repeatability_20261025/audit.json').read_text())
for model,c in audit['checkpoints'].items():
 assert hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()==c['sha256']
 p=R/model;p.mkdir(exist_ok=True);(p/'checkpoint').write_text(c['path']+'\n')
 plan=[]
 for name,seed,speeds,front in [(f'batch_{seed}',seed,[0.,.05,.1],False) for seed in [42,123,2026]]+[('repeat_42',42,[.05],False),('front_42',42,[0.,.05,.1],True)]:
  out=p/name;out.mkdir()
  jobs=[{'motion_id':mid,'steps':1000,'controller':'policy','push_speed':speed,'push_step':250,'output':str(out/f'motion_{mid}_v{round(speed*100):02d}.json')} for speed in speeds for mid in [2,3]]
  (out/'jobs.json').write_text(json.dumps(jobs,indent=2));plan.append({'name':name,'seed':seed,'front':front,'jobs':str(out/'jobs.json')})
 (p/'plan.json').write_text(json.dumps(plan,indent=2))
audit['protocol']='At5s add root world XY velocity0/.05/.1m/s; seeded direction; read physical velocity back. Other diagnostic randomness remains disabled.'
(R/'audit.json').write_text(json.dumps(audit,indent=2))
