"""Reuse audited TW195 corrected data without editing motion/contact arrays."""
from pathlib import Path
import json,shutil,hashlib
s=Path('/root/gpufree-data/datasets/practice9/tw202_v1');old=s.parent/'tw195_v1';s.mkdir(exist_ok=True)
variants={'baseline':'PathRate','precision':'Precision','neck':'PrecisionNeck','root':'PrecisionNeckRoot'}
shutil.copy2(old/'train_corrected.json',s/'train.json')
shutil.copy2(old/'data_audit.json',s/'data_audit.json')
for v in variants:
 out=s/'validation'/v;out.mkdir(parents=True,exist_ok=True)
 shutil.copy2(old/'validation/corrected/manifest.json',out/'manifest.json')
 for seed in [42,123,2026]:
  jobs=json.loads((old/f'validation/corrected/jobs{seed}.json').read_text())
  for j in jobs:j['output']=str(out/Path(j['output']).name)
  (out/f'jobs{seed}.json').write_text(json.dumps(jobs,indent=2))
 shutil.copytree(old/'eval/corrected',s/'eval'/v,dirs_exist_ok=True)
 diag=s/'diagnostic'/v;diag.mkdir(parents=True,exist_ok=True)
 jobs=[dict(motion_id=i,steps=1000,controller='policy',output=str(diag/(name+'.json'))) for i,name in [(2,'corrected_static'),(3,'corrected_upper')]]
 (diag/'jobs.json').write_text(json.dumps(jobs,indent=2))
checkpoint=Path((old/'corrected/final_checkpoint').read_text().strip())
(s/'initial_checkpoint').write_text(str(checkpoint)+'\n')
m=json.loads((s/'train.json').read_text());audit=dict(variants=variants,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),train_manifest_sha256=hashlib.sha256((s/'train.json').read_bytes()).hexdigest(),motions=len(m['motions']),frames=sum(r['frames'] for r in m['motions']),seed=42,environments=1024,additional_iterations=2000,physics_hz=200,control_hz=50,contact_unchanged=True)
(s/'experiment.json').write_text(json.dumps(audit,indent=2));print(audit)
