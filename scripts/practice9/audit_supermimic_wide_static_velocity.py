"""Original archive checks plus complete retained-level regression."""
import argparse,ast,hashlib,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--original',type=Path,required=True);a=p.parse_args()
assert not (a.root/'integrity.json').exists()
plan=json.loads((a.root/'plan.json').read_text());assert plan['levels']==[1.,2.,8.,16.]
v=json.loads((a.root/'variant.json').read_text());assert v['modified_fields']==['plan.contract','plan.levels'] and v['physics_and_reset_unchanged']
source=Path(__file__).with_name('audit_supermimic_root_velocity_branches.py');s=source.read_text()
assert s.count('"root_velocity_state_branches_v1"')==1
s=s.replace('"root_velocity_state_branches_v1"','"wide_static_root_velocity_branches_v1"').replace('"integrity.json"','"base_checks.json"')
sys.argv=[str(source),'--root',str(a.root)]
exec(compile(ast.parse(s),str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
base=json.loads((a.root/'base_checks.json').read_text());assert base['passed']
old=json.loads((a.original/'results.json').read_text());count=0;maxdiff=0.;proof=[]
for row in old['rows']:
 if row['warmup'] not in plan[row['split']+'_warmups']:continue
 name=row['file'];newpath=a.root/name;oldpath=a.original/name
 with np.load(newpath) as n,np.load(oldpath) as o:
  assert set(n.files)==set(o.files)
  for k in n.files:
   if np.issubdtype(n[k].dtype,np.floating):
    np.testing.assert_allclose(n[k],o[k],rtol=0,atol=1e-5)
    maxdiff=max(maxdiff,float(np.max(np.abs(n[k]-o[k]))))
   else:np.testing.assert_array_equal(n[k],o[k])
 proof.append({'file':name,'original_sha256':hashlib.sha256(oldpath.read_bytes()).hexdigest()});count+=1
assert count==5*(len(plan['fit_warmups'])+len(plan['validation_warmups']))
base.update(retained_original_files=count,retained_original_max_difference=maxdiff,original_files=proof,variant_sha256=hashlib.sha256((a.root/'variant.json').read_bytes()).hexdigest(),auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),policy_acceptance=False)
(a.root/'integrity.json').write_text(json.dumps(base,indent=2));print('COMPLETE',count,'original files; max difference',maxdiff)
