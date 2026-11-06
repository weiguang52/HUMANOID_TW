"""Frozen finite static root responses on independently verified wider branches."""
import ast,hashlib,json,sys
from pathlib import Path
collection=Path(sys.argv[sys.argv.index('--collection')+1]);output=Path(sys.argv[sys.argv.index('--output')+1])
proof=json.loads((collection/'integrity.json').read_text());plan=json.loads((collection/'plan.json').read_text())
assert proof['passed'] and len(proof['files'])==72 and proof['retained_original_files']==40
assert plan['contract']=='wide_static_root_velocity_branches_v1' and plan['levels']==[1.,2.,8.,16.]
source=Path(__file__).with_name('audit_supermimic_root_state_response.py');tree=ast.parse(source.read_text());count=0
for n in ast.walk(tree):
 if isinstance(n,ast.Constant) and n.value=='root_velocity_state_branches_v1':n.value='wide_static_root_velocity_branches_v1';count+=1
assert count==1
exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
d=json.loads((output/'audit.json').read_text());assert all(x['weights_and_norms_unchanged'] and len(x['conditions'])==32 for x in d['models'])
record={'base_auditor_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'integrity_sha256':hashlib.sha256((collection/'integrity.json').read_bytes()).hexdigest(),'modified_fields':['accepted_contract_name'],'scope':'Finite branch responses only, not local Jacobian or closed-loop stability; no training or policy acceptance'}
(output/'provenance.json').write_text(json.dumps(record,indent=2))
