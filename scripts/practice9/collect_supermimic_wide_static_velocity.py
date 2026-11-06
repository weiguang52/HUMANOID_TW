"""Fit-evidence-selected static native velocity branches; no policy changes."""
import ast,hashlib,json,sys
from pathlib import Path
source=Path(__file__).with_name('collect_supermimic_root_velocity_branches.py')
output=Path(sys.argv[sys.argv.index('--output')+1]);assert not output.exists()
proposal=output.parent/'proposal.json';p=json.loads(proposal.read_text())
assert p['selection_split']=='fit_only' and p['levels']==[1,2,8,16]
root=output.parent.parent/'additive_wide_root_pair_v1'
assert hashlib.sha256((root/'static_velocity_coverage_summary.json').read_bytes()).hexdigest()==p['selection_evidence_sha256']
tree=ast.parse(source.read_text());changes=[]
for n in ast.walk(tree):
 if isinstance(n,ast.Dict):
  keys=[k.value if isinstance(k,ast.Constant) else None for k in n.keys]
  if 'contract' in keys and isinstance(n.values[keys.index('contract')],ast.Constant) and n.values[keys.index('contract')].value=='root_velocity_state_branches_v1':
   n.values[keys.index('contract')]=ast.Constant('wide_static_root_velocity_branches_v1')
   n.values[keys.index('levels')]=ast.parse(repr([float(x) for x in p['levels']]),mode='eval').body
   changes.append(n)
assert len(changes)==1
record={'base_collector_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'proposal_sha256':hashlib.sha256(proposal.read_bytes()).hexdigest(),'modified_fields':['plan.contract','plan.levels'],'physics_and_reset_unchanged':True,'training_approved':False}
try:exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
finally:
 if output.exists():(output/'variant.json').write_text(json.dumps(record,indent=2))
