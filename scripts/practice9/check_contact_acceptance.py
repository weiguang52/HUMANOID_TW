"""Predeclared numerical gates. Passing is not a substitute for visual/independent validation."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'validation_artifacts/contact_ablation_20261025'
results={}
for variant in ['baseline','contact']:
 p=D/variant/'evaluation_colored/neck_only'
 c=json.loads((p/'comparison.json').read_text())
 assert len(c['rows'])==42
 diagnostics={}
 for case in ['corrected_static','corrected_upper']:
  meta=json.loads((p/'diagnostic'/f'{case}.json').read_text())
  with np.load(p/'diagnostic'/f'{case}.npz') as a:
   drift=float(np.linalg.norm(a['anchor_actual'][-1,:2]-a['anchor_actual'][50,:2])*1000)
  diagnostics[case]={'seconds':meta['seconds'],'terminations':meta['termination_counts'],'drift_mm':drift}
 results[variant]={'categories':c['categories'],'diagnostics':diagnostics}
b=results['baseline'];c=results['contact'];gates={}
for case,limit in [('corrected_static',109.1526566),('corrected_upper',35.0555629)]:
 d=c['diagnostics'][case]
 gates[case+'_complete']=d['seconds']>=20 and not any(d['terminations'].values())
 gates[case+'_drift_original']=d['drift_mm']<limit if case=='corrected_static' else d['drift_mm']<=limit
 reference=b['diagnostics'][case]
 comparable=gates[case+'_complete'] and reference['seconds']>=20 and not any(reference['terminations'].values())
 gates[case+'_drift_baseline']=(d['drift_mm']<reference['drift_mm'] if case=='corrected_static' else d['drift_mm']<=reference['drift_mm']) if comparable else None
for group,target in [('standing_upper',6),('flat_forward',23),('turning',3)]:
 gates[group]=c['categories'][group]['clean']>=target
original=json.loads((ROOT/'validation_artifacts/head_locked_20261024/evaluation_colored/neck_only/comparison.json').read_text())
for group in c['categories']:
 gates[group+'_no_original_count_regression']=c['categories'][group]['clean']>=original['categories'][group]['clean']
 gates[group+'_no_baseline_count_regression']=c['categories'][group]['clean']>=b['categories'][group]['clean']
out={'gates':gates,'numerical_gates_pass':all(gates.values()),'results':results,'remaining':['Inspect per-group wrist/upper/heading errors and full video evidence.','Independent validation required before claiming robust improvement.','No automatic promotion.']}
(D/'acceptance.json').write_text(json.dumps(out,indent=2));print(json.dumps(gates,indent=2))
