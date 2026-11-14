"""Apply original runtime coordinate interpretation and require reflected PhysX tensors."""
from pathlib import Path
from issue20_common import ROOT,save
import json
p=Path(__file__).with_name('audit_issue16_runtime.py');s=p.read_text().replace('from issue16_common import ROOT,sha','from issue20_common import ROOT,sha');exec(compile(s,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
v=json.loads((ROOT/'runtime_symmetry_audit.json').read_text());assert v['body_authored_consistency_passed'] and v['joint_effective_parameter_checks_passed']
for r in v['effective_reflection_comparisons']:
 assert r['mass_relative_difference']<1e-7 and r['COM_reflection_distance_m']<1e-6 and r['inertia_frobenius_relative_difference']<1e-4,r
save('runtime_strict_model_gate.json',{'passed':True,'authored_and_runtime_parameters_checked':True,'body_count':31,'joint_count':30});print('ISSUE20_RUNTIME_SYMMETRY_PASS',flush=True)
