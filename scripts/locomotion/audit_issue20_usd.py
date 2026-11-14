"""Quantify strict-model structural reflection using the unchanged issue16 metrics."""
from pathlib import Path
import json
from issue20_common import ROOT,save
p=Path(__file__).with_name('audit_issue16_symmetry.py');s=p.read_text().replace('from issue16_common import ROOT,sha','from issue20_common import ROOT,sha');exec(compile(s,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
v=json.loads((ROOT/'usd_symmetry_audit.json').read_text());errors=[]
for row in v['body_comparisons']:
 for key,tol in [('mass_relative_difference',1e-7),('COM_reflection_distance_m',1e-6),('inertia_frobenius_relative_difference',1e-5),('authored_body_origin_reflection_distance_m',1e-6),('collision_vertex_cloud_hausdorff_m',1e-6)]:
  if row.get(key,float('inf'))>tol:errors.append([row['body'],key,row.get(key),tol])
for row in v['joint_comparisons']:
 for key,tol in [('axis_mismatch_degrees',1e-3),('pivot_reflection_distance_m',1e-6),('limit_max_difference_degrees',1e-5),('default_reflection_absolute_difference_radians',1e-7)]:
  if row[key]>tol:errors.append([row['joint'],key,row[key],tol])
 if not row['PD_config_equal']:errors.append([row['joint'],'PD mismatch'])
save('strict_usd_preflight.json',{'passed':not errors,'errors':errors,'scope':'mass, inertia, COM, joint axis/pivot/limit/default, collision vertex clouds, actual source actuator PD; runtime readback separately','body_count':v['body_count'],'joint_count':v['joint_count']});print('STRICT_USD_PREFLIGHT',not errors,errors,flush=True);assert not errors,errors
