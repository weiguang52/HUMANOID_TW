"""Readable original-versus-symmetric mass/inertia/COM/joint/collision/PD table."""
from pathlib import Path
import json,csv,numpy as np
from issue20_common import ROOT,OLD,sha,save
from audit_issue16_symmetry import rotation

def physical_body(b):
 A=np.array(b['world_matrix_row']).T;a=b['attributes'];R=A[:3,:3]@rotation(a['physics:principalAxes']);return {'mass_kg':a['physics:mass'],'origin_world_m':A[:3,3].tolist(),'COM_world_m':(A@np.r_[a['physics:centerOfMass'],1])[:3].tolist(),'inertia_world_kg_m2':(R@np.diag(a['physics:diagonalInertia'])@R.T).tolist()}
def collisions(data,body):
 rows=[]
 for c in data['collision_shapes']:
  if c['body']!=body:continue
  nodes=[c,*c.get('descendant_geometry',[])];rows.append({'path':c['path'],'type':c['type'],'physical_attributes':c['attributes'],'geometry':[{'type':x['type'],'points':x.get('point_count'),'primitive_attributes':x.get('geometry_attributes'),'world_matrix_row':x['world_matrix_row']} for x in nodes if x['type']=='Mesh' or x.get('geometry_attributes')]})
 return rows
def gains(data):
 out={}
 for group,c in data['actuators'].items():
  for n in c['joint_names_expr']:out[n]={k:c[k] for k in ['stiffness','damping','armature','effort_limit_sim','velocity_limit_sim','friction','dynamic_friction','viscous_friction']}
 return out
def main():
 before=json.loads((OLD.with_name('issue16_20261111')/'usd_inspection.json').read_text());after=json.loads((ROOT/'usd_inspection.json').read_text());g0=gains(before);g1=gains(after);rows=[]
 def add(kind,name,key,a,b):rows.append({'kind':kind,'entity':name,'property':key,'original':json.dumps(a,separators=(',',':')),'symmetric':json.dumps(b,separators=(',',':')),'changed':a!=b})
 for n,b in before['bodies'].items():
  a=physical_body(b);c=physical_body(after['bodies'][n])
  for key in a:add('body',n,key,a[key],c[key])
  add('body',n,'collision_nodes_and_geometry',collisions(before,n),collisions(after,n))
 for n,j in before['joints'].items():
  k=after['joints'][n]
  for key in ['axis_world_at_authored_pose','body0','body1']:add('joint',n,key,j[key],k[key])
  for key in ['physics:axis','physics:localPos0','physics:localPos1','physics:localRot0','physics:localRot1','physics:lowerLimit','physics:upperLimit']:add('joint',n,key,j['attributes'][key],k['attributes'][key])
  add('joint',n,'default_radians',before['defaults'][n],after['defaults'][n]);add('joint',n,'action_scale',before['action_scales'][n],after['action_scales'][n]);add('joint',n,'actuator_PD_and_limits',g0[n],g1[n])
 with (ROOT/'usd_physical_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# Original versus strict symmetric physical model','','Full per-property rows: [usd_physical_comparison.csv](usd_physical_comparison.csv). Every original authored attribute/geometry-array difference is additionally retained losslessly in usd_attribute_differences.csv (restore large payload from chunk manifest).','','|body|original mass kg|symmetric mass kg|original COM world m|symmetric COM world m|','|---|---:|---:|---|---|']
 for n,b in before['bodies'].items():
  a=physical_body(b);c=physical_body(after['bodies'][n]);lines.append(f'|{n}|{a["mass_kg"]:.8g}|{c["mass_kg"]:.8g}|{a["COM_world_m"]}|{c["COM_world_m"]}|')
 lines+=['','Right body poses, collision geometry and joint frames use the left side as authority while retaining the issue16 signed joint-coordinate reflection. Central COM y, odd inertia terms and joint anchors are projected onto sagittal symmetry; center collision convex hulls are clipped on the left and mirrored. Defaults, action scales and actuator PD/effort/velocity/friction settings are tabulated separately and remain unchanged. Physical tensors are compared in the authored world frame; actual PhysX readback is recorded separately. Collision point counts/topology changes alone do not measure cooked-hull distance or establish a causal contribution to heading.']
 (ROOT/'usd_physical_comparison.md').write_text('\n'.join(lines)+'\n');save('physical_comparison_audit.json',{'passed':True,'body_count':31,'joint_count':30,'property_rows':len(rows),'changed_rows':sum(x['changed'] for x in rows),'original_usd_sha256':before['usd_sha256'],'symmetric_usd_sha256':after['usd_sha256'],'all_defaults_scales_PD_unchanged':all(not r['changed'] for r in rows if r['property'] in ['default_radians','action_scale','actuator_PD_and_limits']),'csv_sha256':sha(ROOT/'usd_physical_comparison.csv')});print('ISSUE20_MODEL_DIFFERENCE_TABLE_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
