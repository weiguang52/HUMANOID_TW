"""Interpret zero-step PhysX readback using installed SDK frame conventions."""
import ast,csv,json
from pathlib import Path
import numpy as np
from issue16_common import ROOT,sha
from issue16_mirror import counterpart
from audit_issue16_symmetry import rotation,S

def main():
 folder=ROOT/'runtime_readback';metadata=folder/'metadata.json'
 m=json.loads(metadata.read_text());x=np.load(folder/'physics.npz')
 u=json.loads((ROOT/'usd_inspection.json').read_text())
 assert m['raw_readback_completed'] and m['formal_env_steps']==0 and not m['policy_loaded_or_called']
 assert sha(folder/'physics.npz')==m['physics_array_sha256']
 sdk=next(Path('/root/gpufree-data/isaacsim/extscache').glob('omni.physics.tensors*/omni/physics/tensors/impl/api.py'))
 code=sdk.read_text();tree=ast.parse(code);methods={}
 for cls in tree.body:
  if isinstance(cls,ast.ClassDef) and cls.name=='ArticulationView':
   for f in cls.body:
    if isinstance(f,ast.FunctionDef) and f.name in m['getter_docstrings']:
     methods[f.name]={'docstring':ast.get_docstring(f),'source':ast.get_source_segment(code,f)}
 assert 'qx, qy, qz' in methods['get_coms']['docstring']
 assert 'rigid-body-prim frame' in methods['get_inertias']['docstring'] and 'column-major' in methods['get_inertias']['docstring']
 evidence={'installed_sdk_path':str(sdk),'installed_sdk_sha256':sha(sdk),'methods':methods}
 (folder/'installed_sdk_getter_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
 links=m['raw_PhysX_link_names'];dofs=m['raw_PhysX_DOF_names'];joints=m['robot_joint_names']
 assert set(links)==set(u['bodies']) and set(dofs)==set(joints)==set(u['joints'])
 body_rows=[];world={}
 for i,name in enumerate(links):
  b=u['bodies'][name];a=b['attributes'];A=np.array(b['world_matrix_row']).T
  p=x['raw_COM_poses'][0,i];R=rotation(np.r_[p[6],p[3:6]])
  I=x['raw_inertias'][0,i].reshape(3,3).T
  Ra=rotation(a['physics:principalAxes']);expected=Ra@np.diag(a['physics:diagonalInertia'])@Ra.T
  world[name]={'COM':(A@np.r_[p[:3],1])[:3],'I':A[:3,:3]@I@A[:3,:3].T,'mass':float(x['raw_masses'][0,i])}
  # Principal axes are non-unique for near-equal eigenvalues; compare reconstructed tensor.
  principal=R.T@I@R
  body_rows.append(dict(body=name,mass_kg=float(x['raw_masses'][0,i]),
   authored_mass_kg=a['physics:mass'],mass_absolute_difference_kg=abs(float(x['raw_masses'][0,i])-a['physics:mass']),
   COM_local_m=p[:3].tolist(),authored_COM_local_m=a['physics:centerOfMass'],
   COM_authored_distance_m=float(np.linalg.norm(p[:3]-a['physics:centerOfMass'])),
   inertia_local_kg_m2=I.tolist(),authored_inertia_local_kg_m2=expected.tolist(),
   inertia_authored_frobenius_absolute_difference_kg_m2=float(np.linalg.norm(I-expected)),
   inertia_authored_frobenius_relative_difference=float(np.linalg.norm(I-expected)/np.linalg.norm(expected)),
   principal_frame_off_diagonal_norm=float(np.linalg.norm(principal-np.diag(np.diag(principal))))))
 seen=set();pairs=[]
 for name,a in world.items():
  partner=counterpart(name);key=tuple(sorted([name,partner]))
  if key in seen:continue
  seen.add(key);b=world[partner]
  pairs.append(dict(body=name,partner=partner,central_self=name==partner,
   mass_relative_difference=abs(a['mass']-b['mass'])/max(a['mass'],b['mass']),
   COM_reflection_distance_m=float(np.linalg.norm(a['COM']-S@b['COM'])),
   inertia_frobenius_absolute_difference_kg_m2=float(np.linalg.norm(a['I']-S@b['I']@S)),
   inertia_frobenius_relative_difference=float(np.linalg.norm(a['I']-S@b['I']@S)/np.linalg.norm(a['I']))))
 gains={}
 for cfg in u['actuators'].values():
  for name in cfg['joint_names_expr']:gains[name]=cfg
 mapping={r['joint']:r for r in u['joint_mapping_authored_pose']}
 joint_rows=[];all_checks=[]
 action_names=m['action_joint_names'];assert set(action_names)==set(joints)
 def action_value(key,index):
  value=x[key]
  return float(value) if value.ndim==0 else float(value[0,index] if value.ndim==2 else value[index])
 for i,name in enumerate(dofs):
  partner=counterpart(name);pi=dofs.index(partner);ri=joints.index(name);r=mapping[name];a=u['joints'][name]['attributes'];g=gains[name]
  expected={'raw_DOF_stiffness':g['stiffness'],'raw_DOF_damping':g['damping'],'raw_DOF_armature':g['armature'],
   'raw_DOF_effort_limit':g['effort_limit_sim'],'raw_DOF_velocity_limit':g['velocity_limit_sim']}
  row=dict(joint=name,partner=partner,mirror_sign=r['sign'],effective={},configured={},checks={})
  for key,v in expected.items():
   value=float(x[key][0,i]);row['effective'][key]=value;row['configured'][key]=v
   row['checks'][key+'_matches_config']=bool(np.isclose(value,v,rtol=1e-6,atol=1e-7))
   row['checks'][key+'_LR_equal']=bool(x[key][0,i]==x[key][0,pi])
  limits=x['raw_DOF_limits'][0,i];want=np.deg2rad([a['physics:lowerLimit'],a['physics:upperLimit']])
  row['effective_limits_radians']=limits.tolist()
  row['checks']['limits_match_authored']=bool(np.allclose(limits,want,rtol=1e-6,atol=1e-6))
  row['checks']['limits_reflect_partner']=bool(np.allclose(limits,sorted(r['sign']*x['raw_DOF_limits'][0,pi]),atol=1e-6))
  default=float(x['data_default_joint_pos'][0,ri]);row['default_radians']=default
  row['checks']['default_matches_config']=bool(np.isclose(default,r['default'],atol=1e-7))
  row['checks']['default_reflects_partner']=bool(np.isclose(default,r['sign']*x['data_default_joint_pos'][0,joints.index(partner)],atol=1e-7))
  ai=action_names.index(name)
  row['action_scale']=action_value('action_scale',ai);row['action_offset']=action_value('action_offset',ai)
  row['checks']['action_scale_matches_config']=bool(np.isclose(row['action_scale'],r['action_scale'],atol=1e-7))
  row['checks']['action_offset_matches_default']=bool(np.isclose(row['action_offset'],r['default'],atol=1e-7))
  row['authored_stage_drive_attributes']=m['authored_stage_joint_drive_attributes'].get(name)
  row['checks']['authored_drive_attributes_present']=row['authored_stage_drive_attributes'] is not None
  joint_rows.append(row);all_checks.extend(row['checks'].values())
 body_matches=all(r['mass_absolute_difference_kg']<1e-6 and r['COM_authored_distance_m']<1e-6 and r['inertia_authored_frobenius_relative_difference']<1e-4 for r in body_rows)
 result=dict(audit_completed=True,zero_step_validation_only=True,body_count=len(body_rows),joint_count=len(joint_rows),
  body_authored_consistency_passed=body_matches,joint_effective_parameter_checks_passed=all(all_checks),
  comparison_tolerances=dict(mass_kg=1e-6,COM_m=1e-6,inertia_relative=1e-4,joint_relative=1e-6,joint_absolute=1e-6),
  frame_conventions='COM: local xyz plus principal-axis quaternion xyzw. Inertia: column-major 3x3 tensor at COM expressed in rigid-body-prim frame; not rotated again by COM quaternion. World reflection uses authored body transforms at authored pose, not the dynamic reset pose.',
  body_authored_comparisons=body_rows,effective_reflection_comparisons=pairs,joint_comparisons=joint_rows,
  all_environment_parameter_identity=m['parameter_identical_across_environments'],
  scope='64 zero-step copies of exact clean evaluation scene. Formal jobs separately bind 1024-env mass/material interventions. Authored USD drives and effective ImplicitActuator overrides are separately recorded; no training or dynamics trial added.',
  exact_PhysX_cooked_collision_hulls_extracted=False,
  source_sha256=sha(__file__),inputs={str(p):sha(p) for p in [metadata,folder/'physics.npz',ROOT/'usd_inspection.json',folder/'installed_sdk_getter_evidence.json']})
 (ROOT/'runtime_symmetry_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 with (ROOT/'runtime_joint_parameters.csv').open('w',newline='') as f:
  rows=[{'joint':r['joint'],'partner':r['partner'],'sign':r['mirror_sign'],**r['effective'],**r['checks']} for r in joint_rows]
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('ISSUE16_RUNTIME_AUDIT',body_matches,all(all_checks),flush=True)
 if not body_matches or not all(all_checks):
  raise RuntimeError('Effective readback discrepancy preserved for investigation; do not silently claim consistency.')
if __name__=='__main__':main()
