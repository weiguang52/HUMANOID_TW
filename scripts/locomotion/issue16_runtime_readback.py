"""Zero-step readback of effective PhysX and actuator parameters in the exact clean evaluator."""
import inspect,json
from pathlib import Path
import numpy as np
import torch
from pxr import Usd,UsdPhysics
from isaaclab.utils.io import dump_yaml
from issue16_common import ROOT,sha

def collect(base,original_source_sha,instrumented_source_sha):
 robot=base.scene['robot'];view=robot.root_physx_view;arrays={};unsupported={};docs={}
 getters={
 'raw_masses':'get_masses','raw_COM_poses':'get_coms','raw_inertias':'get_inertias','raw_materials':'get_material_properties',
 'raw_DOF_limits':'get_dof_limits','raw_DOF_stiffness':'get_dof_stiffnesses','raw_DOF_damping':'get_dof_dampings',
 'raw_DOF_armature':'get_dof_armatures','raw_DOF_effort_limit':'get_dof_max_forces','raw_DOF_velocity_limit':'get_dof_max_velocities',
 'raw_DOF_friction':'get_dof_friction_coefficients'}
 for key,name in getters.items():
  method=getattr(view,name,None)
  if method is None:unsupported[name]='getter unavailable';continue
  docs[name]=inspect.getdoc(method)
  try:
   value=method()
   arrays[key]=value.detach().cpu().numpy().copy() if isinstance(value,torch.Tensor) else np.array(value)
  except Exception as error:unsupported[name]=repr(error)
 for key in ['default_mass','default_inertia','default_joint_pos','default_joint_vel','joint_stiffness','joint_damping','joint_armature','joint_friction_coeff','joint_effort_limits','joint_velocity_limits','soft_joint_pos_limits']:
  try:
   value=getattr(robot.data,key);arrays['data_'+key]=value.detach().cpu().numpy().copy()
  except (AttributeError,RuntimeError) as error:unsupported['robot.data.'+key]=repr(error)
 term=base.action_manager.get_term('JointPositionAction')
 for key in ['_scale','_offset','_processed_actions']:
  value=getattr(term,key);arrays['action'+key]=value.detach().cpu().numpy().copy() if isinstance(value,torch.Tensor) else np.array(value)
 for key in ['raw_masses','raw_COM_poses','raw_inertias','raw_DOF_limits','raw_DOF_stiffness','raw_DOF_damping','raw_DOF_armature','raw_DOF_effort_limit','raw_DOF_velocity_limit']:
  assert key in arrays,('Required readback unavailable',key,unsupported)
 shared=view.shared_metatype
 link_names=list(shared.link_names);dof_names=list(shared.dof_names)
 assert len(link_names)==arrays['raw_masses'].shape[1] and len(dof_names)==arrays['raw_DOF_stiffness'].shape[1]
 assert len(set(link_names))==31 and len(set(dof_names))==30
 assert np.isfinite(arrays['raw_masses']).all() and (arrays['raw_masses']>0).all()
 assert np.array_equal(arrays['raw_masses'],arrays['data_default_mass'])
 identical={}
 for key,value in arrays.items():
  if value.ndim>0 and value.shape[0]==base.num_envs:identical[key]=bool(np.array_equal(value,np.broadcast_to(value[:1],value.shape)))
 for key in ['raw_masses','raw_COM_poses','raw_inertias','raw_DOF_limits','raw_DOF_stiffness','raw_DOF_damping']:assert identical[key],key
 import omni.usd
 stage=omni.usd.get_context().get_stage();joint_drives={}
 for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
  if not str(prim.GetPath()).startswith('/World/envs/env_0/'):continue
  if prim.IsA(UsdPhysics.RevoluteJoint):
   joint_drives[prim.GetName()]={a.GetName():str(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith('drive:')}
 layers=[]
 for layer in stage.GetUsedLayers():
  p=Path(layer.realPath)
  if p.is_file():layers.append(dict(path=str(p),sha256=sha(p)))
 actual=Path('/root/gpufree-data/datasets/practice9/custom_robot/usd/urdf0711_training_30dof_tw123_yaw_waist_v3.usd')
 assert any(Path(x['path']).resolve()==actual.resolve() and x['sha256']==sha(actual) for x in layers),'Actual model layer not found'
 folder=ROOT/'runtime_readback';folder.mkdir(exist_ok=True)
 np.savez_compressed(folder/'physics.npz',**arrays)
 dump_yaml(str(folder/'env.yaml'),base.cfg)
 metadata=dict(raw_readback_completed=True,formal_env_steps=0,formal_initial_trials=0,training_updates=0,policy_loaded_or_called=False,num_envs=base.num_envs,
  raw_PhysX_link_names=link_names,raw_PhysX_DOF_names=dof_names,robot_body_names=robot.body_names,robot_joint_names=robot.joint_names,action_joint_names=list(term._joint_names),
  getter_docstrings=docs,unsupported_optional_fields=unsupported,parameter_identical_across_environments=identical,
  array_shapes={k:list(v.shape) for k,v in arrays.items()},array_dtypes={k:str(v.dtype) for k,v in arrays.items()},
  authored_stage_joint_drive_attributes=joint_drives,drive_note='Authored USD attributes are distinguished from effective GPU PhysX readback; ImplicitActuator configuration overrides are not inferred from USD defaults.',
  physics_array_sha256=sha(folder/'physics.npz'),env_yaml_sha256=sha(folder/'env.yaml'),source_sha256=sha(__file__),
  original_evaluator_source_sha256=original_source_sha,instrumented_source_sha256=instrumented_source_sha,resolved_actual_stage_layers=layers,
  exact_PhysX_cooked_collision_hulls_extracted=False,collision_geometry_source='Complete authored meshes/analytic primitives and cooking attributes are in the separate frozen USD inspection.')
 (folder/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
 print('ISSUE16_ZERO_STEP_RUNTIME_READBACK_COMPLETE',len(arrays),flush=True)
