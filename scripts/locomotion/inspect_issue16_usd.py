"""Inspect the exact issue16 USD and derive default-pose axial reflection signs."""
from pathlib import Path
import hashlib,json,math
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True)
app=launcher.app
from pxr import Usd,UsdGeom,UsdPhysics,Gf,Sdf
import numpy as np
from unitree_rl_lab.assets.robots.custom_humanoid import CUSTOM_HUMANOID_30DOF_CFG,CUSTOM_HUMANOID_ACTION_SCALE,CUSTOM_HUMANOID_30DOF_JOINT_NAMES
R=Path('/root/gpufree-data/tw_stage1/issue16_20261111')
P=Path('/root/gpufree-data/datasets/practice9/custom_robot/usd/urdf0711_training_30dof_tw123_yaw_waist_v3.usd')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def convert(v):
 if v is None or isinstance(v,(str,bool,int,float)):return v
 if isinstance(v,dict):return {str(k):convert(x) for k,x in v.items()}
 if isinstance(v,Sdf.AssetPath):return dict(path=v.path,resolved=v.resolvedPath)
 if hasattr(v,'GetReal'):return [float(v.GetReal()),*map(float,v.GetImaginary())]
 try:return [convert(x) for x in v]
 except TypeError:return str(v)
def attrs(p):return {a.GetName():convert(a.Get()) for a in p.GetAttributes() if a.GetName().startswith(('physics:','physx'))}
try:
 stage=Usd.Stage.Open(str(P));assert stage
 cache=UsdGeom.XformCache();bodies={};joints={};collisions=[];geometry={};names=CUSTOM_HUMANOID_30DOF_JOINT_NAMES
 for p in stage.Traverse(Usd.TraverseInstanceProxies()):
  if p.HasAPI(UsdPhysics.RigidBodyAPI):
   bodies[p.GetName()]={'path':str(p.GetPath()),'attributes':attrs(p),'world_matrix_row':convert(cache.GetLocalToWorldTransform(p))}
  if p.IsA(UsdPhysics.RevoluteJoint):
   j=UsdPhysics.RevoluteJoint(p);b0=j.GetBody0Rel().GetTargets();b1=j.GetBody1Rel().GetTargets();axis=j.GetAxisAttr().Get()
   q=j.GetLocalRot0Attr().Get();v=Gf.Vec3d(*{'X':(1,0,0),'Y':(0,1,0),'Z':(0,0,1)}[axis]);v=Gf.Rotation(Gf.Quatd(q)).TransformDir(v)
   if b0:v=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(b0[0])).TransformDir(v)
   v=np.array(v);v=v/np.linalg.norm(v)
   joints[p.GetName()]={'path':str(p.GetPath()),'body0':list(map(str,b0)),'body1':list(map(str,b1)),'attributes':attrs(p),'axis_world_at_authored_pose':v.tolist()}
  if p.HasAPI(UsdPhysics.CollisionAPI):
   parent=p
   while parent and not parent.IsPseudoRoot() and not parent.HasAPI(UsdPhysics.RigidBodyAPI):parent=parent.GetParent()
   x=cache.GetLocalToWorldTransform(p);key='collider_'+str(len(collisions));record={'path':str(p.GetPath()),'body':parent.GetName() if parent else None,'type':p.GetTypeName(),'attributes':attrs(p),'world_matrix_row':convert(x)}
   if p.IsA(UsdGeom.Mesh):
    m=UsdGeom.Mesh(p);points=m.GetPointsAttr().Get();points=np.array(points,dtype='float64')
    geometry[key+'_points_local']=points;geometry[key+'_points_world']=np.array([list(x.Transform(Gf.Vec3d(*v))) for v in points])
    geometry[key+'_face_counts']=np.array(m.GetFaceVertexCountsAttr().Get());geometry[key+'_face_indices']=np.array(m.GetFaceVertexIndicesAttr().Get());record.update(array_key=key,point_count=len(points))
   else:record['geometry_attributes']={a.GetName():convert(a.Get()) for a in p.GetAttributes() if a.GetName() in ['size','radius','height','axis','extent']}
   record['descendant_geometry']=[]
   for child in Usd.PrimRange(p,Usd.TraverseInstanceProxies()):
    if child==p or not child.IsA(UsdGeom.Boundable):continue
    child_record={'path':str(child.GetPath()),'type':child.GetTypeName(),'attributes':attrs(child),'world_matrix_row':convert(cache.GetLocalToWorldTransform(child))}
    if child.IsA(UsdGeom.Mesh):
     mesh=UsdGeom.Mesh(child);points=np.array(mesh.GetPointsAttr().Get(),dtype='float64');ckey=key+'_desc'+str(len(record['descendant_geometry']))
     mat=cache.GetLocalToWorldTransform(child)
     geometry[ckey+'_points_local']=points;geometry[ckey+'_points_world']=np.array([list(mat.Transform(Gf.Vec3d(*v))) for v in points]);geometry[ckey+'_face_counts']=np.array(mesh.GetFaceVertexCountsAttr().Get());geometry[ckey+'_face_indices']=np.array(mesh.GetFaceVertexIndicesAttr().Get())
     child_record.update(array_key=ckey,point_count=len(points))
    else:child_record['geometry_attributes']={a.GetName():convert(a.Get()) for a in child.GetAttributes() if a.GetName() in ['size','radius','height','axis','extent']}
    record['descendant_geometry'].append(child_record)
   collisions.append(record)
 cfg=CUSTOM_HUMANOID_30DOF_CFG;defaults=cfg.init_state.joint_pos
 def paired(n):return n.replace('left_','right_',1) if n.startswith('left_') else n.replace('right_','left_',1) if n.startswith('right_') else n
 table=[]
 for n in names:
  partner=paired(n);assert n in joints and partner in joints,(n,partner)
  axis=np.array(joints[n]['axis_world_at_authored_pose']);source=np.array(joints[partner]['axis_world_at_authored_pose']);reflected=source*np.array([-1,1,-1]);dot=float(axis@reflected);sign=1 if dot>=0 else -1
  table.append({'joint':n,'partner':partner,'sign':sign,'axis_alignment_cos_abs':abs(dot),'axis_mismatch_degrees':float(np.degrees(np.arccos(np.clip(abs(dot),-1,1)))),'default':defaults[n],'partner_default':defaults[partner],'absolute_coordinate_offset':defaults[n]-sign*defaults[partner],'action_scale':CUSTOM_HUMANOID_ACTION_SCALE[n]})
 for t in table:assert next(x for x in table if x['joint']==t['partner'])['sign']==t['sign']
 layers=[]
 for layer in stage.GetUsedLayers():
  p=Path(layer.realPath)
  layers.append({'identifier':layer.identifier,'path':str(p),'sha256':sha(p) if p.is_file() else None})
 out={'usd':str(P),'usd_sha256':sha(P),'meters_per_unit':UsdGeom.GetStageMetersPerUnit(stage),'up_axis':str(UsdGeom.GetStageUpAxis(stage)),'layers':layers,'bodies':bodies,'joints':joints,'collision_shapes':collisions,'joint_mapping_authored_pose':table,'defaults':defaults,'action_scales':CUSTOM_HUMANOID_ACTION_SCALE,'actuators':{k:convert(v.to_dict()) for k,v in cfg.actuators.items()},'source_sha256':sha(__file__)}
 np.savez_compressed(R/'usd_collision_geometry.npz',**geometry)
 (R/'usd_inspection.json').write_text(json.dumps(out,indent=2)+'\n')
 print('ISSUE16_USD_INSPECTION',len(bodies),len(joints),len(collisions),flush=True)
 for t in table:print('MAP',t['joint'],t['partner'],t['sign'],t['axis_mismatch_degrees'],t['absolute_coordinate_offset'],flush=True)
finally:app.close()
