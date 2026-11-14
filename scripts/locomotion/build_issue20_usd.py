"""Construct left-authoritative sagittal physical model without changing defaults."""
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True)
app=launcher.app
from pxr import Usd,UsdGeom,UsdPhysics,Gf,Sdf,Vt
import numpy as np,json,csv
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from issue20_common import ROOT,USD,sha,save
from issue16_mirror import counterpart
S=np.diag([1.,-1.,1.]);C=np.diag([1.,-1.,1.,1.])
def quatR(q):return Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix()
def Rquat(r):
 x,y,z,w=Rotation.from_matrix(r).as_quat();return Gf.Quatf(float(w),Gf.Vec3f(float(x),float(y),float(z)))
def mat(p):return np.array(UsdGeom.Xformable(p).GetLocalTransformation(),float).T
def setmat(p,m):UsdGeom.Xformable(p).MakeMatrixXform().Set(Gf.Matrix4d(m.T.tolist()))
def reflect_geometry(p):
 for v in Usd.PrimRange(p):
  if v!=p and v.IsA(UsdGeom.Xformable):setmat(v,C@mat(v)@C)
  if v.IsA(UsdGeom.Mesh):
   mesh=UsdGeom.Mesh(v);pts=np.asarray(mesh.GetPointsAttr().Get(),float)@S;mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
   ns=mesh.GetNormalsAttr().Get()
   if ns is not None and len(ns):mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy((np.asarray(ns)@S).astype(np.float32)))
   counts=list(mesh.GetFaceVertexCountsAttr().Get());indices=list(mesh.GetFaceVertexIndicesAttr().Get());out=[];offset=0
   for n in counts:out+=list(reversed(indices[offset:offset+n]));offset+=n
   mesh.GetFaceVertexIndicesAttr().Set(out)
   if len(pts):mesh.GetExtentAttr().Set([Gf.Vec3f(*map(float,pts.min(0))),Gf.Vec3f(*map(float,pts.max(0)))])
def inertia(p):
 a=UsdPhysics.MassAPI(p);R=quatR(a.GetPrincipalAxesAttr().Get());return R@np.diag(np.asarray(a.GetDiagonalInertiaAttr().Get()))@R.T
try:
 ROOT.mkdir(parents=True,exist_ok=True);out=ROOT/'strict_symmetric.usd';original=Usd.Stage.Open(str(USD));flat=original.Flatten();stage=Usd.Stage.Open(flat);root=stage.GetDefaultPrim();cache=UsdGeom.XformCache()
 bodies={p.GetName():p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)};original_world={n:np.array(cache.GetLocalToWorldTransform(p),float).T for n,p in bodies.items()};body_paths={n:str(p.GetPath()) for n,p in bodies.items()}
 while True:
  instances=[p for p in stage.Traverse() if p.IsInstanceable()]
  if not instances:break
  for p in instances:p.SetInstanceable(False)
 original_values={str(a.GetPath()):str(a.Get()) for p in stage.Traverse() for a in p.GetAttributes()};source_layer=Sdf.Layer.CreateAnonymous();source_layer.TransferContent(flat)
 contract=json.loads((ROOT/'issue_snapshot.json').read_text());axes=json.loads((__import__('pathlib').Path('tests/locomotion/fixtures/issue16_joint_axes.json')).read_text());signs={x['joint']:x['sign'] for x in axes['joints']}
 for n in list(bodies):
  if n.startswith('right_'):
   left=counterpart(n);path=body_paths[n];stage.RemovePrim(path);assert Sdf.CopySpec(source_layer,body_paths[left],stage.GetRootLayer(),path);p=stage.GetPrimAtPath(path);setmat(p,C@original_world[left]@C);reflect_geometry(p)
   a=UsdPhysics.MassAPI(p);a.GetCenterOfMassAttr().Set(Gf.Vec3f(*(S@np.asarray(a.GetCenterOfMassAttr().Get()))));a.GetPrincipalAxesAttr().Set(Rquat(S@quatR(a.GetPrincipalAxesAttr().Get())@S))
  elif not n.startswith('left_'):
   p=stage.GetPrimAtPath(body_paths[n]);m=original_world[n].copy();m[1,3]=0.;u,_,vt=np.linalg.svd((m[:3,:3]+S@m[:3,:3]@S)/2);m[:3,:3]=u@vt;assert np.linalg.det(m[:3,:3])>0;setmat(p,m)
   a=UsdPhysics.MassAPI(p);com=np.asarray(a.GetCenterOfMassAttr().Get()).copy();com[1]=0;a.GetCenterOfMassAttr().Set(Gf.Vec3f(*map(float,com)));I=inertia(p);I=(I+S@I@S)/2;w,Q=np.linalg.eigh(I)
   if np.linalg.det(Q)<0:Q[:,0]*=-1
   assert np.all(w>0);a.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float,w)));a.GetPrincipalAxesAttr().Set(Rquat(Q))
 # Center colliders: clip authored convex hull to left half-plane, then reflect it.
 cache=UsdGeom.XformCache()
 for n,path in body_paths.items():
  if n.startswith(('left_','right_')):continue
  p=stage.GetPrimAtPath(path);inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(p),float).T);points=[];settings=None;collider_roots=[]
  for v in Usd.PrimRange(p):
   if v.HasAPI(UsdPhysics.CollisionAPI):
    collider_roots.append(str(v.GetPath()));settings=v
    for g in Usd.PrimRange(v):
     if g.IsA(UsdGeom.Mesh):
      x=np.asarray(UsdGeom.Mesh(g).GetPointsAttr().Get(),float);M=inv@np.array(cache.GetLocalToWorldTransform(g),float).T;points.append((np.c_[x,np.ones(len(x))]@M.T)[:,:3])
  if not points:raise AssertionError(('No center collision mesh',n))
  cloud=np.unique(np.concatenate(points),axis=0);h=ConvexHull(cloud);left=cloud[cloud[:,1]>=0].tolist()
  for face in h.simplices:
   for i,j in [(face[0],face[1]),(face[1],face[2]),(face[2],face[0])]:
    a,b=cloud[i],cloud[j]
    if a[1]*b[1]<0:
     q=a+(b-a)*(-a[1]/(b[1]-a[1]));q[1]=0;left.append(q.tolist())
  left=np.asarray(left);symmetric=np.unique(np.r_[left,left@S],axis=0);h=ConvexHull(symmetric);faces=h.simplices.copy()
  for i,tri in enumerate(faces):
   a,b,c=symmetric[tri]
   if np.dot(np.cross(b-a,c-a),h.equations[i,:3])<0:faces[i]=tri[::-1]
  physical={a.GetName():(a.GetTypeName(),a.Get()) for a in settings.GetAttributes() if a.GetName().startswith(('physics:','physx'))}
  for pathc in collider_roots:stage.OverridePrim(pathc).SetActive(False)
  mesh=UsdGeom.Mesh.Define(stage,path+'/strict_symmetric_collision');mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(symmetric.astype(np.float32)));mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist());mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateVisibilityAttr('invisible');UsdPhysics.CollisionAPI.Apply(mesh.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
  for key,(typ,value) in physical.items():
   if value is not None:mesh.GetPrim().CreateAttribute(key,typ).Set(value)
 source_stage=Usd.Stage.Open(source_layer)
 # Joint coordinate frames must retain the original signed M_a convention.
 for p in list(stage.Traverse()):
  if not p.IsA(UsdPhysics.RevoluteJoint):continue
  n=p.GetName();sign=signs[n];joint=UsdPhysics.RevoluteJoint(p)
  if n.startswith('right_'):
   lp=source_stage.GetPrimAtPath(str(p.GetPath()).replace('/right_','/left_'));source=UsdPhysics.RevoluteJoint(lp);axis=str(source.GetAxisAttr().Get());axis_i='XYZ'.index(axis);H=S.copy()
   if H[axis_i,axis_i]!=-sign:
    H[axis_i,axis_i]*=-1;other=(axis_i+1)%3;H[other,other]*=-1
   for side in [0,1]:
    getattr(joint,'GetLocalPos'+str(side)+'Attr')().Set(Gf.Vec3f(*(S@np.asarray(getattr(source,'GetLocalPos'+str(side)+'Attr')().Get()))))
    getattr(joint,'GetLocalRot'+str(side)+'Attr')().Set(Rquat(S@quatR(getattr(source,'GetLocalRot'+str(side)+'Attr')().Get())@H))
    targets=getattr(source,'GetBody'+str(side)+'Rel')().GetTargets();getattr(joint,'GetBody'+str(side)+'Rel')().SetTargets([Sdf.Path('/'.join(counterpart(v) for v in str(t).split('/'))) for t in targets])
   lo=float(source.GetLowerLimitAttr().Get());hi=float(source.GetUpperLimitAttr().Get());lo,hi=sorted([sign*lo,sign*hi]);joint.GetLowerLimitAttr().Set(lo);joint.GetUpperLimitAttr().Set(hi)
   for a in source.GetPrim().GetAttributes():
    key=a.GetName()
    if key.startswith(('drive:','physxJoint:')) and a.Get() is not None:
     value=a.Get()*sign if key.endswith(('targetPosition','targetVelocity')) else a.Get();p.CreateAttribute(key,a.GetTypeName()).Set(value)
  elif not n.startswith('left_'):
   for side in [0,1]:
    attr=getattr(joint,'GetLocalPos'+str(side)+'Attr')();v=np.asarray(attr.Get()).copy();v[1]=0;attr.Set(Gf.Vec3f(*map(float,v)))
    attr=getattr(joint,'GetLocalRot'+str(side)+'Attr')();R=quatR(attr.Get());H=S.copy();axis_i='XYZ'.index(str(joint.GetAxisAttr().Get()))
    if H[axis_i,axis_i]!=-sign:H[axis_i,axis_i]*=-1;H[(axis_i+1)%3,(axis_i+1)%3]*=-1
    u,_,vt=np.linalg.svd((R+S@R@H)/2);R=u@vt;assert np.linalg.det(R)>0;attr.Set(Rquat(R))
   if sign==-1:
    lim=max(abs(float(joint.GetLowerLimitAttr().Get())),abs(float(joint.GetUpperLimitAttr().Get())));joint.GetLowerLimitAttr().Set(-lim);joint.GetUpperLimitAttr().Set(lim)
 stage.GetRootLayer().Export(str(out));new_values={str(a.GetPath()):str(a.Get()) for p in stage.Traverse() for a in p.GetAttributes()};diff=[{'attribute':k,'original':original_values.get(k),'symmetric':new_values.get(k)} for k in sorted(set(original_values)|set(new_values)) if original_values.get(k)!=new_values.get(k)]
 with (ROOT/'usd_attribute_differences.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['attribute','original','symmetric']);w.writeheader();w.writerows(diff)
 save('symmetric_usd_construction.json',{'passed':True,'original_usd':str(USD),'original_sha256':sha(USD),'symmetric_usd':str(out),'symmetric_sha256':sha(out),'left_authoritative':True,'center_colliders':'left-half clipped authored convex hull mirrored, explicit masses unchanged; symmetric center COM/inertia/joint anchors','joint_coordinate_signs':signs,'changed_attributes':len(diff),'training_updates':0,'defaults_modified':False,'default_asset_modified':False})
 print('STRICT_USD_BUILT',str(out),out.stat().st_size,len(diff),flush=True)
except BaseException:
 import traceback;traceback.print_exc()
 raise
finally:app.close()
