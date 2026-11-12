"""Quantify reflected USD mass/inertia/COM/joint/PD/authored-collision differences."""
import csv,json,math,re,itertools
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree,ConvexHull,QhullError
from issue16_common import ROOT,sha
from issue16_mirror import counterpart
S=np.diag([1.,-1.,1.])
def rotation(q):
 q=np.array(q,dtype=float);q=q/np.linalg.norm(q);w,x,y,z=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def main():
 source=ROOT/'usd_inspection.json';data=json.loads(source.read_text());bodies=data['bodies'];joint=data['joints'];rows=data['joint_mapping_authored_pose']
 geometry=np.load(ROOT/'usd_collision_geometry.npz');points={};geometry_meta={}
 for c in data['collision_shapes']:
  name=c['body'];points.setdefault(name,[]);geometry_meta.setdefault(name,[])
  for g in [c,*c.get('descendant_geometry',[])]:
   if g.get('array_key'):
    k=g['array_key'];points[name].append(geometry[k+'_points_world'])
    geometry_meta[name].append(dict(type='Mesh',path=g['path'],points=len(geometry[k+'_points_world']),faces=len(geometry[k+'_face_counts']),indices=len(geometry[k+'_face_indices']),face_indices_sha256=__import__('hashlib').sha256(geometry[k+'_face_indices'].tobytes()).hexdigest()))
   elif g['type']=='Cube':
    size=float(g['geometry_attributes']['size'])
    local=np.array(list(itertools.product([-size/2,size/2],repeat=3)))
    matrix=np.array(g['world_matrix_row'])
    world=(np.c_[local,np.ones(8)]@matrix)[:,:3]
    points[name].append(world)
    geometry_meta[name].append(dict(type='Cube',path=g['path'],points=8,faces=6,indices=24,
     vertex_definition='Schema size-based eight corners transformed by the complete authored Xform. Authored extent attribute retained separately; not used as a substitute for Cube size.',
     size=size,authored_extent=g['geometry_attributes'].get('extent'),world_vertices=world.tolist()))
 clouds={};hulls={}
 for name,values in points.items():
  if not values:continue
  cloud=np.unique(np.concatenate(values),axis=0);clouds[name]=cloud
  try:
   hull=ConvexHull(cloud);hulls[name]=dict(volume_m3=float(hull.volume),vertices=len(hull.vertices),facets=len(hull.simplices))
  except QhullError as e:hulls[name]=dict(volume_m3=None,error=str(e)[:300])
 body_rows=[];seen=set()
 for name,body in bodies.items():
  partner=counterpart(name)
  assert partner in bodies,(name,partner)
  key=tuple(sorted([name,partner]))
  if key in seen:continue
  seen.add(key);other=bodies[partner];a=body['attributes'];b=other['attributes']
  A=np.array(body['world_matrix_row']).T;B=np.array(other['world_matrix_row']).T
  comA=(A@np.r_[a['physics:centerOfMass'],1])[:3];comB=(B@np.r_[b['physics:centerOfMass'],1])[:3]
  Ra=A[:3,:3]@rotation(a['physics:principalAxes']);Rb=B[:3,:3]@rotation(b['physics:principalAxes'])
  Ia=Ra@np.diag(a['physics:diagonalInertia'])@Ra.T;Ib=Rb@np.diag(b['physics:diagonalInertia'])@Rb.T
  massA=a['physics:mass'];massB=b['physics:mass']
  out=dict(body=name,partner=partner,central_self=name==partner,mass_kg=massA,partner_mass_kg=massB,
   mass_absolute_difference_kg=abs(massA-massB),mass_relative_difference=abs(massA-massB)/max(massA,massB),
   COM_world_m=comA.tolist(),reflected_partner_COM_world_m=(S@comB).tolist(),COM_reflection_distance_m=float(np.linalg.norm(comA-S@comB)),
   inertia_world_kg_m2=Ia.tolist(),reflected_partner_inertia_world_kg_m2=(S@Ib@S).tolist(),
   inertia_frobenius_relative_difference=float(np.linalg.norm(Ia-S@Ib@S)/np.linalg.norm(Ia)),
   inertia_frobenius_absolute_difference_kg_m2=float(np.linalg.norm(Ia-S@Ib@S)),
   authored_body_origin_reflection_distance_m=float(np.linalg.norm(A[:3,3]-S@B[:3,3])),
   collision_meshes=geometry_meta.get(name,[]),partner_collision_meshes=geometry_meta.get(partner,[]))
  if name in clouds and partner in clouds:
   ca=clouds[name];cb=clouds[partner]@S
   d1=cKDTree(cb).query(ca,workers=1)[0];d2=cKDTree(ca).query(cb,workers=1)[0]
   out.update(collision_unique_vertices=len(ca),partner_collision_unique_vertices=len(cb),
    collision_vertex_cloud_hausdorff_m=float(max(d1.max(),d2.max())),collision_vertex_cloud_mean_bidirectional_m=float((d1.mean()+d2.mean())/2),
    collision_bbox_min_m=ca.min(0).tolist(),collision_bbox_max_m=ca.max(0).tolist(),
    reflected_partner_collision_bbox_min_m=cb.min(0).tolist(),reflected_partner_collision_bbox_max_m=cb.max(0).tolist(),
    authored_convex_hull=hulls[name],partner_authored_convex_hull=hulls[partner])
  else:out['collision_coverage_note']='At least one side has no authored descendant mesh; inspect raw collider node/type/settings.'
  body_rows.append(out)
 gains={}
 for group,cfg in data['actuators'].items():
  for name in cfg['joint_names_expr']:
   assert name not in gains,name
   gains[name]={k:cfg[k] for k in ['stiffness','damping','armature','effort_limit_sim','velocity_limit_sim','friction','dynamic_friction','viscous_friction']}
 joint_rows=[]
 for row in rows:
  name=row['joint'];partner=row['partner'];sign=row['sign'];a=joint[name];b=joint[partner]
  def pivot(v):
   frame=np.eye(4)
   if v['body0']:
    bodyname=v['body0'][0].split('/')[-1];frame=np.array(bodies[bodyname]['world_matrix_row']).T
   return (frame@np.r_[v['attributes']['physics:localPos0'],1])[:3]
  low=a['attributes']['physics:lowerLimit'];high=a['attributes']['physics:upperLimit']
  pl=b['attributes']['physics:lowerLimit'];ph=b['attributes']['physics:upperLimit'];mirrored=sorted([sign*pl,sign*ph])
  joint_rows.append(dict(joint=name,partner=partner,sign=sign,axis_mismatch_degrees=row['axis_mismatch_degrees'],
   pivot_reflection_distance_m=float(np.linalg.norm(pivot(a)-S@pivot(b))),
   lower_limit_degrees=low,upper_limit_degrees=high,reflected_partner_lower_limit_degrees=mirrored[0],reflected_partner_upper_limit_degrees=mirrored[1],
   limit_max_difference_degrees=max(abs(low-mirrored[0]),abs(high-mirrored[1])),
   default_radians=row['default'],partner_default_radians=row['partner_default'],default_reflection_absolute_difference_radians=abs(row['absolute_coordinate_offset']),
   action_scale=row['action_scale'],PD=gains[name],partner_PD=gains[partner],PD_config_equal=gains[name]==gains[partner]))
 result=dict(audit_completed=True,scope='Exact authored experiment USD, source configuration and resolved mesh geometry. Runtime detailed PhysX readback remains a separate required validation.',
  body_count=len(bodies),joint_count=len(joint),paired_or_central_body_comparisons=len(body_rows),collision_nodes=len(data['collision_shapes']),
  authored_descendant_meshes=sum(g['type']=='Mesh' for v in geometry_meta.values() for g in v),raw_authored_mesh_points=sum(g['points'] for v in geometry_meta.values() for g in v if g['type']=='Mesh'),
  analytic_primitive_count=sum(g['type']=='Cube' for v in geometry_meta.values() for g in v),
  body_comparisons=body_rows,joint_comparisons=joint_rows,
  collision_nodes_without_mesh=[c['path'] for c in data['collision_shapes'] if not any('array_key' in g for g in [c,*c.get('descendant_geometry',[])])],
  collision_interpretation='Vertex-cloud Hausdorff and authored convex-hull volume are reproducible source-geometry diagnostics, sensitive to tessellation and not a claim to extract exact PhysX-cooked hulls. All authored mesh points, topology, transforms and collision/cooking attributes are preserved. No causal contribution to heading is inferred from a small structural difference alone.',
  source_sha256=sha(__file__),inputs={str(source):sha(source),str(ROOT/'usd_collision_geometry.npz'):sha(ROOT/'usd_collision_geometry.npz')},
  usd_sha256=data['usd_sha256'],model_or_configuration_modified=False)
 (ROOT/'usd_symmetry_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 flat=[{k:v for k,v in r.items() if isinstance(v,(str,bool,int,float))} for r in body_rows]
 keys=sorted(set().union(*(r.keys() for r in flat)))
 with (ROOT/'usd_body_symmetry.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)
 print('ISSUE16_STATIC_USD_AUDIT_COMPLETE',len(body_rows),len(joint_rows),'meshpoints',result['raw_authored_mesh_points'],flush=True)
if __name__=='__main__':main()
