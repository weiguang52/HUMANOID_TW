"""Read actual GPU contact forces; never recompute CPU physics."""
import mujoco,mujoco_warp
import numpy as np

def native_foot_contacts(env):
 mujoco_warp.get_data_into(env.sim.mj_data,env.sim.mj_model,env.sim.wp_data)
 model,data=env.sim.mj_model,env.sim.mj_data
 names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(model.ngeom)]
 assert 'terrain' in names and int(model.geom_type[names.index('terrain')])==int(mujoco.mjtGeom.mjGEOM_PLANE)
 feet={side:[i for i,n in enumerate(names) if side+'_foot_collision' in n] for side in ['left','right']}
 assert all(len(ids)==1 for ids in feet.values()), 'Require known sole-box asset'
 result={}
 for side,ids in feet.items():
  gid=ids[0];matrix=data.geom_xmat[gid].reshape(3,3);points=[]
  assert np.isfinite(matrix).all()
  for ci in range(data.ncon):
   co=data.contact[ci];geoms=[int(x) for x in co.geom]
   if gid not in geoms:continue
   force=np.zeros(6);mujoco.mj_contactForce(model,data,ci,force)
   other=geoms[1] if geoms[0]==gid else geoms[0]
   points.append(dict(geoms=[names[i] for i in geoms],ground=bool(names[other]=='terrain'),pos_w=co.pos.tolist(),normal_force_n=float(force[0]),contact_frame_force=force.tolist(),distance_m=float(co.dist)))
  ground=[p for p in points if p['ground'] and p['normal_force_n']>1e-4]
  positions=np.array([p['pos_w'] for p in ground]).reshape(-1,3)
  spans=np.ptp((positions-data.geom_xpos[gid])@matrix,axis=0) if len(positions)>0 else np.zeros(3)
  result[side]=dict(geom=names[gid],sole_size_m=(2*model.geom_size[gid]).tolist(),position_w=data.geom_xpos[gid].tolist(),rotation_w=matrix.tolist(),tilt_deg=float(np.degrees(np.arccos(np.clip(matrix[2,2],-1,1)))),loaded_ground_points=len(ground),ground_normal_force_n=sum(p['normal_force_n'] for p in ground),loaded_span_local_m=spans.tolist(),contacts=points)
 return result
