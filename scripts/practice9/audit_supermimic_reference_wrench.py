"""Reference centroidal wrench from finite-difference FK; optimistic geometry only."""
import argparse,hashlib,itertools,json
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial import ConvexHull
from supermimic_reference_wrench import infer_wrench,ground_cop
p=argparse.ArgumentParser(description=__doc__)
for key in ['plan','asset','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();plan=json.loads(a.plan.read_text());m=mujoco.MjModel.from_xml_path(str(a.asset));d=mujoco.MjData(m);mass=m.body_mass.copy();total=mass.sum();sign=np.array(list(itertools.product([-1,1],repeat=3)));rows=[]
for item in plan['motions']:
 f=Path(item['file']);assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256']
 with np.load(f) as z:
  dt=1/float(z['fps'].item());n=len(z['joint_pos']);adr=[int(m.joint(k).qposadr[0]) for k in z['joint_names']];pos=[];rot=[];polys=[]
  for i in range(n):
   d.qpos[:3]=z['root_pos'][i];d.qpos[3:7]=z['root_quat_wxyz'][i];d.qpos[adr]=z['joint_pos'][i];mujoco.mj_kinematics(m,d);pos.append(d.xipos.copy());rot.append(d.ximat.reshape(-1,3,3).copy());points=[]
   for side in ['left','right']:
    g=m.geom(side+'_foot_collision').id;points.extend(((sign*m.geom_size[g])@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,:2])
   polys.append(ConvexHull(np.array(points)).equations)
  pos=np.array(pos);rot=np.array(rot);com,momentum,force,torque=infer_wrench(pos,rot,mass,m.body_inertia,dt,m.opt.gravity);cop=ground_cop(force,torque);valid=np.arange(2,n-2);margins=[]
  for i in valid:
   eq=polys[i];margins.append(float(np.min(-(eq[:,:2]@cop[i]+eq[:,2])/np.linalg.norm(eq[:,:2],axis=1))))
  assert np.isfinite(force).all() and np.isfinite(torque).all() and np.isfinite(cop).all()
  rows.append(dict(id=item['id'],split=item['split'],reference_sha256=item['sha256'],interior_frames=len(valid),nonpositive_required_vertical_force_frames=int(np.sum(force[valid,2]<=0)),required_force_z_min_n=float(force[valid,2].min()),required_force_z_max_n=float(force[valid,2].max()),required_cop_outside_optimistic_envelope_frames=int(np.sum(np.array(margins)<0)),cop_margin_min_m=min(margins),cop_margin_median_m=float(np.median(margins)),required_friction_ratio_p95=float(np.quantile(np.linalg.norm(force[valid,:2],axis=1)/np.maximum(force[valid,2],1e-12),.95))))
report=dict(scope='Finite-difference full-body FK centroidal momentum and required flat-ground wrench. Excludes two endpoints. Foot-box XY envelope includes airborne feet and is optimistic, not loaded support. No contact-force reconstruction, actuator/friction-cone feasibility proof or new acceptance gate. Finite differences may amplify reference noise.',asset_sha256=hashlib.sha256(a.asset.read_bytes()).hexdigest(),total_mass_kg=float(total),rows=rows);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(rows))
