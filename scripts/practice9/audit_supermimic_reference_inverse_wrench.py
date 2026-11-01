"""Independent free-base inverse-dynamics wrench check on unchanged references."""
import argparse,hashlib,json
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial import ConvexHull,Delaunay
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser(description=__doc__)
for k in ['plan','phase-review','asset','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();plan=json.loads(a.plan.read_text());phase=json.loads(a.phase_review.read_text());m=mujoco.MjModel.from_xml_path(str(a.asset));d=mujoco.MjData(m)
# Independent local inverse calculation has no external contacts. Never sim.step.
# Joint armature contributes only joint rows for fixed prescribed qdd; no base rows.
m.geom_contype[:]=0;m.geom_conaffinity[:]=0;rows=[]
for item,old in zip(plan['motions'],phase['rows']):
 assert item['id']==old['id'];f=Path(item['file']);assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256']==old['reference_sha256']
 with np.load(f) as z:
  n=len(z['joint_pos']);dt=1/float(z['fps'].item());qa=[int(m.joint(k).qposadr[0]) for k in z['joint_names']];va=[int(m.joint(k).dofadr[0]) for k in z['joint_names']];q=np.zeros((n,m.nq));q[:,:3]=z['root_pos'];q[:,3:7]=z['root_quat_wxyz'];q[:,qa]=z['joint_pos'];rot=Rotation.from_quat(q[:,[4,5,6,3]]);worldw=np.zeros((n,3));worldw[1:-1]=(rot[2:]*rot[:-2].inv()).as_rotvec()/(2*dt);worldw[0]=worldw[1];worldw[-1]=worldw[-2];v=np.zeros((n,m.nv));v[:,:3]=np.gradient(q[:,:3],dt,axis=0);v[:,3:6]=rot.inv().apply(worldw);v[:,va]=np.gradient(q[:,qa],dt,axis=0);acc=np.gradient(v,dt,axis=0);samples=[]
  for g in old['source_contact_support']:
   if g['mask']!='both_feet_phase_known':continue
   for s in g['samples']:
    i=s['frame'];d.qpos[:]=q[i];d.qvel[:]=v[i];d.qacc[:]=acc[i];mujoco.mj_inverse(m,d);force=d.qfrc_inverse[:3].copy();torque=rot[i].apply(d.qfrc_inverse[3:6])+np.cross(q[i,:3],force);assert force[2]>0;cop=np.array([-torque[1],torque[0]])/force[2];points=np.array(s['selected_hull_xy_m']);eq=ConvexHull(points).equations;margin=float(np.min(-(eq[:,:2]@cop+eq[:,2])/np.linalg.norm(eq[:,:2],axis=1)));outside=bool(Delaunay(points).find_simplex(cop,tol=1e-10)<0);assert outside==(margin < -1e-10) or abs(margin)<=1e-10
    samples.append(dict(frame=i,sides=s['selected_sides'],inverse_force_z_n=float(force[2]),inverse_cop_xy_m=cop.tolist(),centroidal_cop_xy_m=s['required_cop_xy_m'],cop_difference_m=float(np.linalg.norm(cop-s['required_cop_xy_m'])),signed_margin_m=margin,outside=outside))
  single=[s for s in samples if len(s['sides'])==1];rows.append(dict(id=item['id'],reference_sha256=item['sha256'],known_frames=len(samples),single_support_frames=len(single),inverse_outside_single_support_frames=sum(s['outside'] for s in single),cop_difference_max_m=max(s['cop_difference_m'] for s in samples),cop_difference_median_m=float(np.median([s['cop_difference_m'] for s in samples])),samples=samples))
out=dict(scope='Independent MuJoCo inverse-dynamics unactuated free-base rows, prescribed reference q/v/a. No dynamics integration or production configuration change. Local collision masks disabled only to isolate required external wrench; no assertion about actuator torques/hardware. Compare with centroidal finite-difference method and independent Delaunay support containment.',asset_sha256=hashlib.sha256(a.asset.read_bytes()).hexdigest(),phase_review_sha256=hashlib.sha256(a.phase_review.read_bytes()).hexdigest(),rows=rows);a.output.write_text(json.dumps(out,indent=2)+'\n');print([(r['id'],r['single_support_frames'],r['inverse_outside_single_support_frames'],r['cop_difference_max_m']) for r in rows])
