"""Independent full-sequence reconstruction; diagnostic only, no training export."""
import argparse, datetime, hashlib, json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser(description=__doc__)
for k in ['plan','asset','results','output']: p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--partial',action='store_true',help='Audit completed prefix only; never full acceptance')
a=p.parse_args(); assert not a.output.exists()
h=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
plan=json.loads(a.plan.read_text()); sources=Path(plan['source_plan']); phases=Path(plan['phase_review'])
assert h(sources)==plan['source_plan_sha256'] and h(phases)==plan['phase_review_sha256']
src=json.loads(sources.read_text()); phase=json.loads(phases.read_text()); results=json.loads(a.results.read_text())
selected=[(item,pr) for item,pr in zip(src['motions'],phase['rows']) if not plan.get('source_split') or item['split']==plan['source_split']]
expected=[item['id'] for item,pr in selected]; actual=[r['id'] for r in results['rows']]
assert actual==expected[:len(actual)] and actual
assert a.partial or actual==expected
m=mujoco.MjModel.from_xml_path(str(a.asset)); d=mujoco.MjData(m)
# Local inverse-only model: no contacts/integration and no production asset edits.
m.geom_contype[:]=0; m.geom_conaffinity[:]=0
feet=[m.geom(s+'_foot_collision').id for s in ['left','right']]; rows=[]
for (item,pr),reported in zip(selected,results['rows']):
 assert item['id']==pr['id']==reported['id']; assert h(item['file'])==item['sha256']==pr['reference_sha256']
 correction=a.results.parent/item['id']/'final_correction.npz'
 with np.load(item['file']) as z, np.load(correction) as c:
  names=z['joint_names'].tolist(); lower=[i for i,n in enumerate(names) if n.startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot'))]
  assert len(lower)==14
  qa=[int(m.joint(n).qposadr[0]) for n in names]; va=[int(m.joint(n).dofadr[0]) for n in names]; ids=[m.joint(n).id for n in names]
  n=len(z['joint_pos']); dt=1/float(z['fps'].item()); x=c['delta']; assert x.shape==(n,17) and np.isfinite(x).all()
  original=np.zeros((n,m.nq)); original[:,:3]=z['root_pos']; original[:,3:7]=z['root_quat_wxyz']; original[:,qa]=z['joint_pos']
  q=original.copy(); q[:,:3]+=x[:,:3]; q[:,np.array(qa)[lower]]+=x[:,3:]
  cap=np.array([3. if i in lower else 2. for i in range(len(names))]); adjacent=np.abs(np.diff(q[:,qa],axis=0))/dt
  bounds=max(0,float((m.jnt_range[ids,0]-q[:,qa]).max()),float((q[:,qa]-m.jnt_range[ids,1]).max()))
  samples={s['frame']:s for g in pr['source_contact_support'] if g['mask']=='both_feet_phase_known' for s in g['samples']}
  coms=[]; oldcom=[]; poserr=[]; orierr=[]; heights=[]
  for i in range(n):
   d.qpos[:]=original[i]; mujoco.mj_kinematics(m,d); mujoco.mj_comPos(m,d)
   poses=[(d.geom_xpos[g].copy(),d.geom_xmat[g].reshape(3,3).copy()) for g in feet]
   if plan.get('restoration_contract') and i in samples:
    for k,side in enumerate(['left','right']):
     if side not in samples[i]['selected_sides']:continue
     position,rotation=poses[k];position=position.copy();position[2]=m.geom_size[feet[k],2]+.0002;yaw=np.arctan2(rotation[1,0],rotation[0,0]);poses[k]=(position,Rotation.from_euler('z',yaw).as_matrix())
   oldcom.append((d.xipos*m.body_mass[:,None]).sum(0)/m.body_mass.sum())
   d.qpos[:]=q[i]; mujoco.mj_kinematics(m,d); mujoco.mj_comPos(m,d)
   coms.append((d.xipos*m.body_mass[:,None]).sum(0)/m.body_mass.sum())
   poserr.append(max(np.linalg.norm(d.geom_xpos[g]-v[0]) for g,v in zip(feet,poses)) if not plan.get('restoration_contract') or i in samples else 0.)
   orierr.append(max(Rotation.from_matrix(v[1].T@d.geom_xmat[g].reshape(3,3)).magnitude() for g,v in zip(feet,poses)) if not plan.get('restoration_contract') or i in samples else 0.)
   height=[]
   for g in range(m.ngeom):
    if '_collision' not in m.geom(g).name: continue
    rot=d.geom_xmat[g].reshape(3,3); pos=d.geom_xpos[g]
    if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX: height.append(float(pos[2]-abs(rot[2])@m.geom_size[g]))
    else:
     assert m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
     mesh=m.geom_dataid[g]; verts=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
     height.append(float((verts@rot.T+pos)[:,2].min()))
   heights.append(min(height))
  rot=Rotation.from_quat(q[:,[4,5,6,3]]); w=np.zeros((n,3)); w[1:-1]=(rot[2:]*rot[:-2].inv()).as_rotvec()/(2*dt); w[0]=w[1]; w[-1]=w[-2]
  v=np.zeros((n,m.nv)); v[:,:3]=np.gradient(q[:,:3],dt,axis=0); v[:,3:6]=rot.inv().apply(w); v[:,va]=np.gradient(q[:,qa],dt,axis=0); acc=np.gradient(v,dt,axis=0)
  wrench=[]; errors=[]; olderrors=[]; coperrors=[]
  for i,s in samples.items():
   d.qpos[:]=q[i]; d.qvel[:]=v[i]; d.qacc[:]=acc[i]; mujoco.mj_inverse(m,d)
   force=d.qfrc_inverse[:3].copy(); torque=rot[i].apply(d.qfrc_inverse[3:6])+np.cross(q[i,:3],force)
   source_points=np.array(s['selected_hull_xy_m']);source_equations=ConvexHull(source_points).equations;points=source_points
   if plan.get('restoration_contract'):
    actual=[]
    for side in s['selected_sides']:
     g=m.geom(side+'_foot_collision').id;size=m.geom_size[g];local=np.array([[sx*size[0],sy*size[1],-size[2]] for sx in [-1,1] for sy in [-1,1]]);world=local@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];actual.extend(world[:,:2])
    points=np.array(actual)
   hull=ConvexHull(points); eq=hull.equations
   if force[2]>0:
    cop=np.array([-torque[1],torque[0]])/force[2]; margin=float(np.min(-(eq[:,:2]@cop+eq[:,2])/np.linalg.norm(eq[:,:2],axis=1)))
   else: cop=None; margin=None
   target=np.array(s['selected_hull_xy_m']).mean(0)
   if cop is not None:coperrors.append(cop-target)
   errors.append(np.array(coms[i])[:2]-target); olderrors.append(np.array(oldcom[i])[:2]-target)
   wrench.append(dict(frame=i,sides=s['selected_sides'],fz_n=float(force[2]),cop_xy_m=None if cop is None else cop.tolist(),margin_m=margin,actual_or_source_hull_xy_m=points.tolist(),original_source_hull_margin_m=None if cop is None else float(np.min(-(source_equations[:,:2]@cop+source_equations[:,2])/np.linalg.norm(source_equations[:,:2],axis=1))),outside=margin is None or margin < -1e-10))
  position=float(max(poserr)); orientation=float(max(orierr)); speed=float((adjacent/cap).max()); rms=float(np.sqrt(np.mean(np.square(errors))))
  if 'foot_position_vector_max_m' in reported:assert abs(position-reported['foot_position_vector_max_m'])<1e-10 and abs(orientation-reported['foot_orientation_vector_max_rad'])<1e-10
  if 'com_known_rms_m' in reported:assert abs(rms-reported['com_known_rms_m'])<1e-10
  if 'cop_known_rms_m' in reported:
   assert len(coperrors)==len(samples)
   assert abs(float(np.sqrt(np.mean(np.square(coperrors))))-reported['cop_known_rms_m'])<1e-10
  row=dict(id=item['id'],split=item['split'],frames=n,solver_status=reported['status'],correction_sha256=h(correction),source_sha256=item['sha256'],foot_position_max_m=position,foot_orientation_max_rad=orientation,joint_bound_violation_rad=bounds,adjacent_speed_ratio=speed,central_velocity_ratio=float((abs(v[:,va])/cap).max()),root_delta_max_per_axis_m=abs(x[:,:3]).max(0).tolist(),lower_delta_max_rad=float(abs(x[:,3:]).max()),collision_min_m=float(min(heights)),penetrating_frames=sum(t < -.0005 for t in heights),known_com_rms_before_m=float(np.sqrt(np.mean(np.square(olderrors)))),known_com_rms_after_m=rms,known_cop_rms_after_m=float(np.sqrt(np.mean(np.square(coperrors)))) if len(coperrors)==len(samples) else None,single_support_frames=sum(len(s['sides'])==1 for s in wrench),single_support_cop_outside=sum(len(s['sides'])==1 and s['outside'] for s in wrench),double_support_cop_outside=sum(len(s['sides'])==2 and s['outside'] for s in wrench),wrench_samples=wrench)
  row['numerical_constraints_pass']=bool(position<=1e-5 and orientation<=1e-4 and bounds<=1e-7 and speed<=1+1e-5/3 and np.all(abs(x[:,:3])<=[.0600001,.0600001,.0310001]) and abs(x[:,3:]).max()<=.8000001)
  rows.append(row); print(item['id'],{k:v for k,v in row.items() if k!='wrench_samples'},flush=True)
out=dict(cop_support_hull_contract='Actual corrected selected sole bottom corners' if plan.get('restoration_contract') else 'Original source-selected projected foot hull',foot_target_contract=plan.get('restoration_contract','Original both-foot world poses, all frames'),declared_source_split=plan.get('source_split','all'),complete_source_coverage=(actual==expected),expected_sources=len(expected),audited_sources=len(rows),time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='All original frames, final accepted iterate only. Independent mass-weighted COM, geometry, collision heights and contact-free inverse dynamics. No simulation or training acceptance; fixed original moving feet may remain airborne.',plan_sha256=h(a.plan),asset_sha256=h(a.asset),results_sha256=h(a.results),rows=rows)
a.output.write_text(json.dumps(out,indent=2)+'\n')
