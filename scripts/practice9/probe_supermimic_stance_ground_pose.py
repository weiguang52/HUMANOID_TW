"""Fit-only stance-foot grounding and flattening reachability; never training references."""
import argparse,hashlib,json
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser(description=__doc__)
for k in ['plan','source-plan','asset','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);plan=json.loads(a.plan.read_text());phasepath=Path(plan['source_phase_review']);assert hashlib.sha256(phasepath.read_bytes()).hexdigest()==plan['source_phase_sha256'];phase=json.loads(phasepath.read_text());sources=json.loads(a.source_plan.read_text());m=mujoco.MjModel.from_xml_path(str(a.asset));d=mujoco.MjData(m);feet=[m.geom(s+'_foot_collision').id for s in ['left','right']];base=m.body('base_link').id;reports=[];fd_errors=[]
def skew(v):return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
def left_inverse(phi):
 theta=np.linalg.norm(phi);P=skew(phi);c=1/12 if theta<1e-6 else 1/theta**2-(1+np.cos(theta))/(2*theta*np.sin(theta));return np.eye(3)-.5*P+c*P@P
try:
 for item,review in zip(sources['motions'],phase['rows']):
  assert item['id']==review['id']
  if item['split']!='fit':continue
  f=Path(item['file']);assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256']==review['reference_sha256']
  with np.load(f) as z:
   names=z['joint_names'].tolist();lower=[n for n in names if n.startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot'))];assert len(lower)==14;qa=[int(m.joint(n).qposadr[0]) for n in names];lqa=[int(m.joint(n).qposadr[0]) for n in lower];lva=[int(m.joint(n).dofadr[0]) for n in lower];jids=[m.joint(n).id for n in lower];cols=[0,1,2]+lva
   selected=next(g['samples'] for g in review['source_contact_support'] if g['mask']=='both_feet_phase_known')
   for sample in selected:
    frame=sample['frame'];d.qpos[:3]=z['root_pos'][frame];d.qpos[3:7]=z['root_quat_wxyz'][frame];d.qpos[qa]=z['joint_pos'][frame];qbase=d.qpos.copy();mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);poses=[(d.geom_xpos[g].copy(),d.geom_xmat[g].reshape(3,3).copy()) for g in feet];target=np.array(sample['selected_hull_xy_m']).mean(0)
    for foot_index,side in enumerate(['left','right']):
     if side not in sample['selected_sides']:continue
     pos,rot=poses[foot_index];yaw=np.arctan2(rot[1,0],rot[0,0]);flat=Rotation.from_euler('z',yaw).as_matrix();grounded=pos.copy();grounded[2]=m.geom_size[feet[foot_index],2]+.0002;poses[foot_index]=(grounded,flat)
    bounds_low=np.r_[[-.06,-.06,-.031],np.maximum(-.8,m.jnt_range[jids,0]-qbase[lqa])];bounds_high=np.r_[[.06,.06,.031],np.minimum(.8,m.jnt_range[jids,1]-qbase[lqa])];assert np.all(bounds_low<bounds_high);cache={}
    def calc(x):
     if 'x' in cache and np.array_equal(x,cache['x']):return cache['r'],cache['J']
     d.qpos[:]=qbase;d.qpos[:3]+=x[:3];d.qpos[lqa]+=x[3:];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);rs=[];js=[]
     for g,(position,rotation) in zip(feet,poses):
      jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacGeom(m,d,jp,jr,g);phi=Rotation.from_matrix(rotation.T@d.geom_xmat[g].reshape(3,3)).as_rotvec();rs.extend([(d.geom_xpos[g]-position)/.0001,phi/.001]);js.extend([jp[:,cols]/.0001,left_inverse(phi)@rotation.T@jr[:,cols]/.001])
     jc=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,jc,base);rs.extend([1e-4*x[:3]/.05,1e-4*x[3:]/.8]);js.extend([1e-4*np.column_stack([np.eye(3)/.05,np.zeros((3,14))]),1e-4*np.column_stack([np.zeros((14,3)),np.eye(14)/.8])]);res=np.concatenate(rs);jac=np.vstack(js);cache.update(x=x.copy(),r=res,J=jac);return res,jac
    x0=np.clip(np.zeros(17),bounds_low+1e-9,bounds_high-1e-9)
    if len(fd_errors)<8 and not any(v['id']==item['id'] for v in fd_errors):
     test=np.clip(x0+np.r_[[.0001,-.0002,.0001],np.repeat(.002,14)],bounds_low+1e-7,bounds_high-1e-7);analytic=calc(test)[1].copy();numeric=np.zeros_like(analytic)
     for k in range(17):
      dx=np.zeros(17);dx[k]=1e-7;numeric[:,k]=(calc(test+dx)[0]-calc(test-dx)[0])/(2e-7)
     error=float(abs(analytic-numeric).max());assert error<1e-4,error;fd_errors.append(dict(id=item['id'],frame=frame,scaled_residual_jacobian_max_error=error))
    sol=least_squares(lambda x:calc(x)[0],x0,jac=lambda x:calc(x)[1],bounds=(bounds_low,bounds_high),max_nfev=200,ftol=1e-9,xtol=1e-9,gtol=1e-9);calc(sol.x);position_error=max(float(np.linalg.norm(d.geom_xpos[g]-pos)) for g,(pos,rot) in zip(feet,poses));orientation_error=max(float(Rotation.from_matrix(rot.T@d.geom_xmat[g].reshape(3,3)).magnitude()) for g,(pos,rot) in zip(feet,poses));com_error=float(np.linalg.norm(d.subtree_com[base,:2]-target));minimum=[]
    for g in range(m.ngeom):
     if '_collision' not in m.geom(g).name:continue
     rot=d.geom_xmat[g].reshape(3,3);pos=d.geom_xpos[g]
     if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:height=float(pos[2]-abs(rot[2])@m.geom_size[g])
     else:
      assert m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH;k=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]];height=float((v@rot.T+pos)[:,2].min())
     minimum.append(height)
    row=dict(id=item['id'],split=item['split'],source_family=item['source_family'],reference_sha256=item['sha256'],frame=frame,support_sides=sample['selected_sides'],solver_success=bool(sol.success),nfev=sol.nfev,foot_position_error_m=position_error,foot_orientation_error_rad=orientation_error,com_target_error_m=com_error,collision_min_height_m=min(minimum),root_translation_m=sol.x[:3].tolist(),lower_joint_delta_rad=sol.x[3:].tolist(),target_foot_positions_m=[v[0].tolist() for v in poses],target_foot_rotations=[v[1].tolist() for v in poses],geometry_numerical_check=bool(sol.success and position_error<=1e-5 and orientation_error<=1e-4));reports.append(row)
    if len(reports)%20==0:(a.output/'partial.json').write_text(json.dumps(dict(rows=reports,fd_checks=fd_errors),indent=2)+'\n');print('completed',len(reports),flush=True)
 assert len(reports)==545
 (a.output/'results.json').write_text(json.dumps(dict(scope='Pointwise fit-only stance feet ground-flat; swing original. No COM objective, temporal/speed/wrench or training acceptance. com_target_error_m is descriptive only.',plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),asset_sha256=hashlib.sha256(a.asset.read_bytes()).hexdigest(),fd_checks=fd_errors,rows=reports),indent=2)+'\n');(a.output/'status').write_text('completed_pending_geometric_review_not_trajectory_acceptance\n');print('DONE',len(reports),sum(r['geometry_numerical_check'] for r in reports))
except BaseException:
 (a.output/'status').write_text('failed_preserve_outputs\n');(a.output/'partial.json').write_text(json.dumps(dict(rows=reports,fd_checks=fd_errors),indent=2)+'\n');raise
