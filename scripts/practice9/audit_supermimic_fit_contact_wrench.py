"""All-fit necessary full-wrench/friction and source-support geometry diagnostic."""
import argparse,datetime,hashlib,itertools,json
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser(description=__doc__)
for k in ['plan','asset','state','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();h=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();plan=json.loads(a.plan.read_text());assert h(plan['source_plan'])==plan['source_plan_sha256'] and h(plan['phase_review'])==plan['phase_review_sha256'];sources=json.loads(Path(plan['source_plan']).read_text());phases=json.loads(Path(plan['phase_review']).read_text());m=mujoco.MjModel.from_xml_path(str(a.asset));m.geom_contype[:]=0;m.geom_conaffinity[:]=0;d=mujoco.MjData(m);rows=[]
def cross(v):
 x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
for item,pr in zip(sources['motions'],phases['rows']):
 assert item['id']==pr['id']
 if item['split']!='fit':continue
 assert h(item['file'])==item['sha256']
 with np.load(item['file']) as z:
  names=z['joint_names'].tolist();qa=[int(m.joint(n).qposadr[0]) for n in names];va=[int(m.joint(n).dofadr[0]) for n in names];lower=[qa[i] for i,n in enumerate(names) if n.startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot'))];n=len(z['joint_pos']);dt=1/float(z['fps'].item());original=np.zeros((n,m.nq));original[:,:3]=z['root_pos'];original[:,3:7]=z['root_quat_wxyz'];original[:,qa]=z['joint_pos'];samples=next(g['samples'] for g in pr['source_contact_support'] if g['mask']=='both_feet_phase_known')
  for arm in ['original','retracted_com_fit20','retracted_cop_fit20']:
   q=original.copy();correction_hash=None
   if arm!='original':
    cp=a.state/arm/item['id']/'final_correction.npz';correction_hash=h(cp)
    with np.load(cp) as c:x=c['delta'];q[:,:3]+=x[:,:3];q[:,lower]+=x[:,3:]
   rot=Rotation.from_quat(q[:,[4,5,6,3]]);w=np.zeros((n,3));w[1:-1]=(rot[2:]*rot[:-2].inv()).as_rotvec()/(2*dt);w[0]=w[1];w[-1]=w[-2];v=np.zeros((n,m.nv));v[:,:3]=np.gradient(q[:,:3],dt,axis=0);v[:,3:6]=rot.inv().apply(w);v[:,va]=np.gradient(q[:,qa],dt,axis=0);acc=np.gradient(v,dt,axis=0);records=[]
   for sample in samples:
    i=sample['frame'];d.qpos[:]=q[i];d.qvel[:]=v[i];d.qacc[:]=acc[i];mujoco.mj_inverse(m,d);force=d.qfrc_inverse[:3].copy();torque=rot[i].apply(d.qfrc_inverse[3:6])+np.cross(q[i,:3],force);points=[];coeff=[];geometry=[]
    for side in sample['selected_sides']:
     g=m.geom(side+'_foot_collision').id;R=d.geom_xmat[g].reshape(3,3);corners=np.array([[sx*m.geom_size[g,0],sy*m.geom_size[g,1],-m.geom_size[g,2]] for sx,sy in itertools.product([-1,1],repeat=2)])@R.T+d.geom_xpos[g];mu=float(m.geom_friction[g,0]);geometry.append(dict(side=side,bottom_corners_w_m=corners.tolist(),min_height_m=float(corners[:,2].min()),max_height_m=float(corners[:,2].max()),asset_sliding_friction=mu));corners[:,2]=0;points.extend(corners);coeff.extend([mu]*4)
    count=len(points);Aeq=np.vstack([np.tile(np.eye(3),(1,count)),np.hstack([cross(p) for p in points])]);b=np.r_[force,torque];A=[]
    for corner,mu in enumerate(coeff):
     for theta in np.arange(8)*np.pi/4:
      row=np.zeros(3*count);row[corner*3:corner*3+3]=[np.cos(theta),np.sin(theta),-mu];A.append(row)
    sol=linprog(np.zeros(3*count),A_ub=np.array(A),b_ub=np.zeros(len(A)),A_eq=Aeq,b_eq=b,bounds=[bound for _ in points for bound in [(None,None),(None,None),(0,None)]],method='highs')
    residual=None;true_cone_violation=None
    if sol.success:
     residual=float(abs(Aeq@sol.x-b).max());cf=sol.x.reshape(-1,3);true_cone_violation=float(np.maximum(0,np.linalg.norm(cf[:,:2],axis=1)-np.array(coeff)*cf[:,2]).max())
    records.append(dict(frame=i,sides=sample['selected_sides'],required_force_n=force.tolist(),required_torque_origin_nm=torque.tolist(),resultant_friction_ratio=float(np.linalg.norm(force[:2])/force[2]) if force[2]>0 else None,optimistic_projected_corner_lp_status=int(sol.status),optimistic_projected_corner_lp_success=bool(sol.success),equilibrium_residual=residual,lp_solution_circular_friction_violation_n=true_cone_violation,geometry=geometry))
   row=dict(id=item['id'],arm=arm,source_sha256=item['sha256'],correction_sha256=correction_hash,known_frames=len(records),optimistic_lp_infeasible=sum(r['optimistic_projected_corner_lp_status']==2 for r in records),optimistic_lp_other_failure=sum(r['optimistic_projected_corner_lp_status'] not in [0,2] for r in records),single_support_min_corner_above_5mm=sum(len(r['sides'])==1 and min(g['min_height_m'] for g in r['geometry'])>.005 for r in records),single_support_frames=sum(len(r['sides'])==1 for r in records),resultant_friction_ratio_max=max(r['resultant_friction_ratio'] for r in records if r['resultant_friction_ratio'] is not None),samples=records);rows.append(row);print({k:v for k,v in row.items() if k!='samples'},flush=True)
a.output.write_text(json.dumps(dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='All fit known phases; optimistic ground-projected full-foot corner forces with8-direction OUTER friction approximation. LP infeasibility is a necessary-test failure; success does not prove actual contacts, circular cone, joint torques, actuator tracking or stable control. Uses asset friction, not randomized runtime terrain pair.',asset_sha256=h(a.asset),plan_sha256=h(a.plan),rows=rows),indent=2)+'\n')
