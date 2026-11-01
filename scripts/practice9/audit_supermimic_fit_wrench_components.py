"""Fit-only COM/linear acceleration/angular momentum COP decomposition."""
import argparse,datetime,hashlib,json
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial import ConvexHull
from supermimic_reference_wrench import infer_wrench,ground_cop
p=argparse.ArgumentParser(description=__doc__)
for k in ['plan','asset','corrections','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();h=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest();plan=json.loads(a.plan.read_text());assert h(plan['source_plan'])==plan['source_plan_sha256'] and h(plan['phase_review'])==plan['phase_review_sha256'];src=json.loads(Path(plan['source_plan']).read_text());phase=json.loads(Path(plan['phase_review']).read_text());m=mujoco.MjModel.from_xml_path(str(a.asset));d=mujoco.MjData(m);rows=[]
for item,pr in zip(src['motions'],phase['rows']):
 assert item['id']==pr['id']
 if item['split']!='fit':continue
 assert h(item['file'])==item['sha256'];correction=a.corrections/item['id']/'final_correction.npz'
 with np.load(item['file']) as z,np.load(correction) as c:
  names=z['joint_names'].tolist();qa=[int(m.joint(n).qposadr[0]) for n in names];lower=[i for i,n in enumerate(names) if n.startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot'))];n=len(z['joint_pos']);dt=1/float(z['fps'].item());q=np.zeros((n,m.nq));q[:,:3]=z['root_pos'];q[:,3:7]=z['root_quat_wxyz'];q[:,qa]=z['joint_pos'];x=c['delta'];versions={}
  for label in ['original','corrected']:
   qq=q.copy()
   if label=='corrected':qq[:,:3]+=x[:,:3];qq[:,np.array(qa)[lower]]+=x[:,3:]
   pos=[];rot=[]
   for v in qq:
    d.qpos[:]=v;mujoco.mj_kinematics(m,d);pos.append(d.xipos.copy());rot.append(d.ximat.reshape(m.nbody,3,3).copy())
   com,H,F,T=infer_wrench(np.array(pos),np.array(rot),m.body_mass,m.body_inertia,dt,m.opt.gravity);cop=ground_cop(F,T);hdot=np.gradient(H,dt,axis=0);linear=-com[:,2,None]*F[:,:2]/F[:,2,None];angular=np.column_stack([-hdot[:,1],hdot[:,0]])/F[:,2,None];identity=float(abs(cop-com[:,:2]-linear-angular).max());assert identity<1e-12
   samples=[]
   for group in pr['source_contact_support']:
    if group['mask']!='both_feet_phase_known':continue
    for s in group['samples']:
     if len(s['selected_sides'])!=1:continue
     i=s['frame'];eq=ConvexHull(np.array(s['selected_hull_xy_m'])).equations
     margin=lambda p:float(np.min(-(eq[:,:2]@p+eq[:,2])/np.linalg.norm(eq[:,:2],axis=1)))
     samples.append(dict(frame=i,com_margin_m=margin(com[i,:2]),cop_margin_m=margin(cop[i]),linear_cop_shift_m=linear[i].tolist(),angular_cop_shift_m=angular[i].tolist(),com_xy_m=com[i,:2].tolist(),cop_xy_m=cop[i].tolist(),force_z_n=float(F[i,2])))
   versions[label]=dict(identity_error_m=identity,known_single_frames=len(samples),com_outside=sum(v['com_margin_m'] < -1e-10 for v in samples),cop_outside=sum(v['cop_margin_m'] < -1e-10 for v in samples),linear_shift_norm_median_m=float(np.median([np.linalg.norm(v['linear_cop_shift_m']) for v in samples])),angular_shift_norm_median_m=float(np.median([np.linalg.norm(v['angular_cop_shift_m']) for v in samples])),samples=samples)
  rows.append(dict(id=item['id'],split=item['split'],source_sha256=item['sha256'],correction_sha256=h(correction),versions=versions));print(item['id'],{k:{kk:vv for kk,vv in v.items() if kk!='samples'} for k,v in versions.items()},flush=True)
a.output.write_text(json.dumps(dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='All four predeclared fit sources, original and final fixed-budget corrections; no validation tuning, no training export. COP decomposition is descriptive, not controller authority or hardware proof.',asset_sha256=h(a.asset),plan_sha256=h(a.plan),rows=rows),indent=2)+'\n')
