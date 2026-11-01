"""Fit-only whole-sequence contact feasibility restoration; intermediate feet infeasible, never training NPZ."""
import argparse,hashlib,json
from pathlib import Path
import mujoco,numpy as np,scipy.sparse as sp,osqp
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);plan=json.loads(a.plan.read_text());srcpath=Path(plan['source_plan']);phasepath=Path(plan['phase_review']);assert hashlib.sha256(srcpath.read_bytes()).hexdigest()==plan['source_plan_sha256'];assert hashlib.sha256(phasepath.read_bytes()).hexdigest()==plan['phase_review_sha256'];src=json.loads(srcpath.read_text());phase=json.loads(phasepath.read_text());m=mujoco.MjModel.from_xml_path(str(a.asset));d=mujoco.MjData(m);feet=[m.geom(s+'_foot_collision').id for s in ['left','right']];base=m.body('base_link').id;reports=[];tol=np.tile(np.r_[np.repeat(1e-5,3),np.repeat(1e-4,3)],2)
def log_inverse(phi):
 t=np.linalg.norm(phi);x,y,z=phi;P=np.array([[0,-z,y],[z,0,-x],[-y,x,0]]);c=1/12 if t<1e-6 else 1/t**2-(1+np.cos(t))/(2*t*np.sin(t));return np.eye(3)-.5*P+c*P@P
for item,pr in zip(src['motions'][:1] if a.smoke else src['motions'],phase['rows']):
 assert item['id']==pr['id']
 if plan.get('source_split') and item['split']!=plan['source_split']:continue
 f=Path(item['file']);assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'];dest=a.output/item['id'];dest.mkdir();ref=np.load(f);names=ref['joint_names'].tolist();lower=[i for i,n in enumerate(names) if n.startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot'))];qa=[int(m.joint(n).qposadr[0]) for n in names];lqa=[qa[i] for i in lower];lv=[int(m.joint(names[i]).dofadr[0]) for i in lower];jids=[m.joint(names[i]).id for i in lower];cols=[0,1,2]+lv;n=len(ref['joint_pos']);dt=1/float(ref['fps'].item());assert dt==.02 and len(lower)==14;qbase=np.zeros((n,m.nq));qbase[:,:3]=ref['root_pos'];qbase[:,3:7]=ref['root_quat_wxyz'];qbase[:,qa]=ref['joint_pos'];footposes=[];targets=np.zeros((n,2));known=np.zeros(n)
 for i in range(n):
  d.qpos[:]=qbase[i];mujoco.mj_kinematics(m,d);footposes.append([(d.geom_xpos[g].copy(),d.geom_xmat[g].reshape(3,3).copy()) for g in feet])
 for s in next(g['samples'] for g in pr['source_contact_support'] if g['mask']=='both_feet_phase_known'):
  frame=s['frame'];targets[frame]=np.array(s['selected_hull_xy_m']).mean(0);known[frame]=1
  for k,side in enumerate(['left','right']):
   if side not in s['selected_sides']:continue
   pos,rot=footposes[frame][k];yaw=np.arctan2(rot[1,0],rot[0,0]);position=pos.copy();position[2]=m.geom_size[feet[k],2]+.0002;footposes[frame][k]=(position,Rotation.from_euler('z',yaw).as_matrix())
 D1=sp.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format='csc');D2=sp.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc');V=sp.kron(D1,sp.hstack([sp.csc_matrix((14,3)),sp.eye(14)],format='csc'),format='csc');original_delta=np.diff(ref['joint_pos'][:,lower].astype(float),axis=0).ravel();lo=np.column_stack([np.broadcast_to([-.06,-.06,-.031],(n,3)),np.maximum(-.8,m.jnt_range[jids,0]-qbase[:,lqa])]).ravel();hi=np.column_stack([np.broadcast_to([.06,.06,.031],(n,3)),np.minimum(.8,m.jnt_range[jids,1]-qbase[:,lqa])]).ravel();R=sp.vstack([sp.kron(sp.eye(n),sp.diags(np.r_[np.repeat(1/.05,3),np.repeat(1/.8,14)])),sp.kron(D1,sp.diags(np.r_[np.repeat(1/(dt*.2),3),np.repeat(1/dt,14)])),sp.kron(D2,sp.diags(np.r_[np.repeat(1/(dt*dt*2),3),np.repeat(1/(dt*dt*10),14)]))],format='csc');RtR=R.T@R;trust=np.tile(np.r_[np.repeat(.005,3),np.repeat(.05,14)],n);ftol=np.tile(tol,n)
 dynamic_cop=plan.get('tracking_objective')=='required_inverse_cop'
 if dynamic_cop:
  from supermimic_sequence_inverse_cop import SequenceInverseCOP
  engine=SequenceInverseCOP(a.asset,qbase,lqa,qa,[int(m.joint(name).dofadr[0]) for name in names],dt)
 def geometry(x,jacobian=True,objective_jacobian=True):
  x=x.reshape(n,17);fr=[];cr=[];jf=[];jc=[]
  for i in range(n):
   d.qpos[:]=qbase[i];d.qpos[:3]+=x[i,:3];d.qpos[lqa]+=x[i,3:];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);frows=[];jrows=[]
   for g,(position,rotation) in zip(feet,footposes[i]):
    phi=Rotation.from_matrix(rotation.T@d.geom_xmat[g].reshape(3,3)).as_rotvec();frows.extend([d.geom_xpos[g]-position,phi])
    if jacobian:
     jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacGeom(m,d,jp,jr,g);jrows.extend([jp[:,cols],log_inverse(phi)@rotation.T@jr[:,cols]])
   fr.append(np.concatenate(frows));cr.append((d.subtree_com[base,:2]-targets[i])*known[i]/.001)
   if jacobian:
    jcom=np.zeros((3,m.nv));mujoco.mj_jacSubtreeCom(m,d,jcom,base);jf.append(np.vstack(jrows));jc.append(jcom[:2,cols]*known[i]/.001)
  objective=np.concatenate(cr);objective_J=sp.block_diag(jc,format='csc') if jacobian else None
  if dynamic_cop:
   cop,force=engine.evaluate(x.reshape(n,17));objective=((cop-targets)*known[:,None]/.001).ravel()
   if jacobian and objective_jacobian:objective_J=sp.diags(np.repeat(known,2)/.001)@engine.jacobian(x.reshape(n,17))
  return np.concatenate(fr),objective,sp.block_diag(jf,format='csc') if jacobian else None,objective_J
 weights=np.repeat(known,12)/np.tile(tol,n);W=sp.diags(weights);regularizer=plan['restoration_prior_weight']*R;penalty=regularizer.T@regularizer
 with np.load(Path(plan['constraint_seed'])/(item['id']+'_constraint_seed.npz')) as z:x=z['delta'].ravel().copy()
 fr,_,Jf,_=geometry(x);direction=np.random.default_rng(42).normal(size=x.shape)*np.tile(np.r_[np.repeat(.0001,3),np.repeat(.001,14)],n);eps=1e-4;fp=geometry(x+eps*direction,False)[0];fm=geometry(x-eps*direction,False)[0];fd_error=float(abs((fp-fm)/(2*eps)-Jf@direction).max());assert fd_error<1e-7
 collision=None;collision_fd_error=None
 if plan.get('collision_constraints'):
  from supermimic_sequence_collision_height import SequenceCollisionHeight
  collision=SequenceCollisionHeight(m,qbase,lqa,cols)
  test=x+direction;heights,H=collision.evaluate(test);cp=collision.evaluate(test+eps*direction,False)[0];cm=collision.evaluate(test-eps*direction,False)[0];collision_fd_error=float(abs((cp-cm)/(2*eps)-H@direction).max());assert collision_fd_error<1e-7,collision_fd_error
 history=[];status='iteration_budget_exhausted';iterations=2 if a.smoke else plan['numerics']['iterations'];A=sp.vstack([sp.eye(n*17),V],format='csc')
 for iteration in range(iterations):
  old=float(np.linalg.norm(weights*fr)**2+np.linalg.norm(regularizer@x)**2);J=W@Jf;P=2*(J.T@J+penalty);gradient=2*(J.T@(weights*fr)+penalty@x);l=np.r_[np.maximum(lo-x,-trust),-.06-original_delta-V@x];u=np.r_[np.minimum(hi-x,trust),.06-original_delta-V@x]
  A=sp.vstack([sp.eye(n*17),V],format='csc')
  if collision is not None:
   heights,H=collision.evaluate(x);A=sp.vstack([A,H],format='csc');l=np.r_[l,-.0005+1e-6-heights];u=np.r_[u,np.repeat(np.inf,len(heights))]
  cs=1/np.sqrt(np.maximum(P.diagonal(),1e-12));C=sp.diags(cs);B=A@C;rs=1/np.maximum(np.asarray(abs(B).max(axis=1).toarray()).ravel(),1e-12);D=sp.diags(rs);solver=osqp.OSQP();solver.setup(P=sp.triu(C@P@C,format='csc'),q=cs*gradient,A=(D@B).tocsc(),l=rs*l,u=rs*u,verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=100000,polishing=True);sol=solver.solve();entry=dict(iteration=iteration,qp_status=sol.info.status,qp_iterations=sol.info.iter,merit_before=old)
  if sol.info.status_val!=1:
   status='qp_failed_preserve';history.append(entry);sp.save_npz(dest/'failed_P.npz',P);sp.save_npz(dest/'failed_A.npz',A);np.savez_compressed(dest/'failed_vectors.npz',gradient=gradient,lower=l,upper=u);break
  step=cs*sol.x;violation=float(max(0,(l-A@step).max(),(A@step-u).max()));entry['unscaled_constraint_violation']=violation
  if violation>1e-7:status='linear_constraint_failed';history.append(entry);break
  accepted=False
  for h in range(11):
   alpha=2.**(-h);candidate=x+alpha*step;nf=geometry(candidate,False)[0];new=float(np.linalg.norm(weights*nf)**2+np.linalg.norm(regularizer@candidate)**2)
   if collision is not None:
    candidate_heights=collision.evaluate(candidate,False)[0]
    if candidate_heights.min()<-.0005:continue
   if new<old-1e-9:
    x=candidate;fr,_,Jf,_=geometry(x);accepted=True;entry.update(alpha=alpha,merit_after=new);break
  if collision is not None:entry['nonlinear_collision_min_m']=float(collision.evaluate(x,False)[0].min())
  vectors=fr.reshape(n,2,6)[known.astype(bool)];position=float(np.linalg.norm(vectors[:,:,:3],axis=2).max());orientation=float(np.linalg.norm(vectors[:,:,3:],axis=2).max());entry.update(known_foot_position_max_m=position,known_foot_orientation_max_rad=orientation);history.append(entry);np.savez_compressed(dest/('iteration_%02d.npz'%iteration),delta=x.reshape(n,17));(dest/'history.json').write_text(json.dumps(history,indent=2)+'\n');print(item['id'],entry,flush=True)
  if not accepted:status='line_search_no_progress';break
  actual_violation=float(max(0,(lo-x).max(),(x-hi).max(),(-.06-original_delta-V@x).max(),(V@x-(.06-original_delta)).max()));assert actual_violation<=1e-7
  if position<=1e-5 and orientation<=1e-4:status='foot_target_tolerances_reached_pending_independent_audit';break
 (dest/'history.json').write_text(json.dumps(history,indent=2)+'\n');np.savez_compressed(dest/'final_correction.npz',delta=x.reshape(n,17));q=ref['joint_pos'][:,lower].astype(float)+x.reshape(n,17)[:,3:];speed=float((abs(np.diff(q,axis=0))/.06).max());assert speed<=1+1e-5/3;row=dict(id=item['id'],split=item['split'],frames=n,known_frames=int(known.sum()),status=status,iterations=len(history),speed_ratio=speed,directional_foot_error=fd_error,collision_directional_error=collision_fd_error);reports.append(row);(a.output/'partial.json').write_text(json.dumps(dict(rows=reports),indent=2)+'\n');ref.close()
(a.output/'results.json').write_text(json.dumps(dict(scope='Feasibility restoration diagnostic only; last iterate, all failures retained. Feet intermediate may be infeasible. No dynamics/physics/training acceptance.',rows=reports),indent=2)+'\n')
