"""Full-frame sparse sequential QP diagnostic; outputs corrections, not training NPZ."""
import argparse,hashlib,json
from pathlib import Path
import mujoco,numpy as np,scipy.sparse as sp,osqp
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);plan=json.loads(a.plan.read_text());srcpath=Path(plan['source_plan']);phasepath=Path(plan['phase_review']);assert hashlib.sha256(srcpath.read_bytes()).hexdigest()==plan['source_plan_sha256'];assert hashlib.sha256(phasepath.read_bytes()).hexdigest()==plan['phase_review_sha256'];src=json.loads(srcpath.read_text());phase=json.loads(phasepath.read_text());m=mujoco.MjModel.from_xml_path(str(a.asset));d=mujoco.MjData(m);feet=[m.geom(s+'_foot_collision').id for s in ['left','right']];base=m.body('base_link').id;reports=[];tol=np.tile(np.r_[np.repeat(1e-5,3),np.repeat(1e-4,3)],2)
def log_inverse(phi):
 t=np.linalg.norm(phi);x,y,z=phi;P=np.array([[0,-z,y],[z,0,-x],[-y,x,0]]);c=1/12 if t<1e-6 else 1/t**2-(1+np.cos(t))/(2*t*np.sin(t));return np.eye(3)-.5*P+c*P@P
for item,pr in zip(src['motions'][:1] if a.smoke else src['motions'],phase['rows']):
 assert item['id']==pr['id'];f=Path(item['file']);assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'];dest=a.output/item['id'];dest.mkdir();ref=np.load(f);names=ref['joint_names'].tolist();lower=[i for i,n in enumerate(names) if n.startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot'))];qa=[int(m.joint(n).qposadr[0]) for n in names];lqa=[qa[i] for i in lower];lv=[int(m.joint(names[i]).dofadr[0]) for i in lower];jids=[m.joint(names[i]).id for i in lower];cols=[0,1,2]+lv;n=len(ref['joint_pos']);dt=1/float(ref['fps'].item());assert dt==.02 and len(lower)==14;qbase=np.zeros((n,m.nq));qbase[:,:3]=ref['root_pos'];qbase[:,3:7]=ref['root_quat_wxyz'];qbase[:,qa]=ref['joint_pos'];footposes=[];targets=np.zeros((n,2));known=np.zeros(n)
 for i in range(n):
  d.qpos[:]=qbase[i];mujoco.mj_kinematics(m,d);footposes.append([(d.geom_xpos[g].copy(),d.geom_xmat[g].reshape(3,3).copy()) for g in feet])
 for s in next(g['samples'] for g in pr['source_contact_support'] if g['mask']=='both_feet_phase_known'):targets[s['frame']]=np.array(s['selected_hull_xy_m']).mean(0);known[s['frame']]=1
 D1=sp.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format='csc');D2=sp.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc');V=sp.kron(D1,sp.hstack([sp.csc_matrix((14,3)),sp.eye(14)],format='csc'),format='csc');original_delta=np.diff(ref['joint_pos'][:,lower].astype(float),axis=0).ravel();lo=np.column_stack([np.broadcast_to([-.06,-.06,-.031],(n,3)),np.maximum(-.8,m.jnt_range[jids,0]-qbase[:,lqa])]).ravel();hi=np.column_stack([np.broadcast_to([.06,.06,.031],(n,3)),np.minimum(.8,m.jnt_range[jids,1]-qbase[:,lqa])]).ravel();R=sp.vstack([sp.kron(sp.eye(n),sp.diags(np.r_[np.repeat(1/.05,3),np.repeat(1/.8,14)])),sp.kron(D1,sp.diags(np.r_[np.repeat(1/(dt*.2),3),np.repeat(1/dt,14)])),sp.kron(D2,sp.diags(np.r_[np.repeat(1/(dt*dt*2),3),np.repeat(1/(dt*dt*10),14)]))],format='csc');RtR=R.T@R;trust=np.tile(np.r_[np.repeat(.005,3),np.repeat(.05,14)],n);ftol=np.tile(tol,n)
 def geometry(x,jacobian=True):
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
  return np.concatenate(fr),np.concatenate(cr),sp.block_diag(jf,format='csc') if jacobian else None,sp.block_diag(jc,format='csc') if jacobian else None
 def merit(x,fr,cr):return float(cr@cr+np.linalg.norm(R@x)**2+10000*np.maximum(abs(fr)/ftol-1,0)@np.maximum(abs(fr)/ftol-1,0))
 with np.load(Path(plan['constraint_seed'])/(item['id']+'_constraint_seed.npz')) as z:x=z['delta'].ravel().copy()
 fr,cr,Jf,Jc=geometry(x);direction=np.tile(np.r_[[.0001,-.0001,.0001],np.repeat(.001,14)],n);eps=1e-5;fp,cp,_,_=geometry(x+eps*direction,False);fm,cm,_,_=geometry(x-eps*direction,False);fd_error=float(abs((fp-fm)/(2*eps)-Jf@direction).max());com_fd_error=float(abs((cp-cm)/(2*eps)-Jc@direction).max());assert fd_error<1e-7 and com_fd_error<1e-4;history=[];status='iteration_budget_exhausted';iterations=2 if a.smoke else plan['numerics']['iterations']
 for iteration in range(iterations):
  old=merit(x,fr,cr);P=2*(Jc.T@Jc+RtR);gradient=2*(Jc.T@cr+RtR@x);A=sp.vstack([sp.eye(n*17,format='csc'),V,sp.diags(1/ftol)@Jf],format='csc');l=np.r_[np.maximum(lo-x,-trust),-.06-original_delta-V@x,-1-fr/ftol];u=np.r_[np.minimum(hi-x,trust),.06-original_delta-V@x,1-fr/ftol];particular=np.zeros(n*17);null=None
  if plan.get('linearized_foot_formulation')=='exact_nullspace':
   blocks=[];scaledJ=sp.diags(1/ftol)@Jf
   for i in range(n):
    block=scaledJ[i*12:(i+1)*12,i*17:(i+1)*17].toarray();U,S,VT=np.linalg.svd(block,full_matrices=True);rank=int(np.sum(S>S[0]*1e-10));particular[i*17:(i+1)*17]=-VT[:rank].T@((U[:,:rank].T@(fr[i*12:(i+1)*12]/tol))/S[:rank]);blocks.append(sp.csc_matrix(VT[rank:].T))
   null=sp.block_diag(blocks,format='csc');cut=n*17+14*(n-1);rawA=A[:cut];H=(null.T@P@null).tocsc();g=null.T@(P@particular+gradient);B=(rawA@null).tocsc();ll=l[:cut]-rawA@particular;uu=u[:cut]-rawA@particular;cs=1/np.sqrt(np.maximum(H.diagonal(),1e-12));C=sp.diags(cs);B=B@C;rs=1/np.maximum(np.asarray(abs(B).max(axis=1).toarray()).ravel(),1e-12);D=sp.diags(rs);solver=osqp.OSQP();solver.setup(P=sp.triu(C@H@C,format='csc'),q=cs*g,A=(D@B).tocsc(),l=rs*ll,u=rs*uu,verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=20000,polishing=True);sol=solver.solve();step=particular+null@(cs*sol.x)
  else:
   solver=osqp.OSQP();solver.setup(P=sp.triu(P,format='csc'),q=gradient,A=A,l=l,u=u,verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=20000,polishing=True);sol=solver.solve();step=sol.x
  entry=dict(iteration=iteration,solver_status=sol.info.status,solver_iterations=sol.info.iter,primal_residual=sol.info.prim_res,dual_residual=sol.info.dual_res,merit_before=old)
  if sol.info.status_val!=1:
   status='qp_failed_preserve';history.append(entry);sp.save_npz(dest/'failed_qp_P.npz',P);sp.save_npz(dest/'failed_qp_A.npz',A);np.savez_compressed(dest/'failed_qp_vectors.npz',gradient=gradient,lower=l,upper=u,solver_x=sol.x);break
  linear_violation=float(max(0,(l-A@step).max(),(A@step-u).max()));entry['unscaled_linear_constraint_violation']=linear_violation
  if linear_violation>1e-7:status='unscaled_linear_constraints_failed';history.append(entry);break
  accepted=False
  for h in range(11):
   alpha=2.**(-h);candidate=x+alpha*step;nf,nc,_,_=geometry(candidate,False);new=merit(candidate,nf,nc)
   if new<old-1e-9:
    x=candidate;fr,cr,Jf,Jc=geometry(x);accepted=True;entry.update(alpha=alpha,merit_after=new,foot_position_max_m=float(abs(fr.reshape(n,12)[:,[0,1,2,6,7,8]]).max()),foot_orientation_max_rad=float(abs(fr.reshape(n,12)[:,[3,4,5,9,10,11]]).max()),com_known_rms_m=float(np.sqrt(np.mean(cr.reshape(n,2)[known.astype(bool)]**2))*.001));break
  history.append(entry);np.savez_compressed(dest/('iteration_%02d.npz'%iteration),delta=x.reshape(n,17));(dest/'history.json').write_text(json.dumps(history,indent=2)+'\n');print(item['id'],entry,flush=True)
  if not accepted:status='line_search_no_progress';break
  violation=float(max(0,(lo-x).max(),(x-hi).max(),(-.06-original_delta-V@x).max(),(V@x-(.06-original_delta)).max()));assert violation<=1e-7,violation
 (dest/'history.json').write_text(json.dumps(history,indent=2)+'\n');np.savez_compressed(dest/'final_correction.npz',delta=x.reshape(n,17));q=ref['joint_pos'][:,lower].astype(float)+x.reshape(n,17)[:,3:];speed=float((abs(np.diff(q,axis=0))/.06).max());assert speed<=1+1e-5/3;row=dict(id=item['id'],split=item['split'],frames=n,status=status,iterations=len(history),directional_foot_jacobian_error=fd_error,directional_com_jacobian_error=com_fd_error,speed_ratio=speed,foot_position_component_max_m=float(abs(fr.reshape(n,12)[:,[0,1,2,6,7,8]]).max()),com_known_rms_m=float(np.sqrt(np.mean(cr.reshape(n,2)[known.astype(bool)]**2))*.001));reports.append(row);(a.output/'partial.json').write_text(json.dumps(dict(rows=reports),indent=2)+'\n');ref.close()
(a.output/'results.json').write_text(json.dumps(dict(scope='Sequence optimization diagnostics only. Corrections are not training references; nonlinear geometry/collision/full derivatives/wrench/physics acceptance still required.',rows=reports),indent=2)+'\n');print('SEQUENCE_DIAGNOSTIC_FINISHED',reports)
