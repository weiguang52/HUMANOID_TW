"""Actual loaded-contact Jacobian rank and FD check; no dynamics/gain acceptance."""
import argparse,hashlib,json
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import lsq_linear
p=argparse.ArgumentParser(description=__doc__)
for k in ['collection','early-review','asset','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();m=mujoco.MjModel.from_xml_path(str(a.asset));d=mujoco.MjData(m);meta=json.loads(a.early_review.read_text())['rows'];episodes=json.loads((a.collection/'results.json').read_text())['rows'];mapping=json.loads((a.collection/'control_mapping.json').read_text());names=mapping['state_joint_names'];qadr=[int(m.joint(n).qposadr[0]) for n in names];lower=[n for n in names if n.startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot'))];cols=[int(m.joint(n).dofadr[0]) for n in lower];assert len(cols)==14;reports=[];fdmax=0.;fdchecks=0
lower_jids=[m.joint(n).id for n in lower];lower_qadr=[int(m.joint(n).qposadr[0]) for n in lower];assert all(m.jnt_limited[j] for j in lower_jids)
for row,review in zip(episodes,meta):
 assert (row['id'],row['frame'])==(review['id'],review['frame']);trace=a.collection/row['trace'];assert hashlib.sha256(trace.read_bytes()).hexdigest()==row['sha256'];contacts=json.loads((a.collection/row['native_contact_file']).read_text())['samples'];fd_done=False;samples=[]
 with np.load(trace) as z:
  for i in range(review['pre_root10deg_steps']):
   state=z['next_state'][i];d.qpos[:3]=state[:3];d.qpos[3:7]=state[3:7];d.qpos[qadr]=state[13:43];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);qbase=d.qpos.copy();mat=[]
   for foot in contacts[i]['feet'].values():
    for contact in foot['contacts']:
     if not contact['ground'] or contact['normal_force_n']<=.01:continue
     geom=next(g for g in contact['geoms'] if '_foot_collision' in g);g=m.geom(geom.replace('robot/','')).id;body=int(m.geom_bodyid[g]);point=np.array(contact['pos_w']);jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,point,body);mat.append(jp)
     if not fd_done:
      local=d.xmat[body].reshape(3,3).T@(point-d.xpos[body]);numeric=np.zeros_like(jp);eps=1e-6
      for c in range(m.nv):
       values=[]
       for sign in [1,-1]:
        d.qpos[:]=qbase;v=np.zeros(m.nv);v[c]=sign;mujoco.mj_integratePos(m,d.qpos,v,eps);mujoco.mj_kinematics(m,d);values.append(d.xpos[body]+d.xmat[body].reshape(3,3)@local)
       numeric[:,c]=(values[0]-values[1])/(2*eps)
      error=float(abs(numeric-jp).max());fdmax=max(fdmax,error);fdchecks+=1;assert error<1e-6,error;d.qpos[:]=qbase;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);fd_done=True
   if not mat:samples.append(dict(step=i+1,contacts=0,scope='No loaded foot constraint'));continue
   j=np.concatenate(mat);jl=j[:,cols];target=-j[:,3:5];solution,_,rank,sv=np.linalg.lstsq(jl,target,rcond=1e-8);residual=np.linalg.norm(jl@solution-target,axis=0)/np.maximum(np.linalg.norm(target,axis=0),1e-12)
   bounded=[]
   lb=np.maximum(-.06,m.jnt_range[lower_jids,0]-d.qpos[lower_qadr]);ub=np.minimum(.06,m.jnt_range[lower_jids,1]-d.qpos[lower_qadr])
   if np.any(lb>=ub):
    bounded.append(dict(status='invalid_original_state_bounds',lower=lb.tolist(),upper=ub.tolist()))
   else:
    for axis in range(2):
     for sign in [-1,1]:
      wanted=target[:,axis]*np.deg2rad(1)*sign;sol=lsq_linear(jl,wanted,bounds=(lb,ub),tol=1e-12,max_iter=200);res=jl@sol.x-wanted
      assert np.all(sol.x>=lb-1e-10) and np.all(sol.x<=ub+1e-10)
      bounded.append(dict(axis=['roll_x','pitch_y'][axis],sign=sign,solver_success=bool(sol.success),solver_status=int(sol.status),relative_residual=float(np.linalg.norm(res)/max(np.linalg.norm(wanted),1e-12)),contact_displacement_residual_max_m=float(abs(res).max()),joint_increment_max_rad=float(abs(sol.x).max()),active_bound_joints=[lower[k] for k in range(14) if sol.active_mask[k]],solution_rad=sol.x.tolist()))
   samples.append(dict(bounded=bounded,step=i+1,contacts=len(mat),lower_joint_rank=int(rank),root_body_frame_axes=['roll_x','pitch_y'],relative_constraint_residual=residual.tolist(),joint_correction_norm_per_rad=np.linalg.norm(solution,axis=0).tolist(),singular_values=sv.tolist()))
 reports.append(dict(id=row['id'],frame=row['frame'],split=row['split'],trace_sha256=row['sha256'],samples=samples))
out=dict(scope='Position constraints at archived final-solver loaded foot points (>0.01N); poststep state timing caveat. Root free-joint rotational columns are body-frame angular directions. Pseudoinverse rank uses rcond1e-8 for descriptive algebra, not an acceptance change. No force, momentum, actuation, friction or dynamic response guarantee.',asset_sha256=hashlib.sha256(a.asset.read_bytes()).hexdigest(),lower_joint_names=lower,fd_checks=fdchecks,fd_max_error=fdmax,fd_passed=True,rows=reports);a.output.write_text(json.dumps(out,indent=2)+'\n');ss=[s for r in reports for s in r['samples'] if s['contacts']];print('fd_checks',fdchecks,'fd_max_error',fdmax,'loaded_samples',len(ss));print('median_residual',np.median([s['relative_constraint_residual'] for s in ss],axis=0),'p95_joint_norm_per_rad',np.quantile([s['joint_correction_norm_per_rad'] for s in ss],.95,axis=0))
