"""Matched early real-loop COM/momentum/support observations; no causal gate."""
import argparse,hashlib,json
from pathlib import Path
import mujoco,numpy as np
p=argparse.ArgumentParser(description=__doc__)
for k in ['root','asset','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();plan=json.loads((a.root/'plan.json').read_text());refs={r['id']:r for r in plan['motions']};m=mujoco.MjModel.from_xml_path(str(a.asset));d=mujoco.MjData(m);names=json.loads((a.root/'bounded_reference/input.json').read_text())['state_joint_names'];qa=[int(m.joint(n).qposadr[0]) for n in names];va=[int(m.joint(n).dofadr[0]) for n in names];base=m.body('base_link').id;mass=m.body_mass;total=mass.sum();out=[];max_h_error=0.;max_v_error=0.
def reference_com(ref,i):
 d.qpos[:3]=ref['root_pos'][i];d.qpos[3:7]=ref['root_quat_wxyz'][i];d.qpos[qa]=ref['joint_pos'][i];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);return d.subtree_com[base].copy()
def real_momentum(state):
 global max_h_error,max_v_error
 d.qpos[:3]=state[:3];d.qpos[3:7]=state[3:7];d.qpos[qa]=state[13:43];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);R=d.xmat[base].reshape(3,3);d.qvel[:3]=state[7:10];d.qvel[3:6]=R.T@state[10:13];d.qvel[va]=state[43:73];mujoco.mj_comVel(m,d);mujoco.mj_subtreeVel(m,d);com=d.subtree_com[base].copy();vel=[];omega=[]
 for b in range(m.nbody):
  jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacBodyCom(m,d,jp,jr,b);vel.append(jp@d.qvel);omega.append(jr@d.qvel)
 vel=np.array(vel);omega=np.array(omega);v=(mass[:,None]*vel).sum(0)/total;rot=d.ximat.reshape(-1,3,3);I=np.einsum('bij,bj,bkj->bik',rot,m.body_inertia,rot);H=(np.einsum('bij,bj->bi',I,omega)+np.cross(d.xipos-com,mass[:,None]*(vel-v))).sum(0);he=float(abs(H-d.subtree_angmom[base]).max());ve=float(abs(v-d.subtree_linvel[base]).max());max_h_error=max(max_h_error,he);max_v_error=max(max_v_error,ve);assert he<1e-8 and ve<1e-8
 return com,v,H
baseline=json.loads((a.root/'bounded_reference/results.json').read_text())['rows'];candidate=json.loads((a.root/'contact_balance/results.json').read_text())['rows']
for x,y in zip(baseline,candidate):
 assert (x['id'],x['frame'])==(y['id'],y['frame']);n=min(10,x['steps'],y['steps'])
 with np.load(refs[x['id']]['file']) as ref:
  desired=[reference_com(ref,x['frame']+i+1) for i in range(n)]
 for arm,row in [('bounded_reference',x),('contact_balance',y)]:
  logs=json.loads((a.root/arm/row['control_file']).read_text());samples=[]
  with np.load(a.root/arm/row['trace']) as z:
   for i in range(n):
    com,v,H=real_momentum(z['next_state'][i]);feet=logs[i]['post_feet'];points=[c for f in feet.values() for c in f['contacts'] if c['ground'] and c['normal_force_n']>.01];normal=sum(c['normal_force_n'] for c in points);cop=(sum((np.array(c['pos_w'][:2])*c['normal_force_n'] for c in points),np.zeros(2))/normal).tolist() if normal else None
    samples.append(dict(step=i+1,com_w_m=com.tolist(),com_error_xy_m=float(np.linalg.norm(com[:2]-desired[i][:2])),com_velocity_w_m_s=v.tolist(),angular_momentum_w_kg_m2_s=H.tolist(),angular_momentum_norm=float(np.linalg.norm(H)),foot_normal_force_n=normal,loaded_ground_points=len(points),normal_weighted_contact_xy_m=cop))
  out.append(dict(id=row['id'],frame=row['frame'],split=row['split'],arm=arm,samples=samples))
report=dict(scope='All32 matched first10 real steps only. Exact kinematic velocities from archived state; mass-weighted body Jacobian momentum independently compared with MuJoCo subtree values. Final-solver foot normal loads are not whole-step impulse or all-body contact force; no friction/momentum causality or stability acceptance.',momentum_crosscheck_max_error=max_h_error,com_velocity_crosscheck_max_error=max_v_error,total_mass_kg=float(total),rows=out);a.output.write_text(json.dumps(report,indent=2)+'\n');print('momentum_error',max_h_error,'velocity_error',max_v_error)
for split in ['fit','validation']:
 for arm in ['bounded_reference','contact_balance']:
  ss=[s for r in out if r['split']==split and r['arm']==arm for s in r['samples']];print(split,arm,'mean_com_xy_mm',1000*np.mean([s['com_error_xy_m'] for s in ss]),'mean_H',np.mean([s['angular_momentum_norm'] for s in ss]),'mean_Fz',np.mean([s['foot_normal_force_n'] for s in ss]))
