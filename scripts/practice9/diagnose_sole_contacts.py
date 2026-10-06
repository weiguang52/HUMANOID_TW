"""Collision-corner diagnostics; velocity proxy is not a measured contact patch."""
from pathlib import Path
import json,sys,importlib.util
import numpy as np,torch
spec=importlib.util.spec_from_file_location('sole','source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/sole_tracking.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
URDF='/root/gpufree-data/datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf'

def event_lags(reference, actual, dt=.02, tolerance=.2):
    # Greedy nearest unique pairing; unmatched events remain explicit.
    pairs=sorted((abs(a-r),i,j,a-r) for i,r in enumerate(reference)
                 for j,a in enumerate(actual) if abs(a-r)*dt<=tolerance)
    used_r=set();used_a=set();lags=[]
    for _,i,j,lag in pairs:
        if i not in used_r and j not in used_a:
            used_r.add(i);used_a.add(j);lags.append(float(lag*dt))
    return dict(lags_s=lags,unmatched_reference=len(reference)-len(used_r),
                unmatched_actual=len(actual)-len(used_a),tolerance_s=tolerance)

def diagnose(p,motion,offset):
 a=np.load(p);end=int(np.flatnonzero(a['terminal'])[0]) if a['terminal'].any() else len(a['terminal'])
 sel=(a['control_step']>=50)&(a['control_step']<end);steps=a['control_step'][sel]
 if not len(steps):return {'insufficient_samples':True}
 vertices,com=g.read_foot_geometry(URDF);corners=torch.tensor(vertices).float()
 pos=torch.from_numpy(a['foot_position'][sel]).float();q=torch.from_numpy(a['foot_quaternion'][sel]).float()
 w=g.world_corners(pos,q,corners);z=w[...,2];lowest=z.min(-1).values
 # IsaacLab body_lin_vel_w aliases COM velocity; transform to material vertices.
 com_w=pos+g.rotate(q,torch.tensor(com).float().expand_as(pos))
 velocity=torch.from_numpy(a['foot_velocity'][sel]).float().unsqueeze(-2)
 omega=torch.from_numpy(a['foot_angular_velocity'][sel]).float().unsqueeze(-2)
 cv=velocity+torch.cross(omega.expand_as(w),w-com_w.unsqueeze(-2),dim=-1)
 speed=cv[...,:2].norm(dim=-1)
 near=z<=lowest.unsqueeze(-1)+.001
 low_speed=speed.masked_fill(~near,float('inf')).min(-1).values.numpy()
 force=a['force'][sel,:,2];contact=force>1.;labels=a['reference_contact'][steps].astype(bool);known=a['reference_known'][steps].astype(bool)
 swing=known&~labels;stance=known&labels
 low=lowest.numpy()
 ref=np.load(motion['file']);bn=ref['body_names'].tolist();ix=[bn.index(n) for n in ['left_foot','right_foot']]
 fi=a['reference_frame'][steps].astype(int)-offset
 rz=g.sole_height(torch.from_numpy(ref['body_pos_w'][fi][:,ix]).float(),torch.from_numpy(ref['body_quat_w'][fi][:,ix]).float(),corners).numpy()
 def mean(x):return float(np.mean(x)) if x.size else None
 # Raw physics contact changes: diagnostics, not filtered reward events.
 transitions=np.diff(contact.astype(int),axis=0)
 ends=np.r_[np.flatnonzero(np.diff(a['control_step'])),len(a['control_step'])-1]
 measured=a['force'][ends][:end,:,2]>1
 reference=a['reference_contact'][:end].astype(bool)
 events=[]
 for foot in range(2):
  re=np.flatnonzero(np.diff(reference[:,foot].astype(int))==1)+1
  ac=np.flatnonzero(np.diff(measured[:,foot].astype(int))==1)+1
  events.append(event_lags(re[re>=50],ac[ac>=50]))
 return dict(samples=len(steps),touchdown_matching=events,sole_reference_rmse_m=float(np.sqrt(np.mean((low-rz)**2))),
  known_swing_sole_reference_error_m=mean(np.abs(low-rz)[swing]),sole_z_m_p05=np.quantile(low,.05,axis=0).tolist(),
  sole_z_m_p95=np.quantile(low,.95,axis=0).tolist(),
  known_swing_sole_height_m=mean(low[swing]),known_swing_contact=mean(contact[swing]),
  known_stance_missing_contact=mean((~contact)[stance]),
  contact_lowest_corner_speed_candidate_min_m_s=mean(low_speed[contact]),
  known_swing_contact_force_n=mean(force[swing]),
  touchdown_events_per_second=((transitions==1).sum(0)/(len(steps)*float(a['physics_dt']))).tolist())

if __name__=='__main__':
 state=Path(sys.argv[1]);variants=sys.argv[2].split(',');target=Path(sys.argv[3]);rows=[]
 m=json.loads((state/'validation/manifest.json').read_text())
 cats={r['id']:r.get('evaluation_class',r['category']) for r in m['motions']}
 for v in variants:
  for p in sorted((state/'validation'/v).glob('*.seed*.npz')):
   motion=next(r for r in m['motions'] if r['id']==p.name.split('.')[0]);offset=sum(r['frames'] for r in m['motions'][:m['motions'].index(motion)])
   r=diagnose(p,motion,offset);r.update(variant=v,replay=p.stem,category=cats[p.name.split('.')[0]]);rows.append(r)
 target.write_text(json.dumps(dict(rows=rows,note='World collision-box minimum Z. Lowest-corner speed is a minimum candidate-corner geometry proxy, not actual contact point measurement; >1N raw contact, no phase edits, first episode excluding 1s.'),indent=2))
 print('diagnosed',len(rows))
