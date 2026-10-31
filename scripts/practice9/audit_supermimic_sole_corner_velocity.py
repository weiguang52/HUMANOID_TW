"""Verify every archived branch corner velocity against native MuJoCo."""
import argparse,hashlib,itertools,json
from pathlib import Path
import mujoco,numpy as np,torch
from supermimic_sole_corner_velocity import SoleCornerVelocity
p=argparse.ArgumentParser(description=__doc__)
for k in ['collection','asset','output']:p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--trace-fit-input',type=Path)
a=p.parse_args();assert not a.output.exists()
if a.trace_fit_input is None:
 integ=json.loads((a.collection/'integrity.json').read_text());assert integ['passed'];hashes={x['file']:x['sha256'] for x in integ['files']}
 source_rows=json.loads((a.collection/'results.json').read_text())['rows']
else:
 source_rows=json.loads(a.trace_fit_input.read_text())['trajectory_files']
 hashes={x['file']:x['sha256'] for x in source_rows}
mapping=json.loads((a.collection/'control_mapping.json').read_text());names=mapping['state_joint_names']
f=SoleCornerVelocity(a.asset,names).cuda();m=mujoco.MjModel.from_xml_path(str(a.asset));d=mujoco.MjData(m)
qads=[int(m.joint(n).qposadr[0]) for n in names];vads=[int(m.joint(n).dofadr[0]) for n in names]
feet=[m.geom(s+'_foot_collision').id for s in ['left','right']];signs=np.array(list(itertools.product([-1,1],repeat=3)))
rows=[];maximum=0.;total=0
for row in source_rows:
 path=(a.collection/row['file']).resolve();assert path.is_relative_to(a.collection.resolve())
 assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[row['file']]
 with np.load(path,allow_pickle=False) as z:states=z['state'].copy()
 native=[]
 for s in states:
  d.qpos[:7]=s[:7];d.qpos[qads]=s[13:43]
  rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,s[3:7].astype(float))
  d.qvel[:3]=s[7:10];d.qvel[3:6]=rotation.reshape(3,3).T@s[10:13];d.qvel[vads]=s[43:73]
  mujoco.mj_fwdPosition(m,d);mujoco.mj_fwdVelocity(m,d)
  velocities=[]
  for g in feet:
   six=np.empty(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_GEOM,g,six,0)
   offsets=(signs*m.geom_size[g])@d.geom_xmat[g].reshape(3,3).T
   velocities.append((six[3:]+np.cross(six[:3],offsets))[:,2])
  native.append(velocities)
 state=torch.tensor(states,device='cuda:0',requires_grad=True)
 predicted=f(state);error=float((predicted-torch.tensor(np.array(native),device='cuda:0')).abs().max())
 maximum=max(maximum,error);predicted.sum().backward();assert torch.isfinite(state.grad).all()
 vertical_error=float((state.grad[:,9]-16).abs().max());assert vertical_error<=1e-5
 rows.append(dict(file=row['file'],sha256=hashes[row['file']],states=len(states),max_native_error_m_s=error,root_vertical_velocity_sum_gradient_error=vertical_error));total+=len(states)
out=dict(passed=maximum<=1e-5,states=total,max_native_error_m_s=maximum,atol_m_s=1e-5,asset_sha256=hashlib.sha256(a.asset.read_bytes()).hexdigest(),rows=rows,
 scope='All selected archived input states; native kinematics/velocity only, no contact force, dynamics step, model training or policy acceptance.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}));assert out['passed']
