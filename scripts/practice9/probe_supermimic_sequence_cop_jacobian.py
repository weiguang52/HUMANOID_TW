"""First ascending fit source only: inverse-COP temporal derivative implementation check."""
import argparse,datetime,hashlib,json
from pathlib import Path
import mujoco,numpy as np
from supermimic_sequence_inverse_cop import SequenceInverseCOP
p=argparse.ArgumentParser(description=__doc__)
for k in ['plan','asset','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir();plan=json.loads(a.plan.read_text());source=Path(plan['source_plan']);h=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();assert h(source)==plan['source_plan_sha256'];item=next(v for v in json.loads(source.read_text())['motions'] if v['split']=='fit');assert item['id']=='000041' and h(item['file'])==item['sha256'];m=mujoco.MjModel.from_xml_path(str(a.asset))
with np.load(item['file']) as z:
 names=z['joint_names'].tolist();qa=[int(m.joint(n).qposadr[0]) for n in names];va=[int(m.joint(n).dofadr[0]) for n in names];lower=[qa[i] for i,n in enumerate(names) if n.startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot'))];q=np.zeros((len(z['joint_pos']),m.nq));q[:,:3]=z['root_pos'];q[:,3:7]=z['root_quat_wxyz'];q[:,qa]=z['joint_pos'];engine=SequenceInverseCOP(a.asset,q,lower,qa,va,1/float(z['fps'].item()))
x=np.zeros((len(q),17));cop,force=engine.evaluate(x);J=engine.jacobian(x);np.savez_compressed(a.output/'base.npz',cop=cop,force=force);from scipy.sparse import save_npz;save_npz(a.output/'jacobian.npz',J)
rng=np.random.default_rng(42);checks=[]
for index in range(3):
 direction=rng.normal(size=x.shape)*np.r_[np.repeat(.0001,3),np.repeat(.001,14)];eps=1e-3;numeric=(engine.evaluate(x+eps*direction)[0]-engine.evaluate(x-eps*direction)[0])/(2*eps);predicted=(J@direction.ravel()).reshape(-1,2);error=float(abs(numeric-predicted).max());checks.append(dict(index=index,max_error_m=error,max_directional_response_m=float(abs(numeric).max()),passed=error<=1e-6));print(checks[-1],flush=True)
# Isolated boundary and interior columns validate colouring's exact temporal support.
columns=[]
for frame in [0,1,len(q)//2,len(q)-2,len(q)-1]:
 dx=np.zeros_like(x);dx[frame,0]=1e-6;numeric=((engine.evaluate(x+dx)[0]-engine.evaluate(x-dx)[0])/2e-6).ravel();error=float(abs(numeric-J[:,frame*17].toarray().ravel()).max());columns.append(dict(frame=frame,error=error,passed=error<=1e-6))
out=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=item,asset_sha256=h(a.asset),scope='Numerical derivative implementation only. No optimisation, training export or physical/model acceptance.',checks=checks,isolated_columns=columns,passed=all(v['passed'] for v in checks+columns));(a.output/'results.json').write_text(json.dumps(out,indent=2)+'\n');print('PASSED',out['passed'])
