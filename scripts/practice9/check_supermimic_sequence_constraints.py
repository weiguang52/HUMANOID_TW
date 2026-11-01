"""Full-frame sparse joint/velocity constraint foundation, not repaired references."""
import argparse,hashlib,json
from pathlib import Path
import mujoco,numpy as np,scipy.sparse as sp,osqp
p=argparse.ArgumentParser(description=__doc__)
for k in ['plan','asset','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);plan=json.loads(a.plan.read_text());m=mujoco.MjModel.from_xml_path(str(a.asset));rows=[]
for item in plan['motions']:
 f=Path(item['file']);assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256']
 with np.load(f) as z:
  names=z['joint_names'].tolist();lower=[i for i,n in enumerate(names) if n.startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot'))];assert len(lower)==14;ids=[m.joint(names[i]).id for i in lower];q=z['joint_pos'][:,lower].astype(float);n=len(q);dt=1/float(z['fps'].item());assert dt==.02
  lo=np.column_stack([np.broadcast_to([-.06,-.06,-.031],(n,3)),np.maximum(-.8,m.jnt_range[ids,0]-q)]).ravel();hi=np.column_stack([np.broadcast_to([.06,.06,.031],(n,3)),np.minimum(.8,m.jnt_range[ids,1]-q)]).ravel();D=sp.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n),format='csc');selection=sp.hstack([sp.csc_matrix((14,3)),sp.eye(14,format='csc')],format='csc');velocity=sp.kron(D,selection,format='csc');original_delta=np.diff(q,axis=0).ravel();A=sp.vstack([sp.eye(n*17,format='csc'),velocity],format='csc');l=np.r_[lo,-3*dt-original_delta];u=np.r_[hi,3*dt-original_delta];solver=osqp.OSQP();solver.setup(P=sp.eye(n*17,format='csc'),q=np.zeros(n*17),A=A,l=l,u=u,verbose=False,eps_abs=1e-8,eps_rel=1e-8,polishing=True,max_iter=20000);result=solver.solve();assert result.info.status_val==1,result.info.status;x=result.x;assert np.isfinite(x).all();violation=float(max(0,(l-A@x).max(),(A@x-u).max()));assert violation<=1e-7
  corrected=q+x.reshape(n,17)[:,3:];speed_ratio=float((abs(np.diff(corrected,axis=0))/(3*dt)).max());assert speed_ratio<=1+1e-5/3
  np.savez_compressed(a.output/(item['id']+'_constraint_seed.npz'),delta=x.reshape(n,17));rows.append(dict(id=item['id'],split=item['split'],source_sha256=item['sha256'],frames=n,variables=n*17,constraints=A.shape[0],solver_status=result.info.status,max_constraint_violation=violation,max_seed_joint_delta_rad=float(abs(x.reshape(n,17)[:,3:]).max()),speed_ratio=speed_ratio))
report=dict(scope='Sparse full-sequence original joint/rate constraints only; objective keeps zero correction. Not smoothing a candidate, not contact/COM solve, no training references or physical acceptance.',osqp_version=osqp.__version__,asset_sha256=hashlib.sha256(a.asset.read_bytes()).hexdigest(),rows=rows);(a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(rows)
