"""Read-only necessary reference speed-cap compatibility, not feasibility proof."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from supermimic.config.tw.velocity_limits import LIMITS,PHYSICS_CONTRACT
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir();r=a.root
manifests=[('training',r/'training_speed_blend_20261026/train.json',332),('fixed7',r/'independent_holdout_20261025/test.json',7)]
rows=[];evidence=[];sets={}
for split,path,count in manifests:
 d=json.loads(path.read_text());assert len(d['motions'])==count
 evidence.append(dict(split=split,file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()));sets[split]={x['id'] for x in d['motions']}
 for row in d['motions']:
  path=Path(row['file']);sha=hashlib.sha256(path.read_bytes()).hexdigest()
  with np.load(path,allow_pickle=False) as z:names=z['joint_names'].tolist();q=z['joint_pos'].astype(np.float64);v=z['joint_vel'].astype(np.float64);fps=float(z['fps'].item())
  assert len(names)==len(set(names))==30 and set(names)==set(LIMITS)
  assert q.shape==v.shape and q.shape[1]==30 and len(q)>=2 and np.isfinite(q).all() and np.isfinite(v).all() and fps==50
  caps=np.array([LIMITS[x] for x in names]);interval=np.diff(q,axis=0)*fps;central=np.gradient(q,1/fps,axis=0)
  stored_bad=np.abs(v)>caps+1e-5;interval_bad=np.abs(interval)>caps+1e-5
  details=[]
  for j,name in enumerate(names):
   details.append(dict(joint=name,cap_rad_s=float(caps[j]),stored_max_ratio=float(np.abs(v[:,j]).max()/caps[j]),interval_max_ratio=float(np.abs(interval[:,j]).max()/caps[j]),stored_bad_frames=np.flatnonzero(stored_bad[:,j]).tolist(),interval_bad_starts=np.flatnonzero(interval_bad[:,j]).tolist(),central_vs_stored_rmse=float(np.sqrt(np.mean((central[:,j]-v[:,j])**2)))))
  rows.append(dict(id=row['id'],split=split,file=str(path),sha256=sha,frames=len(q),fps=fps,stored_violation_frames=int(stored_bad.any(1).sum()),interval_violation_intervals=int(interval_bad.any(1).sum()),stored_max_ratio=float((np.abs(v)/caps).max()),interval_max_ratio=float((np.abs(interval)/caps).max()),joints=details))
assert not sets['training']&sets['fixed7'] and len(rows)==339
summary=[]
for split,_,_ in manifests:
 group=[x for x in rows if x['split']==split]
 summary.append(dict(split=split,motions=len(group),motions_with_stored_violations=sum(x['stored_violation_frames']>0 for x in group),motions_with_interval_violations=sum(x['interval_violation_intervals']>0 for x in group),stored_max_ratio=max(x['stored_max_ratio'] for x in group),interval_max_ratio=max(x['interval_max_ratio'] for x in group)))
out=dict(coverage_and_finite_passed=True,physics=PHYSICS_CONTRACT,limits=LIMITS,manifests=evidence,rows=rows,summary=summary,scope='All original332 training and fixed7 references. Adjacent angle displacement/cap is a necessary exact-tracking condition; passing cannot establish contact/dynamic feasibility. No retiming, sample selection, gate/cap change or optimizer.')
(a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary))
