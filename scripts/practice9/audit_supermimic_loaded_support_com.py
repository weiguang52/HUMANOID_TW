"""Post-step COM versus archived loaded-contact hull; static diagnostic only."""
import argparse,hashlib,json
from pathlib import Path
import mujoco,numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir();r=a.root;contact=r/'contact_probe_v2'
def hull(points):
 points=sorted(set(map(tuple,np.asarray(points).reshape(-1,2))))
 if len(points)<3:return np.asarray(points).reshape(-1,2)
 def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
 lower=[];upper=[]
 for point in points:
  while len(lower)>=2 and cross(lower[-2],lower[-1],point)<=0:lower.pop()
  lower.append(point)
 for point in reversed(points):
  while len(upper)>=2 and cross(upper[-2],upper[-1],point)<=0:upper.pop()
  upper.append(point)
 return np.array(lower[:-1]+upper[:-1])
def polygon(points,point):
 h=hull(points)
 if len(h)<3:return h,0.,None
 edges=np.roll(h,-1,axis=0)-h;area=float(abs(np.sum(h[:,0]*np.roll(h[:,1],-1)-h[:,1]*np.roll(h[:,0],-1)))/2)
 if area<=1e-14:return h,area,None
 diff=np.asarray(point)-h;margin=float(np.min((edges[:,0]*diff[:,1]-edges[:,1]*diff[:,0])/np.linalg.norm(edges,axis=1)))
 return h,area,margin
for points,query,expected in [([[-1,-1],[1,-1],[1,1],[-1,1]],[0,0],1.),([[-1,-1],[1,-1],[1,1],[-1,1]],[2,0],-1.),([[0,0],[2,0],[0,2]],[.5,.5],.5)]:
 _,_,value=polygon(points,query);assert abs(value-expected)<1e-12
assert polygon([[0,0],[1,0],[2,0]],[0,0])[2] is None
asset=r.parent/'robot/tw30.xml';sha=hashlib.sha256(asset.read_bytes()).hexdigest();assert sha=='c42fac383f507a88c81787224e06aede10301fff18596efc12a65af584e22d5a'
m=mujoco.MjModel.from_xml_path(str(asset));d=mujoco.MjData(m);mapping=json.loads((contact/'control_mapping.json').read_text());names=mapping['state_joint_names'];assert len(names)==30
addresses=[int(m.joint(n).qposadr[0]) for n in names];base=m.body('base_link').id
integrity=json.loads((contact/'integrity.json').read_text());assert integrity['finite'] and integrity['contiguous'] and integrity['samples']==780 and integrity['episodes']==16
cache={};sources=[]
for row in json.loads((contact/'results.json').read_text())['rows']:
 path=contact/row['trace']
 with np.load(path,allow_pickle=False) as z:state=z['state'].copy();nxt=z['next_state'].copy()
 assert np.array_equal(nxt[:-1],state[1:]) and np.isfinite(nxt).all();cache[row['trace']]=(nxt,row)
 sources.append(dict(file=row['trace'],sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
rows=[];samples=json.loads((contact/'contact_samples.json').read_text());assert len(samples)==780
for sample in samples:
 filename='seed%d_amp%.2f.npz'%(sample['seed'],sample['amplitude']);states,metadata=cache[filename];i=sample['step'];state=states[i]
 d.qpos[:3]=state[:3];d.qpos[3:7]=state[3:7];d.qpos[addresses]=state[13:43]
 mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);com=d.subtree_com[base].copy();assert np.isfinite(com).all()
 loaded=[x for x in sample['contacts'] if 'terrain' in x['geoms'] and any('_foot_collision' in g for g in x['geoms']) and x['normal_force_n']>.01]
 points=np.array([x['pos'][:2] for x in loaded]).reshape(-1,2);h,area,margin=polygon(points,com[:2]);normal=sum(x['normal_force_n'] for x in loaded)
 cop=(sum((x['normal_force_n']*np.array(x['pos'][:2]) for x in loaded),np.zeros(2))/normal).tolist() if normal>0 else None
 rows.append(dict(seed=sample['seed'],amplitude=sample['amplitude'],step=i,time_s=(i+1)*.02,trace=filename,split=metadata['split'],original_duration_s=metadata['steps']*.02,original_termination_reasons=metadata['termination_reasons'],com_world_m=com.tolist(),loaded_counts={side:sum(any(side+'_foot_collision' in g for g in x['geoms']) for x in loaded) for side in ['left','right']},loaded_hull_xy_m=h.tolist(),loaded_area_m2=area,com_signed_margin_m=margin,normal_force_n=normal,normal_weighted_contact_xy_m=cop))
summary=[]
for amp in [0,.02,.05,.1]:
 group=[x for x in rows if x['amplitude']==amp];steady=[x for x in group if x['time_s']>=2]
 for label,part in [('all',group),('after2s',steady)]:
  margins=[x['com_signed_margin_m'] for x in part if x['com_signed_margin_m'] is not None]
  summary.append(dict(amplitude=amp,range=label,samples=len(part),no_2d_polygon=sum(x['com_signed_margin_m'] is None for x in part),outside_polygon=sum(x<0 for x in margins),margin_min_m=min(margins) if margins else None,margin_median_m=float(np.median(margins)) if margins else None))
out=dict(coverage_passed=True,algorithm_self_tests_passed=True,samples=780,asset_sha256=sha,total_model_mass_kg=float(m.body_mass.sum()),force_threshold_n=.01,contact_sha256=hashlib.sha256((contact/'contact_samples.json').read_bytes()).hexdigest(),sources=sources,summary=summary,rows=rows,scope='Same env.step post-state COM and archived last-solver contact points/forces; possible substep timing offset. Foot-only static projection, other body contacts unavailable; not momentum/friction-cone/causal fall proof. All samples retained, no optimizer or physics modifications.')
(a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary))
