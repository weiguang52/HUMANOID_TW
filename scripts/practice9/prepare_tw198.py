"""Synthetic diagnostic targets; original motion/contact data never mutated."""
import pathlib,json,numpy as np,hashlib,sys
from contact_ik import rebuild,Model
import pinocchio as pin
from scipy.spatial import ConvexHull
S=pathlib.Path('/root/gpufree-data/datasets/practice9/tw198_v1');S.mkdir(exist_ok=True);rows=[];geometries=[];N=1101
for variant in ['original','corrected']:
 m=json.load(open(S.parent/'tw195_v1/validation'/variant/'manifest.json'));r=next(r for r in m['motions'] if r['id']=='000016');src=dict(np.load(r['file']));start=100
 for mode in ['static','upper']:
  a={k:(np.repeat(v[start:start+1],N,axis=0) if v.ndim and v.shape[0]==len(src['joint_pos']) else v.copy()) for k,v in src.items()}
  if mode=='upper':
   ix=[i for i,n in enumerate(src['joint_names']) if any(t in n for t in ['shoulder','elbow','wrist'])];t=np.clip(start+np.maximum(np.arange(N)-50,0),0,len(src['joint_pos'])-1);a['joint_pos'][:,ix]=src['joint_pos'][t][:,ix]
  out,points=rebuild(a,a['joint_pos'],a['root_pos']);mid=variant+'_'+mode;p=S/(mid+'.npz');np.savez_compressed(p,**out);rows.append(dict(r,id=mid,file=str(p),frames=N,weight=1/(N-1),source_motion='000016',diagnostic='synthetic20s,first1sstatic,uppergesturethenhold; fixed pelvis/legs; no claim of original contact ground truth'))
  model=Model(out['joint_names'],out['body_names']);q=model.configuration(out,0);com=pin.centerOfMass(model.m,model.d,q);corners=points[0].reshape(-1,3)[:,:2];hull=ConvexHull(corners);margin=-np.max(hull.equations[:,:2]@com[:2]+hull.equations[:,2]);geometries.append(dict(id=mid,total_mass=float(sum(x.mass for x in model.m.inertias)),initial_com_xy=com[:2].tolist(),optimistic_support_margin_m=float(margin),note='Projected full-foot hull, not measured active-contact polygon.'))
base=dict(m,motions=rows);c=S/'contacts.npz';np.savez_compressed(c,clip_ids=np.array([r['id'] for r in rows]),lengths=np.array([N]*4),foot_names=np.array(['left_foot','right_foot']),contact=np.ones((N*4,2),dtype=np.uint8),known=np.ones((N*4,2),dtype=np.uint8),source_contact4=np.ones((N*4,4),dtype=np.uint8));base['contact_reference']=dict(file=str(c),sha256=hashlib.sha256(c.read_bytes()).hexdigest(),version='synthetic_double_support_diagnostic_not_human_labels');(S/'manifest.json').write_text(json.dumps(base,indent=2));(S/'geometry.json').write_text(json.dumps(geometries,indent=2))
for model in ['original','corrected']:
 for dt in ['200hz','400hz']:
  dest=S/model/dt;dest.mkdir(parents=True,exist_ok=True);jobs=[]
  for i,r in enumerate(rows):
   for controller in ['reference_pd','policy']:jobs.append(dict(motion_id=i,steps=1000,controller=controller,output=str(dest/(r['id']+'_'+controller+'.json'))))
  (dest/'jobs.json').write_text(json.dumps(jobs,indent=2))
print(geometries)
