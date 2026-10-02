"""Derived diagnostic: per-frame lowest foot sole at1mm; no training approval."""
import numpy as np,json,xml.etree.ElementTree as ET,itertools
from scipy.spatial.transform import Rotation
from pathlib import Path
p=Path('/root/gpufree-data/datasets/practice9/s2_squat_test');out=p/'grounded';out.mkdir(exist_ok=True)
u=ET.parse('/root/gpufree-data/datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf').getroot()
for clip in ['000890','001240']:
 manifest=json.load(open(p/(clip+'.manifest.json')));m=manifest['motions'][0];d=dict(np.load(m['file']));soles=[]
 for foot in ['left_foot','right_foot']:
  c=u.find(f"link[@name='{foot}']/collision");size=np.fromstring(c.find('geometry/box').get('size'),sep=' ');xyz=np.fromstring(c.find('origin').get('xyz'),sep=' ');box=np.array(list(itertools.product((-1,1),repeat=3)))*size/2+xyz;j=d['body_names'].tolist().index(foot);q=d['body_quat_w'][:,j];rot=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix();points=np.einsum('nij,kj->nki',rot,box)+d['body_pos_w'][:,j,None,:];soles.append(points[:,:,2].min(1))
 shift=.001-np.minimum(*soles);d['root_pos'][:,2]+=shift;d['body_pos_w'][:,:,2]+=shift[:,None];d['body_lin_vel_w']=np.gradient(d['body_pos_w'],1/float(d['fps'][0]),axis=0).astype(np.float32)
 np.savez(out/(clip+'.npz'),**d)
 m['source_file']=m['file'];m['file']=str(out/(clip+'.npz'));m['quality_pass']=False
 m['diagnostic_grounding']={'initial_original_sole_m':[float(s[0]) for s in soles],'method':'Per-frame minimum foot-box sole at1mm; recompute body linear velocity. Joint angles unchanged. Does not enforce zero foot slip or two-foot contact.','not_training_approved':True}
 manifest['diagnostic_only']=True;(out/(clip+'.manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
