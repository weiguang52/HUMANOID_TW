"""Read-only phase/geometry audit; preserves reference bits and timing."""
import json,pathlib,sys
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from diagnose_sole_contacts import g,URDF
S=pathlib.Path('/root/gpufree-data/datasets/practice9/tw176_v1')
O=pathlib.Path('validation_artifacts/tw186_contact_audit');O.mkdir(parents=True,exist_ok=True)
m=json.load(open(S/'validation/manifest.json'))['motions'];corners=torch.tensor(g.read_foot_geometry(URDF)[0]).float();rows=[]
for v in ['control','sole']:
 for p in sorted((S/'validation'/v).glob('*.seed*.npz')):
  a=np.load(p);mid=p.name[:6];r=next(x for x in m if x['id']==mid);offset=sum(x['frames'] for x in m[:m.index(r)])
  end=int(np.flatnonzero(a['terminal'])[0]) if a['terminal'].any() else len(a['terminal'])
  sel=(a['control_step']>=50)&(a['control_step']<end);steps=a['control_step'][sel]
  if not len(steps):continue
  ref=np.load(r['file']);bn=ref['body_names'].tolist();ix=[bn.index(n) for n in ['left_foot','right_foot']];fi=a['reference_frame'][steps].astype(int)-offset
  rz=g.sole_height(torch.from_numpy(ref['body_pos_w'][fi][:,ix]).float(),torch.from_numpy(ref['body_quat_w'][fi][:,ix]).float(),corners).numpy()
  z=g.sole_height(torch.from_numpy(a['foot_position'][sel]).float(),torch.from_numpy(a['foot_quaternion'][sel]).float(),corners).numpy()
  labels=a['reference_contact'][steps].astype(bool);known=a['reference_known'][steps].astype(bool);f=a['force'][sel,:,2];contact=f>1.;t=steps*.02
  def mean(x):return float(x.mean()) if x.size else None
  for j,side in enumerate(['left','right']):
   swing=known[:,j]&~labels[:,j];stance=known[:,j]&labels[:,j]
   rows.append(dict(variant=v,replay=p.stem,category=r.get('evaluation_class',r['category']),foot=side,known_swing_frames=int(swing.sum()),known_stance_frames=int(stance.sum()),ref_stance_above_5mm=mean((rz[:,j]> .005)[stance]),ref_stance_above_10mm=mean((rz[:,j]>.01)[stance]),ref_swing_below_5mm=mean((rz[:,j]<.005)[swing]),actual_swing_contact=mean(contact[:,j][swing]),actual_stance_missing=mean((~contact[:,j])[stance]),reference_stance_height=mean(rz[:,j][stance]),reference_swing_height=mean(rz[:,j][swing]),actual_swing_height=mean(z[:,j][swing])))
  if '.seed42' not in p.stem:continue
  fig,axs=plt.subplots(4,2,figsize=(15,9),sharex=True)
  for j,side in enumerate(['left','right']):
   axs[0,j].plot(t,rz[:,j]*100,label='reference');axs[0,j].plot(t,z[:,j]*100,label='actual');axs[0,j].set_title(v+' '+mid+' '+side);axs[0,j].set_ylabel('sole height cm');axs[0,j].legend()
   axs[1,j].plot(t,labels[:,j],label='source contact');axs[1,j].plot(t,known[:,j]-.03,label='known');axs[1,j].plot(t,contact[:,j]+.03,alpha=.6,label='Fz>1N');axs[1,j].legend(fontsize=7)
   axs[2,j].plot(t,f[:,j]);axs[2,j].set_ylabel('vertical force N')
   axs[3,j].plot(t,np.linalg.norm(a['foot_velocity'][sel,j,:2],axis=-1));axs[3,j].set_ylabel('foot COM speed m/s');axs[3,j].set_xlabel('simulation seconds')
  fig.tight_layout();fig.savefig(O/(v+'_'+mid+'.png'),dpi=100);plt.close(fig)
(O/'metrics.json').write_text(json.dumps(rows,indent=2));print('rows',len(rows),flush=True)
for v in ['control','sole']:
 rr=[r for r in rows if r['variant']==v and r['category']=='flat_forward']
 print(v,{k:mean(np.array([r[k] for r in rr if r[k] is not None])) for k in ['ref_stance_above_5mm','ref_stance_above_10mm','ref_swing_below_5mm','reference_stance_height','reference_swing_height','actual_swing_contact']},flush=True)
