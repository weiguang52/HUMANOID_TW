import pathlib,json,hashlib,numpy as np
S=pathlib.Path('/root/gpufree-data/datasets/practice9/tw195_v1');a=json.load(open(S/'train_original.json'));b=json.load(open(S/'train_corrected.json'));assert a['contact_reference']==b['contact_reference'];assert [(r['id'],r['frames'],r['weight']) for r in a['motions']]==[(r['id'],r['frames'],r['weight']) for r in b['motions']]
c=np.load(b['contact_reference']['file']);old=np.load(S.parent/'tw154_v1/train.contacts.npz');ids=old['clip_ids'].tolist();oo=np.r_[0,np.cumsum(old['lengths'])];off=0;hashes={}
held=json.load(open(S/'validation/corrected/manifest.json'))['motions'];assert not ({r['source_family'] for r in a['motions']}&{r['source_family'] for r in held})
for x,y in zip(a['motions'],b['motions']):
 n=x['frames'];i=ids.index(x['id'])
 for k in ['contact','known','source_contact4']:assert np.array_equal(c[k][off:off+n],old[k][oo[i]:oo[i+1]])
 off+=n;p=pathlib.Path(y['file']);z=np.load(p);o=np.load(x['file']);assert z['joint_pos'].shape==o['joint_pos'].shape and np.array_equal(z['joint_pos'][:,14:],o['joint_pos'][:,14:]);assert np.array_equal(z['root_pos'][:,:2],o['root_pos'][:,:2]);assert np.array_equal(z['root_quat_wxyz'],o['root_quat_wxyz']);assert float(z['fps'][0])==50;hashes[x['id']]=hashlib.sha256(p.read_bytes()).hexdigest()
 for k in z.files:
  if np.issubdtype(z[k].dtype,np.number):assert np.isfinite(z[k]).all()
assert off==len(c['contact'])
summary=dict(clips=len(hashes),frames=off,contact_bits_known_and_four_markers_exact=True,upper_joints_root_xy_orientation_exact=True,family_disjoint=True,category_mass={cat:sum(r['weight']*(r['frames']-1) for r in b['motions'] if r['category']==cat) for cat in ['walking','standing_upper']})
O=pathlib.Path('validation_artifacts/tw195_setup');(O/'paired_data_audit.json').write_text(json.dumps(summary,indent=2));(O/'corrected_data_sha256.json').write_text(json.dumps(hashes,indent=2));print(summary)
