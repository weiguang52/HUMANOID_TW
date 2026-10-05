from pathlib import Path
import hashlib,json,numpy as np
s=Path('/root/gpufree-data/datasets/practice9/tw154_v1');audit=json.load(open(s/'data_audit.json'))
assert hashlib.sha256((s/'train.json').read_bytes()).hexdigest()==audit['manifest_sha256']
m=json.load(open(s/'train.json'));spec=m['contact_reference'];p=Path(spec['file'])
assert hashlib.sha256(p.read_bytes()).hexdigest()==spec['sha256']==audit['contact_sha256']
a=np.load(p);assert a['clip_ids'].tolist()==[r['id'] for r in m['motions']]
assert a['lengths'].tolist()==[r['frames'] for r in m['motions']]
assert sum(a['lengths'])==a['contact'].shape[0]==a['known'].shape[0]
for c in ('walking','standing_upper'):
 assert abs(sum(r['weight']*(r['frames']-1) for r in m['motions'] if r['category']==c)-.5)<1e-9
assert m['joint_coordinate_contract']=='tw123_yaw_waist_v3'
assert len({r['id'] for r in m['motions']})==len(m['motions'])
print(json.dumps(audit,indent=2))
