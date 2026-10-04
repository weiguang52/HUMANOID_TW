"""Fail closed on mixed data/category weighting and sidecar integrity."""
import json, hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
p=Path('/root/gpufree-data/datasets/practice9/tw123_v1/contact/splits/train.json')
d=json.loads(p.read_text());rows=d['motions'];counts=Counter(r['category'] for r in rows)
assert counts['walking']>=2000 and counts['standing_upper']>=1000,counts
assert d['joint_coordinate_contract']=='tw123_yaw_waist_v3'
mass=defaultdict(float)
for r in rows:
    mass[r['category']]+=r['weight']*(r['frames']-1)
assert abs(mass['walking']-mass['standing_upper'])<1.e-6,mass
spec=d['contact_reference'];cp=Path(spec['file']);sha=hashlib.sha256(cp.read_bytes()).hexdigest()
assert sha==spec['sha256']
with np.load(cp) as a:
    assert a['clip_ids'].tolist()==[r['id'] for r in rows]
    assert a['lengths'].tolist()==[r['frames'] for r in rows]
print(json.dumps(dict(counts=counts,sampling_mass=mass,frames=sum(r['frames'] for r in rows),
    manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),contact_sha256=sha),indent=2))
