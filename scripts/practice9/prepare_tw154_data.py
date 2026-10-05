import json,re,hashlib
from pathlib import Path
import numpy as np
from tw75_sampling import balance
root=Path('/root/gpufree-data/datasets/practice9');out=root/'tw154_v1';out.mkdir(exist_ok=True)
source=json.load(open(root/'tw123_v1/contact/splits/train.json'))
audit={r['id']:r for r in json.load(open(root/'tw151_v1/source_audit.json'))}
ban=re.compile(r'treadmill|stair|step.over|step_up|step_down|turn|circle|backward|sideways|side.step|jump|\brun\b|jog|limp|crouch|squat|kick|dance|balance|obstacle|slope|uphill|downhill|zig|arc\b|curve|stagger|stumble|push|support|table|handrail|beam|parkour|drank|drunk|tip.?toe|tennis|basketball|carry|lift|pick|throw|crutch|shelf',re.I)
accepted=[];decisions=[]
for r in source['motions']:
 if r['category']=='standing_upper':accepted.append(r);continue
 a=audit[r['id']];text=' '.join(a['captions'])+' '+a['source_family']
 reasons=[]
 if ban.search(text):reasons.append('special_motion_text_or_source')
 if a['low_foot_p95_p05_m']>=.05:reasons.append('nonflat_height_review')
 if a['root_height_span_m']>=.15:reasons.append('root_height_review')
 if a['straightness']<.9:reasons.append('not_straight')
 if not re.search(r'forward|walking_(slow|medium|fast)|WalkingStraightForwards',text,re.I):reasons.append('no_forward_evidence')
 decisions.append(dict(id=r['id'],accepted=not reasons,reasons=reasons,**{k:a[k] for k in ('source_family','captions','low_foot_p95_p05_m','straightness')}))
 if not reasons:accepted.append(r)
accepted=balance(accepted)
assert sum(r['category']=='walking' for r in accepted)>=100
original_ids=[r['id'] for r in source['motions']];spec=source['contact_reference'];p=Path(spec['file'])
assert hashlib.sha256(p.read_bytes()).hexdigest()==spec['sha256']
loaded=np.load(p);a={k:loaded[k] for k in loaded.files};offsets=np.r_[0,np.cumsum(a['lengths'])];assert a['clip_ids'].tolist()==original_ids
ix=[original_ids.index(r['id']) for r in accepted]
b={k:a[k].copy() for k in a}
for k in ('contact','known','source_contact4'):b[k]=np.concatenate([a[k][offsets[i]:offsets[i+1]] for i in ix])
b['clip_ids']=a['clip_ids'][ix];b['lengths']=a['lengths'][ix]
cp=out/'train.contacts.npz';np.savez_compressed(cp,**b)
source['motions']=accepted;source['contact_reference']=dict(spec,file=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
(out/'train.json').write_text(json.dumps(source,indent=2))
(out/'selection_decisions.json').write_text(json.dumps(decisions,indent=2))
held=json.load(open(root/'tw123_v1/contact/splits/val.json'))['motions']+json.load(open(root/'tw123_v1/contact/splits/test.json'))['motions']
assert not ({r['source_family'] for r in accepted}&{r['source_family'] for r in held})
summary=dict(policy='Conservative flat-forward subset from existing train; excluded clips preserved, not relabelled or flattened',
 counts={c:sum(r['category']==c for r in accepted) for c in ('walking','standing_upper')},frames=sum(r['frames'] for r in accepted),
 category_mass={c:sum(r['weight']*(r['frames']-1) for r in accepted if r['category']==c) for c in ('walking','standing_upper')},
 manifest_sha256=hashlib.sha256((out/'train.json').read_bytes()).hexdigest(),contact_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),source_family_disjoint=True)
(out/'data_audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
