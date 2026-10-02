"""Audit all rebuilt HumanML3D sources; select scoped categories without a cap."""
import csv,json,re,collections
from pathlib import Path
import numpy as np
DATA=Path('/root/gpufree-data/datasets')
ROOT=DATA/'HumanML3D-official/HumanML3D'
JOINTS=DATA/'practice9/humanml3d_rebuild/staging-v1/HumanML3D/new_joints'
OUT=DATA/'practice9/tw75_dataset_v1';OUT.mkdir(exist_ok=True)
WALK=re.compile(r'\b(walk\w*|march\w*|pac(?:e|es|ing))\b',re.I)
ARM=re.compile(r'\b(wav\w*|gestur\w*|clap\w*|point\w*|salut\w*|arm\w*|hand\w*)\b',re.I)
STAND=re.compile(r'\b(stand\w*|stationary|idle)\b',re.I)
EXCLUDE=re.compile(r'\b(run\w*|jog\w*|jump\w*|hop\w*|danc\w*|sit\w*|seat\w*|squat\w*|crouch\w*|kneel\w*|crawl\w*|lying|lie|lying|kick\w*|punch\w*|stairs?|climb\w*|chair|ladder|swim\w*|cartwheel\w*|somersault\w*)\b',re.I)
family={Path(r['new_name']).stem:r['source_path'] for r in csv.DictReader(open(DATA/'HumanML3D-official/index.csv'))}
splits={}
for split in ['train','val','test']:
 for id in (ROOT/(split+'.txt')).read_text().split():splits[id]=split
heldout={family[id.lstrip('M')] for id,s in splits.items() if s!='train' and id.lstrip('M') in family}
counts=collections.Counter();rows=[];reject=collections.Counter();sources=sorted(JOINTS.glob('*.npy'))
for source in sources:
 id=source.stem;counts['available_arrays']+=1
 if id.startswith('M'):counts['mirror_arrays']+=1;continue
 counts['original_arrays']+=1
 textpath=ROOT/'texts'/(id+'.txt')
 if not textpath.exists():reject['missing_caption']+=1;continue
 captions=[s.split('#')[0] for s in textpath.read_text().splitlines() if s.strip()];text=' '.join(captions)
 if EXCLUDE.search(text):reject['outside_scope_caption']+=1;continue
 category='walking' if WALK.search(text) else ('standing_upper' if ARM.search(text) or STAND.search(text) else None)
 if category is None:reject['no_scoped_caption']+=1;continue
 try:
  x=np.load(source,mmap_mode='r',allow_pickle=False)
  if x.ndim!=3 or x.shape[1:]!=(22,3) or len(x)<40 or not np.isfinite(x).all():reject['invalid_or_short']+=1;continue
  root=x[:,0];xy=root[:,[0,2]];path=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum());height=float(np.ptp(root[:,1]));foot=float(np.max(np.ptp(x[:,[7,8,10,11]][:,:,[0,2]],axis=0)))
  if category=='standing_upper' and (path>.30 or height>.12 or foot>.20):reject['standing_geometry_moving_or_height']+=1;continue
  split=splits.get(id)
  if split is None:reject['missing_official_split']+=1;continue
  group=family.get(id)
  if group is None:reject['missing_source_family']+=1;continue
  if split=='train' and group in heldout:reject['train_source_overlaps_heldout']+=1;continue
  rows.append(dict(id=id,source=str(source),category=category,split=split,source_family=group,captions=captions,frames=len(x),root_path_m=path,root_height_range_m=height,foot_xy_range_m=foot))
 except (ValueError,OSError) as e:reject['array_read_error']+=1
for cat in ['walking','standing_upper']:
 chosen=[r for r in rows if r['category']==cat]
 (OUT/(cat+'.inputs.txt')).write_text(''.join(r['source']+'\n' for r in chosen))
 for split in ['train','val','test']:
  subset=[r for r in chosen if r['split']==split];counts[cat+'_'+split]=len(subset)
  (OUT/(cat+'_'+split+'.ids.txt')).write_text(''.join(r['id']+'\n' for r in subset))
summary=dict(counts=counts,rejections=reject,total_caption_files=len(list((ROOT/'texts').glob('*.txt'))),selected=len(rows),method='Full available-array scan; no top-N cap. Conservative caption+geometry candidates, not quality-approved. Mirror arrays excluded as augmentation duplicates. Official splits retained and overlapping original source families removed from train. Caption keywords are imperfect; manual samples and retarget/support audits required.')
(OUT/'selection.json').write_text(json.dumps(rows,indent=2)+'\n');(OUT/'audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
