"""Completed whole-seed group counts only; acceptance remains undecided."""
from pathlib import Path
import json,datetime
b=Path(__file__).parent;rows=[];pending=[]
def clean(meta):
 t=meta['termination_counts'];return t.get('motion_end',0)>0 and not any(v for k,v in t.items() if k not in ('motion_end','time_out'))
for suffix in ('','_seed123'):
 s=b.parent/f'velocity_balance{suffix}_20261029';manifest=json.load(open(s/'evaluation/manifest.json'))['motions']
 for seed in (42,123,2026):
  file=s/'evaluation/candidate'/f'jobs{seed}.json';done=Path(str(file)+'.done')
  if not done.exists() or done.read_text().strip()!='completed':pending.append([s.name,seed]);continue
  groups={};bad=[];log=(file.parent/f'seed{seed}.log').read_text()
  for model in ('accepted','candidate'):
   groups[model]={}
   for job in json.load(open(s/'evaluation'/model/f'jobs{seed}.json')):
    p=Path(job['output']);m=manifest[job['motion_id']];cat=m.get('evaluation_class',m['category']);meta=json.load(open(p))
    if model=='candidate':assert 'BATCH_COMPLETED '+str(p) in log
    pair=groups[model].setdefault(cat,[0,0]);pair[0]+=int(clean(meta));pair[1]+=1
    if model=='candidate' and not clean(meta):bad.append({'id':m['id'],'category':cat,'termination':meta['termination_counts']})
  preserved=all(groups['candidate'][cat][0]>=v[0] for cat,v in groups['accepted'].items())
  rows.append({'state':s.name,'eval_seed':seed,'groups':groups,'group_counts_preserved':preserved,'unclean_candidate':bad})
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete_fixed_matrix':len(rows)==6,'rows':rows,'pending':pending,'numeric_acceptance':None,'scope':'Only completed whole eval-seed groups; not flat metrics, independent actions, video or final acceptance'}
(b/'completed_group_progress.json').write_text(json.dumps(out,indent=2))
for row in rows:print(row['state'],row['eval_seed'],'groups_preserved',row['group_counts_preserved'],row['groups']['candidate'])
print('pending',pending)
