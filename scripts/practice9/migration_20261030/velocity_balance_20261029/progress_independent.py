from pathlib import Path
import json,datetime
b=Path(__file__).parent;rows=[];pending=[]
def clean(m):
 t=m['termination_counts'];return t.get('motion_end',0)>0 and not any(v for k,v in t.items() if k not in ('motion_end','time_out'))
for s in (b,b.parent/'velocity_balance_seed123_20261029'):
 r=s/'independent';manifest=json.load(open(r/'manifest.json'))['motions']
 for seed in (42,123,2026):
  log=r/'candidate'/f'seed{seed}.log';jobs=json.load(open(r/'candidate'/f'jobs{seed}.json'))
  text=log.read_text() if log.exists() else ''
  if not all('BATCH_COMPLETED '+j['output'] in text for j in jobs):pending.append([s.name,seed]);continue
  accepted={j['motion_id']:j for j in json.load(open(r/'accepted'/f'jobs{seed}.json'))};pairs=[]
  for j in jobs:
   i=j['motion_id'];a=json.load(open(accepted[i]['output']));c=json.load(open(j['output']));pairs.append({'id':manifest[i]['id'],'accepted':clean(a),'candidate':clean(c)})
  assert len(pairs)==7
  row={'state':s.name,'evaluation_seed':seed,'pairs':pairs,'lost':[v for v in pairs if v['accepted'] and not v['candidate']],'gained':[v for v in pairs if not v['accepted'] and v['candidate']]};rows.append(row);print(s.name,seed,'lost',row['lost'],'gained',row['gained'])
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete_full21_both':len(rows)==6,'rows':rows,'pending':pending,'scope':'Completed whole independent evaluation-seed groups only. Does not replace full65 replay, fixed24 joint metrics, body fidelity or final visual acceptance.'}
(b/'independent_completed_progress.json').write_text(json.dumps(out,indent=2));print('pending',pending)
