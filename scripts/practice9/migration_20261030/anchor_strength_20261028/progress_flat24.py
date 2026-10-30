"""Use unchanged canonical metric definitions once all fixed24 flats finish.
All other evaluations continue. This is not the final acceptance decision.
"""
from pathlib import Path
import json,contextlib,io,datetime
b=Path(__file__).parent;states=[b,b.parent/'anchor_strength_seed123_20261028'];pending=[];out=[]
for s in states:
 r=s/'evaluation';manifest=json.load(open(r/'manifest.json'))['motions']
 for seed in (42,123,2026):
  log=(r/'candidate'/f'seed{seed}.log').read_text() if (r/'candidate'/f'seed{seed}.log').exists() else ''
  flat=[j for j in json.load(open(r/'candidate'/f'jobs{seed}.json')) if manifest[j['motion_id']].get('evaluation_class',manifest[j['motion_id']]['category'])=='flat_forward'];assert len(flat)==8
  pending.extend([j['output'] for j in flat if 'BATCH_COMPLETED '+j['output'] not in log])
if pending:print('WAITING',len(pending),'fixed flat trials');raise SystemExit(0)
for s in states:
 r=s/'evaluation';source=(r/'analyze.py').read_text();needle='  for job in jobs:\n';assert source.count(needle)==1
 source=source.replace(needle,needle+"   if manifest[job['motion_id']].get('evaluation_class',manifest[job['motion_id']]['category'])!='flat_forward':continue\n").replace("(r/'metrics.json').write_text","(r/'flat24_progress_metrics.json').write_text")
 with contextlib.redirect_stdout(io.StringIO()):exec(compile(source,str(r/'analyze.py'),'exec'),{'__file__':str(r/'analyze.py')})
 source=(r/'summarize_flat.py').read_text().replace("(r/'metrics.json').read_text()","(r/'flat24_progress_metrics.json').read_text()").replace("(r/'flat_three_seed_summary.json').write_text","(r/'flat24_progress_summary.json').write_text")
 with contextlib.redirect_stdout(io.StringIO()):exec(compile(source,str(r/'summarize_flat.py'),'exec'),{'__file__':str(r/'summarize_flat.py')})
 d=json.load(open(r/'flat24_progress_summary.json'));assert d['complete'] and d['flat_pairs']==24
 parent_name=s.name.replace('anchor_strength','gae_trace');parent=json.load(open(b.parent/parent_name/'evaluation/flat_three_seed_summary.json'))
 diff={k:100*(d['summary']['candidate'][k]/parent['summary']['candidate'][k]-1) for k in d['summary']['candidate']} if d.get('summary') else None
 row={'state':s.name,'fixed24':d,'parent_path_balance_change_percent':diff};out.append(row);print(s.name,'all24_complete_same_duration',d['all_complete_same_duration'],'changes',d.get('candidate_change_percent'),'vs_parent',diff)
(b/'flat24_progress.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'full65_complete':False,'scope':'Only original fixed24 flat pairs, unchanged canonical metric definitions. Full groups, independent cases, replay audit, fidelity, videos and final acceptance remain required.','rows':out},indent=2))
