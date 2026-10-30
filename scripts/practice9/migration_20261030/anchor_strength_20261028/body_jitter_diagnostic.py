from pathlib import Path
import json,numpy as np,sys
s=Path(__file__).parent;case="candidate";scope="matrix"
r=s/"evaluation";seeds=[42,123,2026];rows=[];pending=[]

flat_file=r/'flat_three_seed_summary.json'
if flat_file.exists():
 flat=json.load(open(flat_file))
 if flat.get('complete') and not flat.get('all_complete_same_duration'):
  out={'complete':False,'summary':None,'expected_pairs':24,'reason':'Failed or unequal-duration flat; full24 quality aggregate unavailable; no subset substituted.','full_matrix_observed':True}
  (s/'body_jitter_candidate_matrix.json').write_text(json.dumps(out,indent=2));print(out);raise SystemExit(0)

manifest=json.loads((r/'manifest.json').read_text())['motions']
for seed in seeds:
 gold=json.loads((r/'accepted'/f'jobs{seed}.json').read_text());cand=json.loads((r/('candidate' if scope=='matrix' else case)/f'jobs{seed}.json').read_text())
 for j,k in zip(gold,cand):
  motion=manifest[j['motion_id']]
  if motion.get('evaluation_class',motion['category'])!='flat_forward':continue
  paths=[Path(x['output']) for x in [j,k]]
  if not all(p.exists() and p.with_suffix('.npz').exists() for p in paths):pending.append([seed,motion['id']]);continue
  arrays=[np.load(p.with_suffix('.npz')) for p in paths];ends=[]
  for p,x in zip(paths,arrays):
   meta=json.loads(p.read_text());tc=meta['termination_counts'];assert tc.get('motion_end',0)>0 and not any(v for a,v in tc.items() if a not in ['motion_end','time_out'])
   term=np.flatnonzero(x['terminal']);ends.append(int(term[0]) if len(term) else len(x['terminal']))
  assert ends[0]==ends[1] and np.array_equal(arrays[0]['reference_q'],arrays[1]['reference_q'])
  for model,x in zip(['accepted',case],arrays):
   names=x['joint_names'].tolist();groups={'legs12':[i for i,n in enumerate(names) if any(t in n for t in ['hip','knee','ankle'])],'foot_roll2':[i for i,n in enumerate(names) if 'foot_roll' in n],'waist3':[i for i,n in enumerate(names) if 'waist' in n],'arms10':[i for i,n in enumerate(names) if any(t in n for t in ['shoulder','elbow','wrist'])],'neck3':[i for i,n in enumerate(names) if 'neck' in n or 'head_pitch' in n]}
   assert [len(groups[k]) for k in groups]==[12,2,3,10,3] and sorted(sum(groups.values(),[]))==list(range(30))
   c=np.arange(50,ends[0]);ix=np.searchsorted(x['control_step'],c,side='right')-1;error=x['q'][ix]-x['reference_q'][c];error=error-error.mean(0);freq=np.fft.rfftfreq(len(error),float(x['control_dt']));bands={}
   for window in ['rectangular','hann']:
    w=np.ones(len(error)) if window=='rectangular' else np.hanning(len(error));z=np.fft.rfft(error*w[:,None],axis=0);z[freq<8]=0;hp=np.fft.irfft(z,n=len(error),axis=0);bands[window]={g:float(np.rad2deg(np.sqrt(np.mean(hp[:,ids]**2)/np.mean(w*w)))) for g,ids in groups.items()};bands[window]['all30']=float(np.rad2deg(np.sqrt(np.mean(hp**2)/np.mean(w*w))))
   rows.append({'model':model,'seed':seed,'id':motion['id'],'bands_deg':bands})
expected=len(seeds)*8;complete=len(rows)==2*expected and not pending;summary={}
if complete:
 for model in ['accepted',case]:
  data=[v for v in rows if v['model']==model];summary[model]={w:{g:float(np.mean([v['bands_deg'][w][g] for v in data])) for g in data[0]['bands_deg'][w]} for w in ['rectangular','hann']}
out={'complete':complete,'expected_pairs':expected,'observed_pairs':len(rows)//2,'case':case,'scope':scope,'summary':summary,'rows':rows,'pending':pending,'limitations':'Supplemental body-part >=8Hz q-reference error at control50Hz; does not replace fixed legs12 acceptance or visual review. Full matched successes only, no changed sample selection or threshold.'};(s/f'body_jitter_{case}_{scope}.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['rows','pending']},indent=2))
