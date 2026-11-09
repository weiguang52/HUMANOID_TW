"""Secondary held-out-seed concentration check; no held-out ranking labels."""
import collections,json,hashlib
from pathlib import Path

def greedy_subset(cells,all_trial_count,budget=.2,min_n=20):
 candidates=[(key,v) for key,v in cells.items() if v['n']>=min_n and v['k']>0]
 candidates.sort(key=lambda kv:(-kv[1]['k']/kv[1]['n'],-kv[1]['n'],tuple(map(str,kv[0]))))
 selected=set();union=set()
 for key,v in candidates:
  expanded=union|v['trial_ids']
  if len(expanded)<=budget*all_trial_count+1e-12:selected.add(key);union=expanded
 return selected,union

def analyze(trials,pushes,root):
 low=[x for x in trials if x['stage']==4500 and x['command']=='forward_03'];ep=[x for x in pushes if x['stage']==4500 and x['command']=='forward_03' and x['eligible']];outputs={}
 for unit,rows,outcome in [('baseline',low,'failed'),('push',ep,'failure_within1s')]:
  factors=['static_q','dynamic_q','mass_q','delay_ms'] if unit=='baseline' else ['static_q','mass_q','delay_ms','amplitude_q','direction','support_group']
  def key(x):
   vals=[]
   for f in factors:
    vals.append(('single' if x['support_code'] in [1,2] else ('double' if x['support_code']==3 else 'flight')) if f=='support_group' else x[f])
   return tuple(vals)
  folds=[];cells_out=[]
  for held in range(1,6):
   training=[x for x in rows if x['training_seed']!=held];test=[x for x in rows if x['training_seed']==held];cells={}
   for x in training:
    k=key(x)
    if k not in cells:cells[k]={'n':0,'k':0,'trial_ids':set()}
    cells[k]['n']+=1;cells[k]['k']+=int(x[outcome]);cells[k]['trial_ids'].add(x['trial_id'])
   ntrain=sum(x['training_seed']!=held for x in low);selected,union=greedy_subset(cells,ntrain)
   chosen=[x for x in test if key(x) in selected];testtrials=[x for x in low if x['training_seed']==held];exposed={x['trial_id'] for x in chosen};failed=sum(x[outcome] for x in chosen);allfail=sum(x['failed'] for x in testtrials)
   folds.append({'held_out_seed':held,'all_trials':len(testtrials),'all_first_failures':allfail,'exposed_trials':len(exposed),'captured_first_failures':failed,'trial_coverage':len(exposed)/len(testtrials),'failure_capture':failed/allfail if allfail else None,'training_exposed_trials':len(union),'training_all_trials':ntrain,'selected_cells':len(selected)})
   for cell in sorted(selected,key=lambda v:tuple(map(str,v))):cells_out.append({'held_out_seed':held,'features':dict(zip(factors,cell)),'training_n':cells[cell]['n'],'training_events':cells[cell]['k']})
  n=sum(x['all_trials'] for x in folds);f=sum(x['all_first_failures'] for x in folds);covered=sum(x['exposed_trials'] for x in folds);captured=sum(x['captured_first_failures'] for x in folds)
  outputs[unit]={'folds':folds,'selected_cells':cells_out,'all_trials':n,'all_first_failures':f,'exposed_trials':covered,'captured_first_failures':captured,'held_out_trial_coverage':covered/n,'held_out_failure_capture':captured/f if f else None,'secondary_concentration_detected':bool(covered/n<=.2 and f>0 and captured/f>=.8),'limits':'secondary diagnostic fixed before stratified analysis; training folds overlap and physical streams shared; no independent confirmation or causal infeasibility claim'}
 (root/'secondary_concentration_cv.json').write_text(json.dumps(outputs,indent=2));return outputs
