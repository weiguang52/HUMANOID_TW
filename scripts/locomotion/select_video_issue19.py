"""Select new sym_wrap forward0.5 representative from all15360 initial trials."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/root/gpufree-data/tw_stage1/issue19_20261113')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert json.loads((ROOT/'evaluation_progress.json').read_text())['state']=='complete'
 jobs=[dict(j,output=str(ROOT/'evaluation/clean'/(j['name']+'.json'))) for j in json.loads((ROOT/'evaluation_job_templates.json').read_text()) if j['group']=='sym_wrap' and j['protocol']=='clean'];assert len(jobs)==15
 models={r['group']+'_s'+str(r['training_seed']):r for r in json.loads((ROOT/'training_records.json').read_text())};loaded=[];inputs={};censored=0
 for j in jobs:
  p=Path(j['output']);d=json.loads(p.read_text());source=Path(d['issue18_tracking']['output'])/'forward_05_yaw.npz'
  with np.load(source) as z:failure=z['first_failure_step'].copy();h=z['heading_delta_sum'].copy()*180/np.pi;duration=z['heading_count'][1].copy()*.02
  censored+=int(((failure>=0)|(duration<20)).sum());loaded.append((j,h,failure,duration));inputs[str(source)]=sha(source);inputs[str(p)]=sha(p)
 scope=1 if censored else 0;candidates=[]
 for j,h,failure,duration in loaded:
  model=models[j['model']]
  for i in range(1024):candidates.append(dict(model=j['model'],group='sym_wrap',training_seed=j['training_seed'],eval_seed=j['eval_seed'],env_id=i,heading_abs_degrees=float(abs(h[scope,i])),heading_signed_degrees=float(h[scope,i]),initial_prefix_duration_s=float(duration[i]),survived20s=bool(failure[i]<0 and duration[i]==20),original_job=j['output'],checkpoint=model['checkpoint'],checkpoint_sha256=model['sha256']))
 assert len(candidates)==15360
 median=float(np.median([r['heading_abs_degrees'] for r in candidates]));selected=min(candidates,key=lambda r:(abs(r['heading_abs_degrees']-median),r['training_seed'],r['eval_seed'],r['env_id']))
 selected.update(command='forward_05',candidate_count=15360,survival_filter=False,censored_initial_trials=censored,selection_scope='initial_prefix_all_trials' if censored else 'full20s_all_initial_trials',selection_metric='absolute reset-aware initial-prefix heading for every initial trial (censored exposure disclosed)' if censored else 'absolute reset-aware20s heading for all initial trials',group_median_absolute_heading_degrees=median,selection_inputs_sha256=inputs,source_policy_mode='trained sym_wrap actor inference, without additional projection',source_sha256=sha(__file__))
 (ROOT/'representative_selection.json').write_text(json.dumps(selected,indent=2)+'\n');print('SYM_INT_MEDIAN_SELECTION',selected['model'],selected['env_id'],selected['selection_scope'],flush=True)
if __name__=='__main__':main()
