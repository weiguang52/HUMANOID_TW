"""Apply the frozen30degree heading gate to immutable issue15 lineage data only."""
import csv,json,os,subprocess
from pathlib import Path
from issue16_common import ROOT,REPO,sha
from issue15_math import paired_summary
A=REPO/'validation_artifacts/stage1_issue15_20261111'
def load(n):return json.loads((A/n).read_text())
def main():
 os.chdir(REPO)
 c=load('cohort.json');n=load('node_statistics.json');seed=load('seed_statistics.json');jobs=load('job_statistics.json')
 for name in ['cohort.json','node_statistics.json','seed_statistics.json','job_statistics.json']:
  assert (A/name).read_bytes()==subprocess.check_output(['git','show','770014ee9a4393da6dff34eced89a04d83e788dd:'+str((A/name).relative_to(REPO))])
 commands=['stand','forward_03','forward_05','forward_08'];rows=[];by_node={}
 for node in c['display_order']:
  by_node[node]={}
  for command in commands:
   stats=n[node][command]['metrics'];key='full20s_survivor_heading_mean_abs_degrees';s=stats[key]
   values=[v[key] for v in seed[node][command].values()]
   if all(v is not None for v in values):assert paired_summary(values)==s
   counts=[jobs[f"{c['models'][name]['evaluation_model']}:eval{es}:{command}"] for name in c['nodes'][node] for es in [1,2,3]]
   survived=sum(v['full20s_heading_survivors'] for v in counts);originalN=sum(v['original_trial_count'] for v in counts)
   mean=s['mean'] if s else None;ci=s['bootstrap95'] if s else None
   row=dict(node=node,command=command,training_seeds=n[node][command]['training_seeds'],original_initial_trials=originalN,
    full20s_survivors=survived,failed_initial_trials=originalN-survived,mean_absolute20s_degrees=mean,
    training_bootstrap95_lower=ci[0] if ci else None,training_bootstrap95_upper=ci[1] if ci else None,
    signed20s_degrees=stats['full20s_survivor_heading_signed_mean_degrees']['mean'] if stats['full20s_survivor_heading_signed_mean_degrees'] else None,
    heading_mean_gate_pass=mean is not None and mean<=30,conservative_upper95_gate_pass=ci is not None and ci[1]<=30,
    endpoint_status='missing_or_censored_in_some_seed' if s is None else 'survivor_selected_endpoint' if survived<originalN else 'complete_no_reset20s_endpoint',
    initial_prefix_mean_abs_degrees=stats['initial_prefix_all_heading_mean_abs_degrees']['mean'],
    initial_prefix_mean_duration_s=stats['initial_prefix_mean_duration_s']['mean'],
    all_episode_segments_mean_abs_degrees=stats['all_episode_segments_heading_mean_abs_degrees']['mean'],
    training_seed_signed_means={k:v['full20s_survivor_heading_signed_mean_degrees'] for k,v in seed[node][command].items()},
    training_seed_absolute_means={k:v[key] for k,v in seed[node][command].items()})
   by_node[node][command]=row;rows.append(row)
 node_flags={node:all(v['heading_mean_gate_pass'] for v in commands.values()) for node,commands in by_node.items()}
 result=dict(gate_degrees=30,primary_gate='four finite command means <=30; CI upper reported separately',training_updates=0,baseline_evaluations_rerun=0,rows=rows,all_four_heading_gate_by_node=node_flags,source_sha256=sha(__file__),inputs={str(A/name):sha(A/name) for name in ['cohort.json','node_statistics.json','seed_statistics.json','job_statistics.json']})
 (ROOT/'lineage_heading_gate.json').write_text(json.dumps(result,indent=2)+'\n')
 flat=[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in rows]
 with (ROOT/'lineage_heading_gate.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 lines=['# Frozen30degree heading gate applied to issue15 data','',
  'Zero re-evaluation of the lineage. Endpoint means remain conditional on no-reset20s survivors; prefixes/falls are never treated as full20s endpoints. Upper95 is a separately reported conservative flag, not a changed primary mean gate. Single-training-seed cells have no training CI.','',
  '|Node|Command|Train n|Survived / originalN|Mean abs degrees|Training95 upper|Mean<=30|Endpoint scope|','|---|---|---:|---:|---:|---:|---|---|']
 for r in rows:
  lines.append('|'+ '|'.join(str(r[k]) for k in ['node','command','training_seeds'])+f"|{r['full20s_survivors']}/{r['original_initial_trials']}|{r['mean_absolute20s_degrees']}|{r['training_bootstrap95_upper']}|{r['heading_mean_gate_pass']}|{r['endpoint_status']}|")
 lines+=['','All-four mean-gate node flags: '+json.dumps(node_flags), '',
  'All signed training-seed values, prefix drift/duration and reset-excluded segment metrics remain in lineage_heading_gate.json. A negative group mean is not evidence that every seed or trial turns the same direction.']
 (ROOT/'lineage_heading_gate.md').write_text('\n'.join(lines)+'\n')
 print('ISSUE16_BASELINE_GATE_COMPLETE',len(rows),node_flags,flush=True)
if __name__=='__main__':main()
