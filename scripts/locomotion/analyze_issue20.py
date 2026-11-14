"""Compute full factorial cells and paired trial/seed S,D with prerequisite stopping."""
from pathlib import Path
import json,numpy as np
from issue20_common import ROOT,sha,save
from issue15_analysis_data import load_yaw,job_statistics
from issue19_analysis import decompose,summarize_decomposition
from issue20_analysis import half_cycles,paired
from issue18_analysis import summary,complete_mean
COMMANDS=['forward_03','forward_05','forward_08','stand']
def avg(a):
 a=np.asarray(a);v=a[np.isfinite(a)];return float(v.mean()) if len(v) else None
def includes_zero(v):return v['ci95'] is not None and v['ci95'][0]<=0<=v['ci95'][1]
def analyze(usd):
 records=[r for r in json.loads((ROOT/'evaluation_records.json').read_text()) if r['usd']==usd];assert len(records)==30
 jobs={};cells={};cache={};paired_jobs={};inputs={};rawdir=ROOT/'paired_raw'/usd;rawdir.mkdir(parents=True,exist_ok=True)
 for row in records:
  output=Path(row['output']);assert sha(output)==row['output_sha256'];v=json.loads(output.read_text());mom=Path(v['issue18_tracking']['output']);assert v['issue18_tracking']['model_optimizer_state_unchanged']
  assert v['issue20']['phase_offset']==row['phase'];assert not v['observation_noise'];assert [x['command'] for x in v['results']]==COMMANDS
  for result in v['results']:
   command=result['command'];z,y,m,meta=load_yaw(mom,command,result);idx=np.load(mom/(command+'_phase20.npz'))['pre_action_idx'];assert idx.shape==(1000,1024)
   h,hr=half_cycles(y['heading_yaw'],y['heading_start'],m,idx,row['phase']);d=decompose(y['heading_yaw'],y['heading_start'],m);stat=job_statistics(z,y,meta,result);stat.update(h);stat.update(summarize_decomposition(d));signed=y['heading_delta_sum'][1]*180/np.pi;full=d['survived20s'];stat.update(signed20_degrees=avg(np.where(full,signed,np.nan)),absolute20_degrees=avg(np.where(full,abs(signed),np.nan)),full20s_count=int(full.sum()),initial_trials=1024)
   key=(row['training_seed'],row['eval_seed'],row['phase'],command);jobs[':'.join(map(str,key))]=stat;cache[key]=(y,m,hr,mom)
   for p in mom.glob(command+'*'):
    if p.is_file():inputs[str(p)]=sha(p)
 for ts in range(1,6):
  for es in [1,2,3]:
   for command in COMMANDS:
    y0,m0,h0,mom0=cache[ts,es,0,command];y1,m1,h1,mom1=cache[ts,es,10,command];p=paired(y0['heading_yaw'],y0['heading_start'],m0,y1['heading_yaw'],y1['heading_start'],m1)
    # Compare actual written initial conditions, including origin-relative polar/axial fields.
    a=np.load(mom0/(command+'_initial_state.npz'));b=np.load(mom1/(command+'_initial_state.npz'));contract=json.loads((mom0/'mirror_runtime_contract.json').read_text());names=contract['joint_names'];from issue16_mirror import Mirror101
    mirror=Mirror101(names,contract['action_names'],contract['action_scales']);perm=mirror.perm.numpy();sign=mirror.sign.numpy();expected=a['root'].copy();origin=a['env_origins'];expected[:,:3]=(expected[:,:3]-origin)*np.array([1,-1,1],np.float32)+origin;expected[:,3:7]*=np.array([1,-1,1,-1],np.float32);expected[:,7:10]*=np.array([1,-1,1],np.float32);expected[:,10:13]*=np.array([-1,1,-1],np.float32)
    errors={'root':float(abs(b['root']-expected).max()),'joint_pos':float(abs(b['joint_pos']-a['joint_pos'][:,perm]*sign).max()),'joint_vel':float(abs(b['joint_vel']-a['joint_vel'][:,perm]*sign).max()),'last_action':float(abs(b['last_action']-a['last_action'][:,perm]*sign).max())};assert max(errors.values())==0,(ts,es,command,errors)
    p['half_swap_left_residual_degrees']=h1['complete_cycle_left_degrees']+h0['complete_cycle_right_degrees'];p['half_swap_right_residual_degrees']=h1['complete_cycle_right_degrees']+h0['complete_cycle_left_degrees']
    np.savez_compressed(rawdir/(f's{ts}_eval{es}_{command}.npz'),**p)
    paired_jobs[f'{ts}:{es}:{command}']={k:avg(v) for k,v in p.items() if k!='paired_complete20s'}|{'paired_complete20s_count':int(p['paired_complete20s'].sum()),'initial_state_mirror_errors':errors}
 for command in COMMANDS:
  for phase in [0,10]:
   seedvalues=[{k:complete_mean([jobs[f'{ts}:{es}:{phase}:{command}'][k] for es in [1,2,3]]) for k in jobs[f'{ts}:1:{phase}:{command}'] if isinstance(jobs[f'{ts}:1:{phase}:{command}'][k],(int,float)) or jobs[f'{ts}:1:{phase}:{command}'][k] is None} for ts in range(1,6)]
   cells[f'{phase}:{command}']={k:summary([v[k] for v in seedvalues]) for k in seedvalues[0]}
 paired_seeds={};paired_groups={}
 for command in COMMANDS:
  vals=[{k:complete_mean([paired_jobs[f'{ts}:{es}:{command}'][k] for es in [1,2,3]]) for k in paired_jobs[f'{ts}:1:{command}'] if k!='initial_state_mirror_errors'} for ts in range(1,6)];paired_seeds[command]=vals;paired_groups[command]={k:summary([v[k] for v in vals]) for k in vals[0]}
 gates={command:{'S_mean_degrees':paired_groups[command]['S20_degrees']['mean'],'S_ci95':paired_groups[command]['S20_degrees']['ci95'],'S_within5':paired_groups[command]['S20_degrees']['mean'] is not None and abs(paired_groups[command]['S20_degrees']['mean'])<=5,'half_swap_left_ci_includes0':includes_zero(paired_groups[command]['half_swap_left_residual_degrees']),'half_swap_right_ci_includes0':includes_zero(paired_groups[command]['half_swap_right_residual_degrees']),'all_initial_trials_have20s_pairs':all(v['paired_complete20s_count']==1024 for v in paired_seeds[command])} for command in COMMANDS}
 passed=all(v['S_within5'] and v['half_swap_left_ci_includes0'] and v['half_swap_right_ci_includes0'] and v['all_initial_trials_have20s_pairs'] for v in gates.values())
 save(f'{usd}_analysis.json',{'job_statistics':jobs,'cell_statistics':cells,'paired_job_statistics':paired_jobs,'paired_seed_statistics':paired_seeds,'paired_group_statistics':paired_groups,'gates':gates,'prerequisite_passed':passed,'input_sha256':inputs,'rules_sha256':sha(ROOT/'classification_rules.json')})
 return passed
if __name__=='__main__':
 import sys;print('ISSUE20_ANALYZED',sys.argv[1],analyze(sys.argv[1]),flush=True)
