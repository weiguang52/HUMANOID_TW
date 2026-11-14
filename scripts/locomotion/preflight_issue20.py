"""Full-size zero-factor action/state replay and archived issue19 heading equality."""
import subprocess,json,time,os
import numpy as np
from issue20_common import ROOT,OLD,REPO,sha,save

def args(name,baseline=False,phase=0,usd=''):
 row=json.loads((ROOT/'cohort.json').read_text())[0];out=ROOT/'preflight'/name/'result.json';out.parent.mkdir(parents=True,exist_ok=True)
 a=['python','-u','scripts/locomotion/eval_issue20.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1024','--seed','1','--checkpoint',row['checkpoint'],'--experiment_profile','robust08','--mass_randomization','bounded','--last_action_observation','requested','--issue5_condition','clean','--extended_commands','--command_filter','stand,forward_03,forward_05,forward_08','--paired_resets','--issue18_mode','original','--issue19_mode','sym_wrap','--issue18_records',str(out.parent/'moments'),'--output',str(out),'--issue20_phase',str(phase)]
 if baseline:a+=['--issue20_baseline']
 if usd:a+=['--issue20_usd',usd]
 return a,out

def run(name,**kw):
 a,out=args(name,**kw);started=time.time()
 with out.with_suffix('.log').open('w') as log:
  child=subprocess.Popen(a,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
  save('preflight_progress.json',{'state':name,'driver_pid':os.getpid(),'child_pid':child.pid,'started_at':started})
  code=child.wait()
 assert code==0 and out.exists(),(name,code);return out

def main():
 os.chdir(REPO);assert json.loads((ROOT/'actor_phase_validation.json').read_text())['passed']
 baseline=run('original_evaluator_observed',baseline=True);candidate=run('zero_phase_candidate')
 oldrow=next(x for x in json.loads((OLD/'evaluation_records.json').read_text()) if x['name']=='clean_sym_wrap_s1_eval1');old=__import__('pathlib').Path(oldrow['output']).parent/'moments'/oldrow['name'];checks=[]
 for command in ['stand','forward_03','forward_05','forward_08']:
  a=baseline.parent/'moments';b=candidate.parent/'moments';assert (a/(command+'_step_sha256.json')).read_bytes()==(b/(command+'_step_sha256.json')).read_bytes(),command
  with np.load(old/(command+'_yaw.npz')) as x,np.load(b/(command+'_yaw.npz')) as y:
   for key in x.files:assert np.array_equal(x[key],y[key],equal_nan=True),(command,key)
  checks.append({'command':command,'steps':1000,'action_and_root_joint_state_bitwise_equal':True,'all_archived_issue19_yaw_arrays_bitwise_equal':True})
 save('zero_phase_reproduction.json',{'passed':True,'num_envs':1024,'evaluation_seed':1,'training_seed':1,'checks':checks,'baseline_observer_only':True,'baseline_output':str(baseline),'candidate_output':str(candidate),'training_updates':0})
 save('preflight_progress.json',{'state':'zero_phase_reproduction_complete','driver_pid':os.getpid()});print('ISSUE20_ZERO_PHASE_REPRODUCTION_PASS',flush=True)
if __name__=='__main__':main()
