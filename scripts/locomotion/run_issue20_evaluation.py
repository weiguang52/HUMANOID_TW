"""Fixed-budget evaluation; symmetric-USD prerequisite first, stop if it fails."""
import os,time,json,subprocess,signal,shutil
from pathlib import Path
from issue20_common import ROOT,OLD,REPO,sha,save
COMMANDS=['forward_03','forward_05','forward_08','stand']
def check_wait():
 while not (ROOT/'no_policy_standing.json').exists():
  pid=json.loads((ROOT/'usd_preflight_driver.json').read_text())['driver_pid'];assert Path('/proc/'+str(pid)).exists(),'USD preflight driver terminal without completed gates'
  save('evaluation_progress.json',{'state':'waiting_for_USD_preflight','driver_pid':os.getpid()});time.sleep(10)
 for name in ['actor_phase_validation.json','zero_phase_reproduction.json','strict_usd_preflight.json','runtime_strict_model_gate.json','no_policy_standing.json']:
  assert json.loads((ROOT/name).read_text())['passed'],name

def main():
 os.chdir(REPO);assert not (ROOT/'evaluation_records.json').exists(),'Audit partial formal records before resumption';check_wait()
 files={}
 for pattern in ['scripts/locomotion/*issue20*.py','scripts/locomotion/issue20_*.py']:
  for p in REPO.glob(pattern):files[str(p)]=sha(p)
 for name in ['eval_issue19.py','eval_cmd.py','issue19_actor.py','issue19_tracking.py','issue18_policy.py','issue18_tracking.py','issue17_symmetry.py','issue16_mirror.py','issue16_tracking.py','issue15_tracking.py','issue13_tracking.py','issue20_common.py']:
  p=REPO/'scripts/locomotion'/name;files[str(p)]=sha(p)
 for p in (REPO/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion').rglob('*.py'):files[str(p)]=sha(p)
 files[str(ROOT/'classification_rules.json')]=sha(ROOT/'classification_rules.json');files[str(ROOT/'strict_symmetric.usd')]=sha(ROOT/'strict_symmetric.usd');save('frozen_evaluation_sources.json',{'files':files,'frozen_at':time.time(),'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
 models=json.loads((ROOT/'cohort.json').read_text());done=[]
 for usd in ['symmetric','original']:
  if usd=='original' and not json.loads((ROOT/'symmetric_analysis.json').read_text())['prerequisite_passed']:break
  for model in models:
   for es in [1,2,3]:
    for phase in [0,10]:
     assert not (ROOT/'stop').exists();assert shutil.disk_usage(ROOT).free>2*1024**3
     for p,h in files.items():assert sha(p)==h,p
     assert sha(model['checkpoint'])==model['sha256'];ts=model['training_seed'];name=f'{usd}_s{ts}_eval{es}_k{phase}';folder=ROOT/'evaluation'/usd;folder.mkdir(parents=True,exist_ok=True);out=folder/(name+'.json');mom=folder/'moments'/name
     a=['python','-u','scripts/locomotion/eval_issue20.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1024','--seed',str(es),'--checkpoint',model['checkpoint'],'--experiment_profile','robust08','--mass_randomization','bounded','--last_action_observation','requested','--issue5_condition','clean','--extended_commands','--command_filter',','.join(COMMANDS),'--paired_resets','--issue18_mode','original','--issue19_mode','sym_wrap','--issue18_records',str(mom),'--output',str(out),'--issue20_phase',str(phase)]
     if usd=='symmetric':a+=['--issue20_usd',str(ROOT/'strict_symmetric.usd')]
     started=time.time()
     with out.with_suffix('.log').open('w') as log:
      child=subprocess.Popen(a,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('evaluation_progress.json',{'state':'evaluating','driver_pid':os.getpid(),'child_pid':child.pid,'job':name,'completed':len(done),'maximum_jobs':60,'started_at':started})
      while child.poll() is None:
       if (ROOT/'stop').exists():os.killpg(child.pid,signal.SIGTERM);child.wait();raise RuntimeError('Explicit stop requested')
       time.sleep(5)
     assert child.returncode==0 and out.exists(),(name,child.returncode)
     v=json.loads(out.read_text());assert v['issue19_actor']=='sym_wrap' and v['issue20']['phase_offset']==phase and v['issue18_tracking']['model_optimizer_state_unchanged'];assert all(x['num_envs']==1024 and x['seconds']==20 for x in v['results'])
     if usd=='original' and phase==0:
      import numpy as np
      old=OLD/'evaluation/clean/moments'/f'clean_sym_wrap_s{ts}_eval{es}'
      for cmd in COMMANDS:
       with np.load(old/(cmd+'_yaw.npz')) as x,np.load(mom/(cmd+'_yaw.npz')) as y:
        for key in x.files:assert np.array_equal(x[key],y[key],equal_nan=True),(name,cmd,key)
     done.append({'name':name,'usd':usd,'training_seed':ts,'eval_seed':es,'phase':phase,'checkpoint':model['checkpoint'],'checkpoint_sha256':model['sha256'],'output':str(out),'output_sha256':sha(out),'commands':COMMANDS,'wall_seconds':time.time()-started});save('evaluation_records.json',done);print('ISSUE20_FORMAL_DONE',len(done),name,flush=True)
  subprocess.run(['python','scripts/locomotion/analyze_issue20.py',usd],check=True)
  if usd=='symmetric' and not json.loads((ROOT/'symmetric_analysis.json').read_text())['prerequisite_passed']:
   save('prerequisite_stop.json',{'required_by_issue':True,'reason':'Symmetric USD mirrored-trajectory prerequisite failed or incomplete; do not infer Phi/Gamma or run further original USD cells','completed_formal_jobs':len(done),'skipped_original_USD_jobs':30,'recommendation':'Investigate simulation equivariance, including contact solver and finite-precision clock/state effects; no extra experiments in this issue','training_updates':0})
 save('evaluation_progress.json',{'state':'complete_with_prerequisite_stop' if len(done)==30 else 'complete','driver_pid':os.getpid(),'completed':len(done),'training_updates':0});(ROOT/'formal_evaluation_complete').write_text(str(len(done))+' formal jobs; stopped only if prerequisite required it.\n');print('ISSUE20_FORMAL_EVALUATION_TERMINAL',len(done),flush=True)
if __name__=='__main__':main()
