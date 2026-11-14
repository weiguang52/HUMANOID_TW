"""Cold process per command/phase/condition; exact parent initial states."""
import os,subprocess,time,json,shutil
from pathlib import Path
from issue21_common import ROOT,OLD,REPO,sha,save
os.chdir(REPO);models=json.loads((ROOT/'cohort.json').read_text());files={str(REPO/'scripts/locomotion'/n):sha(REPO/'scripts/locomotion'/n) for n in ['eval_issue21.py','issue21_runtime.py','issue21_common.py','issue20_runtime.py','issue19_actor.py','issue17_symmetry.py','issue16_mirror.py','eval_cmd.py','issue21_open_noreset.py','eval_issue21_open_noreset.py']}
for p in (REPO/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion').rglob('*.py'):files[str(p)]=sha(p)
files[str(ROOT/'classification_rules.json')]=sha(ROOT/'classification_rules.json');files[str(OLD/'strict_symmetric.usd')]=sha(OLD/'strict_symmetric.usd')
for model in models:
 for cmd in ['stand','forward_05']:
  p=OLD/'evaluation/symmetric/moments'/('symmetric_s%d_eval1_k0'%model['training_seed'])/(cmd+'_initial_state.npz');files[str(p)]=sha(p)
if (ROOT/'frozen_evaluation_sources.json').exists():assert json.loads((ROOT/'frozen_evaluation_sources.json').read_text())['files']==files
else:save('frozen_evaluation_sources.json',{'files':files,'frozen_at':time.time(),'cold_start_per_case':True,'parent_initials_matched':True})
(ROOT/'evaluation').mkdir(exist_ok=True);(ROOT/'jobs').mkdir(exist_ok=True);(ROOT/'controls').mkdir(exist_ok=True)
plan=[]
for mode,kind in [('ground','closed'),('ground','open'),('air','open')]:
 for model in models:
  for n in [16,1]:
   for cmd in ['stand','forward_05']:
    for phase in [0,10]:plan.append({'mode':mode,'kind':kind,'model':model,'n':n,'command':cmd,'phase':phase,'epsilon':None})
for model in models:
 for epsilon in [0.,-.2,-.05,-.01,.01,.05,.2]:plan.append({'mode':'perturb','kind':'closed','model':model,'n':1024,'command':'forward_05','phase':0,'epsilon':epsilon})
assert len(plan)==155;save('cold_start_plan.json',plan);done=json.loads((ROOT/'cold_start_records.json').read_text()) if (ROOT/'cold_start_records.json').exists() else []
for rec in done:assert sha(rec['output'])==rec['sha256']
for item in plan:
 model=item['model'];ts=model['training_seed'];mode=item['mode'];n=item['n'];cmd=item['command'];kind=item['kind'];phase=item['phase'];eps=item['epsilon'];name=('%s_%s_s%d_n%d_%s_k%d'%(mode,kind,ts,n,cmd,phase)) if eps is None else ('perturb_s%d_e%+.2f'%(ts,eps));out=(ROOT/'controls'/('s%d.json'%ts)) if eps==0 else ROOT/'jobs'/(name+'.json')
 if any(x['name']==name for x in done):continue
 for p,d in files.items():assert sha(p)==d,p
 assert sha(model['checkpoint'])==model['sha256'];assert shutil.disk_usage(ROOT).free>3*1024**3;assert not out.exists(),out
 entry='eval_issue21_open_noreset.py' if mode=='ground' and kind=='open' else 'eval_issue21.py'
 a=['python','-u','scripts/locomotion/'+entry,'--task','Unitree-TW-30dof-Velocity','--headless','--num_envs',str(n),'--seed','1','--checkpoint',model['checkpoint'],'--experiment_profile','robust08','--mass_randomization','bounded','--last_action_observation','requested','--issue5_condition','clean','--paired_resets','--seconds','20','--issue21_mode',mode,'--training_seed',str(ts),'--issue21_kind',kind,'--issue21_phase',str(phase),'--issue21_command',cmd,'--output',str(out)]
 if eps is not None:a+=['--issue21_epsilon',str(eps)]
 started=time.time()
 with out.with_suffix('.log').open('w') as log:
  child=subprocess.Popen(a,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('evaluation_progress.json',{'state':'cold_start_running','job':name,'completed':len(done),'total_jobs':155,'logical_scope':25,'driver_pid':os.getpid(),'child_pid':child.pid,'started_at':started});code=child.wait()
 assert code==0 and out.exists(),(name,code);v=json.loads(out.read_text());assert v['model_unchanged'] and v['cold_start'] and len(v['records'])==1;done.append({'name':name,'output':str(out),'sha256':sha(out),'seconds':time.time()-started,'mode':mode,'kind':kind,'training_seed':ts,'num_envs':n,'command':cmd,'phase':phase,'epsilon':eps});save('cold_start_records.json',done);print('ISSUE21_COLD_CASE_DONE',len(done),name,flush=True)
logical=[]
for mode,ns in [('ground',[16,1]),('air',[16,1]),('perturb',[1024])]:
 for model in models:
  for n in ns:
   ts=model['training_seed'];group=[x for x in done if x['mode']==mode and x['training_seed']==ts and x['num_envs']==n and x['epsilon']!=0];assert len(group)==(8 if mode=='ground' else 4 if mode=='air' else 6);parts=[json.loads(Path(x['output']).read_text()) for x in group];meta=parts[0].copy();meta['records']=[v for x in parts for v in x['records']];meta['source_subjobs']=[x['output'] for x in group];meta['all_fresh_processes']=True;out=ROOT/'evaluation'/('%s_s%d_n%d.json'%(mode,ts,n));out.write_text(json.dumps(meta,indent=2)+'\n');logical.append({'name':out.stem,'output':str(out),'sha256':sha(out),'seconds':sum(x['seconds'] for x in group)})
assert len(logical)==25;save('evaluation_records.json',logical);save('evaluation_progress.json',{'state':'formal_diagnostics_complete','jobs':25,'cold_processes':len(done),'controls':5,'driver_pid':os.getpid()});(ROOT/'formal_evaluation_complete').write_text('25 logical groups,155 cold processes including5 independent zero-epsilon controls\n');print('ISSUE21_ALL_COLD_EVALUATIONS_COMPLETE',flush=True)
