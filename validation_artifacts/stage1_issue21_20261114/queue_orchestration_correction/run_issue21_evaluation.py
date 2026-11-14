"""Sequential frozen diagnostic queue; no model or default edits."""
import os,subprocess,time,json,shutil
from pathlib import Path
from issue21_common import ROOT,OLD,REPO,sha,save
os.chdir(REPO);assert not (ROOT/'evaluation_records.json').exists(),'Audit interrupted records before resume'
models=json.loads((ROOT/'cohort.json').read_text());files={str(p):sha(p) for p in (REPO/'scripts/locomotion').glob('*issue21*.py')}
for n in ['issue20_runtime.py','issue19_actor.py','issue17_symmetry.py','issue16_mirror.py','eval_cmd.py']:files[str(REPO/'scripts/locomotion'/n)]=sha(REPO/'scripts/locomotion'/n)
for p in (REPO/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion').rglob('*.py'):files[str(p)]=sha(p)
files[str(ROOT/'classification_rules.json')]=sha(ROOT/'classification_rules.json');files[str(OLD/'strict_symmetric.usd')]=sha(OLD/'strict_symmetric.usd');save('frozen_evaluation_sources.json',{'files':files,'frozen_at':time.time()});done=[]
for mode,counts in [('ground',[16,1]),('air',[16,1]),('perturb',[1024])]:
 for model in models:
  for n in counts:
   for p,d in files.items():assert sha(p)==d,p
   assert sha(model['checkpoint'])==model['sha256'];assert shutil.disk_usage(ROOT).free>3*1024**3
   name='%s_s%d_n%d'%(mode,model['training_seed'],n);out=ROOT/'evaluation'/(name+'.json');out.parent.mkdir(exist_ok=True);assert not out.exists()
   a=['python','-u','scripts/locomotion/eval_issue21.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs',str(n),'--seed','1','--checkpoint',model['checkpoint'],'--experiment_profile','robust08','--mass_randomization','bounded','--last_action_observation','requested','--issue5_condition','clean','--paired_resets','--seconds','20','--issue21_mode',mode,'--training_seed',str(model['training_seed']),'--output',str(out)]
   started=time.time()
   with out.with_suffix('.log').open('w') as log:
    child=subprocess.Popen(a,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('evaluation_progress.json',{'state':'running','job':name,'completed':len(done),'total_jobs':25,'driver_pid':os.getpid(),'child_pid':child.pid,'started_at':started});code=child.wait()
   assert code==0 and out.exists(),(name,code)
   v=json.loads(out.read_text());assert v['model_unchanged'];done.append({'name':name,'output':str(out),'sha256':sha(out),'seconds':time.time()-started});save('evaluation_records.json',done);print('ISSUE21_DONE',len(done),name,flush=True)
save('evaluation_progress.json',{'state':'formal_diagnostics_complete','jobs':len(done),'driver_pid':os.getpid()});(ROOT/'formal_evaluation_complete').write_text('25 jobs complete\n')
