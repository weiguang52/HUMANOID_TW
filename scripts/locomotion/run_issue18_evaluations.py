"""Bounded, resumable queue: 45 strict-inference evaluations; never trains."""
import hashlib,json,os,shutil,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue18_20261113')
REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p); t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def audit(path,job,model):
 d=read(path); assert d['deterministic_policy'] is True
 assert d['issue17_clock']['integer_enabled']==job['integer_clock']
 t=d['issue18_tracking']; assert t['checkpoint_sha256']==model['checkpoint_sha256']
 assert t['model_optimizer_state_unchanged'] and t['training_updates']==0
 assert t['issue18_actual_feedback_verified'] and t['issue18_pre_action_phase_reconstruction_verified']
 assert [x['command'] for x in d['results']]==job['commands']
 for x in d['results']:
  assert x['num_envs']==job['num_envs'] and x['seconds']==20
 for command in job['commands']:
  f=read(Path(t['output'])/(command+'_feedback.json'))
  assert f['passed'] and f['last_action_max_error']==0 and f['applied_action_max_error']==0
  assert f['actually_applied_action_checks']==1000 and f['clock_reset_zero']
 assert sha(model['checkpoint'])==model['checkpoint_sha256']
 return {'output':str(path),'sha256':sha(path),'feedback_and_phase_passed':True}
def main():
 os.chdir(REPO); jobs=read(ROOT/'evaluation_job_templates.json');assert len(jobs)==45
 models={x['model']:x for x in read(ROOT/'cohort.json')['models']}
 state={'status':'checking_preflights','driver_pid':os.getpid(),'total':45,'completed':[],'started_at':time.time()};write(ROOT/'evaluation_progress.json',state)
 # The two actual 64-environment runs must finish successfully before formal jobs.
 for name in ['control6000_s1','control_int_s1']:
  p=ROOT/'preflight'/name/'results.json'; deadline=time.time()+900
  while not p.exists():
   assert time.time()<deadline,'preflight has not produced final results';time.sleep(10)
  template=dict(commands=jobs[0]['commands'],integer_clock=models[name]['integer_clock'],num_envs=64)
  audit(p,template,models[name])
 v=read(ROOT/'projection_validation.json');assert v['passed'] and v['models']==15
 for n,h in v['helper_source_sha256'].items():assert sha(REPO/'scripts/locomotion'/n)==h
 assert read(ROOT/'original_phase_reuse_audit.json')['passed']
 sources=[Path(__file__),*[(REPO/'scripts/locomotion'/n) for n in ['eval_issue18.py','issue18_policy.py','issue18_tracking.py','issue17_symmetry.py','issue16_mirror.py','issue16_tracking.py','issue15_tracking.py','issue13_tracking.py','eval_cmd.py']]]
 source_hashes={str(p):sha(p) for p in sources}
 frozen={'sources':source_hashes,'models':{m['checkpoint']:m['checkpoint_sha256'] for m in models.values()},'protocol':{str(ROOT/n):sha(ROOT/n) for n in ['classification_rules.json','cohort.json','original_reuse_manifest.json','original_phase_reuse_audit.json','evaluation_job_templates.json','projection_validation.json']},'zero_training':True,'formal_jobs':45}
 fp=ROOT/'formal_inputs_frozen.json'
 if fp.exists():assert read(fp)==frozen
 else:write(fp,frozen)
 write(ROOT/'preflight_audit.json',{'passed':True,'cases':2,'commands':12,'actual_feedback_error':0,'phase_reconstruction_exact':True,'time':time.time()})
 for job in jobs:
  for section in ['sources','models','protocol']:
   for p,h in frozen[section].items():assert sha(p)==h,('frozen input changed',p)
  model=models[job['model']];out=ROOT/'evaluation/clean'/(job['name']+'.json');folder=ROOT/'evaluation/clean/moments'/job['name'];log=ROOT/'evaluation/logs'/(job['name']+'.log');out.parent.mkdir(parents=True,exist_ok=True);log.parent.mkdir(parents=True,exist_ok=True)
  if not out.exists():
   assert not folder.exists(),('partial job requires explicit audit before rerun',str(folder))
   assert shutil.disk_usage(ROOT).free>350_000_000,'insufficient free disk'
   args=['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue18eval','python','-u','scripts/locomotion/eval_issue18.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1024','--seed',str(job['eval_seed']),'--checkpoint',model['checkpoint'],'--experiment_profile','robust08','--mass_randomization','bounded','--last_action_observation','requested','--issue5_condition','clean','--extended_commands','--command_filter',','.join(job['commands']),'--paired_resets','--issue18_mode','strict','--issue18_records',str(folder),'--output',str(out)]
   if not job['integer_clock']:args+=['--issue17_legacy_clock']
   with log.open('wb') as f:
    child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT)
    state.update(status='running',current=job['name'],child_pid=child.pid,log=str(log),updated_at=time.time());write(ROOT/'evaluation_progress.json',state)
    code=child.wait()
   assert code==0,(job['name'],code,str(log))
  receipt=audit(out,job,model);state['completed'].append(dict(name=job['name'],**receipt));state.update(status='between_jobs',updated_at=time.time());write(ROOT/'evaluation_progress.json',state)
 state.update(status='completed',finished_at=time.time(),command_records=270,initial_trials=276480);write(ROOT/'evaluation_progress.json',state)
if __name__=='__main__':
 try:main()
 except BaseException:
  p=ROOT/'evaluation_progress.json';d=read(p) if p.exists() else {};d.update(status='failed',error=traceback.format_exc(),updated_at=time.time());write(p,d);raise
