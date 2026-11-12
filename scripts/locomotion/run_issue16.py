"""Run the frozen issue16 zero-training mirror and fixed-default-PD evaluation budget."""
import json,os,time,math
from pathlib import Path
from issue16_common import ROOT,REPO,COMMANDS,run,evaluate_args,sha

def build_jobs(cohort):
 jobs=[]
 for node in ['robust05','requested4500','control6000']:
  for name in cohort['nodes'][node]:
   for seed in [1,2,3]:
    out=ROOT/'mirror'/f'{name}_eval{seed}.json'
    jobs.append(dict(name=f'mirror_{name}_eval{seed}',model=name,seed=seed,mode='mirror',commands=COMMANDS,output=str(out),initial_trials=6144))
 out=ROOT/'no_policy/default_eval1.json'
 jobs.append(dict(name='default_PD_stand_eval1',model='control6000_s1',seed=1,mode='default',commands=['stand'],output=str(out),initial_trials=1024))
 return jobs

def verify_frozen():
 frozen=json.loads((ROOT/'frozen_evaluation_sources.json').read_text())
 assert os.environ.get('TW_RESET_STD') is None
 for path,h in frozen['files'].items():assert sha(path)==h,('Frozen input/source changed',path)
 assert __import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip()==frozen['base_commit']

def audit_job(job,cohort):
 data=json.loads(Path(job['output']).read_text());model=cohort['models'][job['model']]
 assert data['checkpoint']==model['checkpoint'] and data['deterministic_policy'] and data['paired_resets']
 assert data['seed']==job['seed'] if 'seed' in data else True
 assert data['observation_dims']=={'policy':[101],'critic':[134]}
 assert [r['command'] for r in data['results']]==job['commands']
 for row in data['results']:
  assert row['seed']==job['seed'] and row['num_envs']==1024 and row['seconds']==20 and 0<=row['failed_trials']<=1024
  assert all(math.isfinite(v) for v in row['metrics'].values())
 meta=data['issue16_tracking'];intervention=data['issue16_intervention']
 assert meta['mode']==job['mode'] and meta['training_updates']==0 and meta['model_optimizer_state_unchanged']
 assert meta['checkpoint_sha256']==sha(model['checkpoint'])==model['sha256']
 assert intervention['mode']==job['mode'] and intervention['calls']==1000*len(job['commands'])
 assert intervention['gait_reflection'].startswith('Global scalar clock')
 if job['mode']=='default':assert meta['default_target_error_max']==0
 physics=meta['physics_audit']
 assert physics['mass_equal_nominal'] and physics['materials_equal_across_environments']
 assert all(not physics[k] for k in ['pushes','observation_noise','mass_event','material_event'])
 for key in ['material_static_min','material_static_max','material_dynamic_min','material_dynamic_max','terrain_static','terrain_dynamic']:assert physics[key]==1,key
 assert data['issue16_source']['original_sha256']==sha(REPO/'scripts/locomotion/eval_cmd.py')
 return dict(passed=True,name=job['name'],mode=job['mode'],initial_trials=job['initial_trials'],commands=len(job['commands']),training_updates=0,output_sha256=sha(job['output']))

def main():
 os.chdir(REPO)
 assert json.loads((ROOT/'smoke_audit.json').read_text())['passed']
 cohort=json.loads((ROOT/'cohort.json').read_text());jobs=json.loads((ROOT/'formal_jobs.json').read_text());assert jobs==build_jobs(cohort)
 assert len(jobs)==34 and sum(j['initial_trials'] for j in jobs)==203776
 process=ROOT/'formal_driver_process.json'
 if process.exists():
  old=json.loads(process.read_text());p=Path('/proc')/str(old['pid'])
  if p.exists() and (p/'stat').read_text().split()[2]!='Z' and old['pid']!=os.getpid():raise RuntimeError('A prior formal driver is still live')
 process.write_text(json.dumps(dict(pid=os.getpid(),state='running',started_unix=time.time())))
 receipts=[]
 for i,job in enumerate(jobs):
  verify_frozen()
  assert __import__('shutil').disk_usage(ROOT).free>1024**3,'Less than1GiB free; inspect without deleting original data'
  p=Path(job['output'])
  if not p.exists():
   args=evaluate_args(cohort['models'][job['model']],job['seed'],p,commands=job['commands'],mode=job['mode'])
   run(args,p.with_suffix('.log'),job['name'])
  receipts.append(audit_job(job,cohort))
  (ROOT/'formal_job_audits.json').write_text(json.dumps(receipts,indent=2)+'\n')
  print('FORMAL_COMPLETED',i+1,'of',len(jobs),flush=True)
 verify_frozen();assert len(receipts)==34
 (ROOT/'formal_evaluation_complete').write_text('34formal jobs203776initial trials; zero training\n')
 process.write_text(json.dumps(dict(pid=os.getpid(),state='complete',finished_unix=time.time(),jobs=34)))
 print('ISSUE16_FORMAL_COMPLETE',flush=True)
if __name__=='__main__':main()
