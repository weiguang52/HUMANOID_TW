"""Correct only ground-open lifecycle; preserve rejected reset-contaminated data."""
import os,json,time,subprocess,shutil
from pathlib import Path
from issue21_common import ROOT,REPO,sha,save
os.chdir(REPO)
while not (ROOT/'formal_evaluation_complete').exists():
 p=json.loads((ROOT/'evaluation_progress.json').read_text());assert Path('/proc/%d'%p['driver_pid']).exists(),'Original queue failed; inspect before resume';save('open_replay_correction_progress.json',{'state':'waiting_for_original_queue','driver_pid':os.getpid()});time.sleep(10)
dest=ROOT/'publication/rejected_ground_open_auto_resets';records=json.loads((ROOT/'cold_start_records.json').read_text());old=[x for x in records if x['mode']=='ground' and x['kind']=='open'];assert len(records)==155 and len(old)==40
if not dest.exists():
 dest.mkdir();(dest/'original_cold_start_records.json').write_text(json.dumps(records,indent=2)+'\n');shutil.copy2(ROOT/'evaluation_records.json',dest/'original_evaluation_records.json');shutil.copy2(ROOT/'formal_evaluation_complete',dest/'original_formal_marker.txt');inventory=[]
 for rec in old:
  p=Path(rec['output']);meta=json.loads(p.read_text());assert len(meta['records'])==1
  for source in [p,p.with_suffix('.log'),Path(meta['records'][0]['path'])]:
   assert source.is_relative_to(ROOT);rel=source.relative_to(ROOT);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);digest=sha(source);size=source.stat().st_size;shutil.move(str(source),str(target));assert sha(target)==digest;inventory.append({'original':str(source),'retained':str(target),'bytes':size,'sha256':digest})
 for ts in range(1,6):
  for n in [16,1]:shutil.copy2(ROOT/'evaluation'/('ground_s%d_n%d.json'%(ts,n)),dest/('original_ground_s%d_n%d.json'%(ts,n)))
 save('ground_open_lifecycle_correction.json',{'reason':'169 of340 ground-open trajectories triggered episode termination. Automatic resets contaminate the requested continuous200-step open-loop response. Re-evaluate the same40 conditions, same inputs/policies/physical parameters, evaluating termination conditions but suppressing only automatic-reset requests. No parameter scan or training.','rejected_completed_processes':40,'rejected_compute_seconds':sum(x['seconds'] for x in old),'retained_remotely':str(dest),'inventory':inventory,'source_closed_actions_unchanged':True,'thresholds_unchanged':True,'time':time.time()})
 (ROOT/'formal_evaluation_complete').unlink()
else:records=json.loads((dest/'original_cold_start_records.json').read_text());old=[x for x in records if x['mode']=='ground' and x['kind']=='open']
files={str(REPO/'scripts/locomotion'/n):sha(REPO/'scripts/locomotion'/n) for n in ['issue21_open_noreset.py','eval_issue21_open_noreset.py']};save('frozen_open_noreset_sources.json',{'files':files,'frozen_at':time.time()});f=ROOT/'frozen_evaluation_sources.json';frozen=json.loads(f.read_text())
for p,d in files.items():
 if p in frozen['files']:assert frozen['files'][p]==d
 else:frozen['files'][p]=d
frozen['open_lifecycle_extension']='frozen_open_noreset_sources.json';f.write_text(json.dumps(frozen,indent=2)+'\n')
done=json.loads((ROOT/'continuous_open_replay_records.json').read_text()) if (ROOT/'continuous_open_replay_records.json').exists() else []
for rec in old:
 if any(x['name']==rec['name'] for x in done):continue
 for p,d in frozen['files'].items():assert sha(p)==d,p
 ts=rec['training_seed'];n=rec['num_envs'];cmd=rec['command'];phase=rec['phase'];model=json.loads((ROOT/'cohort.json').read_text())[ts-1];out=Path(rec['output']);assert not out.exists();a=['python','-u','scripts/locomotion/eval_issue21_open_noreset.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs',str(n),'--seed','1','--checkpoint',model['checkpoint'],'--experiment_profile','robust08','--mass_randomization','bounded','--last_action_observation','requested','--issue5_condition','clean','--paired_resets','--seconds','20','--issue21_mode','ground','--training_seed',str(ts),'--issue21_kind','open','--issue21_phase',str(phase),'--issue21_command',cmd,'--output',str(out)];started=time.time()
 with out.with_suffix('.log').open('w') as log:
  child=subprocess.Popen(a,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('evaluation_progress.json',{'state':'continuous_ground_open_replay','job':rec['name'],'completed':len(done),'total_jobs':40,'core_processes_complete':155,'driver_pid':os.getpid(),'child_pid':child.pid,'started_at':started});assert child.wait()==0,rec['name']
 meta=json.loads(out.read_text());assert meta['diagnostic_lifecycle']['all_environment_episode_steps_at_end']==200 and meta['diagnostic_lifecycle']['one_continuous_physical_trajectory_per_env'];new=dict(rec,sha256=sha(out),seconds=time.time()-started,rejected_previous_sha256=rec['sha256'],continuous_open_lifecycle_corrected=True);done.append(new);save('continuous_open_replay_records.json',done);print('CONTINUOUS_GROUND_OPEN_DONE',len(done),rec['name'],flush=True)
byname={x['name']:x for x in done};active=[byname.get(x['name'],x) for x in records];assert len(active)==155;save('cold_start_records.json',active);logical=[]
for mode,ns in [('ground',[16,1]),('air',[16,1]),('perturb',[1024])]:
 for ts in range(1,6):
  for n in ns:
   group=[x for x in active if x['mode']==mode and x['training_seed']==ts and x['num_envs']==n and x['epsilon']!=0];parts=[json.loads(Path(x['output']).read_text()) for x in group];meta=parts[0].copy();meta['records']=[v for x in parts for v in x['records']];meta['source_subjobs']=[x['output'] for x in group];meta['all_fresh_processes']=True
   if mode=='ground':meta['all_open_replays_continuous200_no_automatic_resets']=True
   out=ROOT/'evaluation'/('%s_s%d_n%d.json'%(mode,ts,n));out.write_text(json.dumps(meta,indent=2)+'\n');logical.append({'name':out.stem,'output':str(out),'sha256':sha(out),'seconds':sum(x['seconds'] for x in group)})
save('evaluation_records.json',logical);save('evaluation_progress.json',{'state':'formal_diagnostics_complete','jobs':25,'active_cold_processes':155,'executed_cold_processes':195,'controls':5,'driver_pid':os.getpid()});(ROOT/'formal_evaluation_complete').write_text('25 logical groups,155 active cold cases,5 controls;40 rejected reset-contaminated open cases separately retained\n');save('open_replay_correction_progress.json',{'state':'complete','driver_pid':os.getpid(),'corrected_processes':40});print('ISSUE21_CONTINUOUS_OPEN_CORRECTION_COMPLETE40',flush=True)
