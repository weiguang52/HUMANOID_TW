"""Compact read-only actual-process status for the fixed remote study."""
import csv,json,os,time,shutil
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue12_20261110')
def live(pid):
 try:
  os.kill(pid,0)
  return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]!='Z'
 except (ProcessLookupError,FileNotFoundError):return False
p=json.loads((ROOT/'progress.json').read_text())
s={'phase':p['phase'],'state':p['state'],'job_pid':p['pid'],'job_live_now':live(p['pid']),
   'elapsed_seconds':round(time.time()-p['started_unix']),
   'free_disk_GiB':round(shutil.disk_usage(ROOT).free/1024**3,2)}
if (ROOT/'formal_driver_process.json').exists():
 d=json.loads((ROOT/'formal_driver_process.json').read_text());s['driver']={**d,'live_now':live(d['pid'])}
for name,field in [('calibration_state.json','calibrations_complete'),('formal_training_state.json','training_runs_complete')]:
 if (ROOT/name).exists():s[field]=len(json.loads((ROOT/name).read_text()))
args=p['args']
if '--issue12_audit_dir' in args:
 audit=Path(args[args.index('--issue12_audit_dir')+1]);path=audit/'reward_contributions.csv'
 if path.exists():
  rows=list(csv.DictReader(path.open()));s['monitor_rows']=len(rows)
  if rows:s['last_monitor_row']=rows[-1]
if (ROOT/'final_models.json').exists():
 models=json.loads((ROOT/'final_models.json').read_text());s['evaluation_jobs_complete']=sum((ROOT/f'{name}_eval{seed}.json').exists() for name in models for seed in [1,2,3])
print(json.dumps(s,indent=2))
