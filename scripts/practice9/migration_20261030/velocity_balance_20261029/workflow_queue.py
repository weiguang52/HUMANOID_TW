from pathlib import Path
import subprocess,time,json,sys,traceback,os
b=Path(__file__).parent;states=[b,b.parent/'velocity_balance_seed123_20261029'];py=sys.executable
def alive(name):return subprocess.run(['tmux','has-session','-t',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
def start(name,cmd):subprocess.run(['tmux','new-session','-d','-s',name,cmd],check=True)
def wait_worker(name,path,complete='completed'):
 while True:
  status=path.read_text().strip() if path.exists() else ''
  if status=='failed':raise RuntimeError(str(path)+' failed')
  live=alive(name)
  if status==complete and not live:return
  if not live:raise RuntimeError(name+' handle missing status='+status)
  time.sleep(10)
try:
 (b/'queue_status').write_text('waiting_live_training')
 for s,name in zip(states,['velocity_balance42','velocity_balance123']):wait_worker(name,s/'actual_formal/status','trained')
 for s in states:subprocess.run([py,str(s/'audit.py'),'formal'],check=True,stdout=open(s/'formal_audit.log','w'),stderr=subprocess.STDOUT)
 subprocess.run([py,str(b/'audit_training_completion.py')],check=True)
 for seed,gpu in [(42,0),(123,1)]:start('velocity_balance_eval'+str(seed),'bash '+str(b/'run_evaluation_seed.sh')+' '+str(seed)+' '+str(gpu))
 (b/'queue_status').write_text('both_candidate65_running')
 for s,seed in zip(states,[42,123]):wait_worker('velocity_balance_eval'+str(seed),s/'evaluation_status')
 (b/'queue_status').write_text('numeric_outputs_complete_pending_full_audit')
except Exception:
 traceback.print_exc();(b/'queue_status').write_text('failed_requires_inspection');raise
