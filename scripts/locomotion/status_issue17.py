"""Read-only status from actual process identities and observed progress."""
import json,re,time,shutil,statistics
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue17_20261113')
def read(n,default=None):
 p=ROOT/n;return json.loads(p.read_text()) if p.exists() else default

def live(pid,identity=None):
 p=Path('/proc')/str(pid)
 try:return p.exists() and (p/'stat').read_text().split()[2]!='Z' and (identity is None or identity.encode() in (p/'cmdline').read_bytes())
 except OSError:return False

def main():
 d=read('training_progress.json',{});records=read('training_records.json',[]);report=dict(formal_training_completed=len(records),formal_training_total=10,formal_updates_completed_in_finished_jobs=sum(r['updates'] for r in records),actual_training_driver_live=live(d.get('driver_pid'),'run_issue17_training.py'),formal_training_complete=(ROOT/'formal_training_complete').exists(),free_bytes=shutil.disk_usage(ROOT).free)
 if d.get('state')=='training':
  name=d['name'];p=ROOT/'training'/name/'console.log';s=re.sub(r'\x1b\[[0-9;]*m','',p.read_text(errors='replace')) if p.exists() else '';iteration=re.findall(r'Learning iteration\s+(\d+)/(\d+)',s);times=[float(x) for x in re.findall(r'Iteration time:\s+([0-9.]+)s',s)];loaded=read('training/'+name+'/loaded_state_audit.json',{})
  current=dict(name=name,child_pid=d['child_pid'],actual_child_live=live(d['child_pid'],'train.py'),last_logged_iteration=int(iteration[-1][0]) if iteration else None)
  if iteration and 'iteration' in loaded:
   count=min(1500,max(0,int(iteration[-1][0])-loaded['iteration']+1));current.update(observed_updates=count,required_updates=1500,recent_median_update_seconds=statistics.median(times[-10:]) if times else None)
   if times:current['current_job_eta_seconds']=(1500-count)*statistics.median(times[-10:])
  report['current_training_job']=current
 else:report['training_state']=d.get('state')
 report['formal_evaluation_completed']=len(read('evaluation_records.json',[]));report['formal_evaluation_total']=90
 for name in ['postprocess','statistics','media','figure']:
  state=read(name+'_progress.json')
  if state:report[name]=dict(state=state['state'],driver_pid=state['driver_pid'],actual_process_live=live(state['driver_pid']))
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
