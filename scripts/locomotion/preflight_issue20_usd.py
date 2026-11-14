"""Static and runtime USD gates plus no-policy 20-second standing test."""
from pathlib import Path
import subprocess,time,json,os
from issue20_common import ROOT,REPO,save,sha
from preflight_issue20 import args

def run(script,extra=(),name=None):
 log=ROOT/(name or Path(script).stem+'.log')
 with log.open('w') as f:
  child=subprocess.Popen(['python','-u',script,*extra],cwd=REPO,stdout=f,stderr=subprocess.STDOUT);save('usd_preflight_progress.json',{'state':script,'driver_pid':os.getpid(),'child_pid':child.pid});code=child.wait()
 assert code==0,(script,code)

def main():
 os.chdir(REPO);save('usd_preflight_driver.json',{'driver_pid':os.getpid()})
 while Path('/proc/118856').exists():time.sleep(3)
 assert json.loads((ROOT/'symmetric_usd_construction.json').read_text())['symmetric_sha256']==sha(ROOT/'strict_symmetric.usd')
 while not (ROOT/'symmetric_usd_construction.json').exists():
  assert Path('/proc/118071').exists(),'USD builder terminal without receipt';time.sleep(3)
 if not (ROOT/'strict_usd_preflight.json').exists() or not json.loads((ROOT/'strict_usd_preflight.json').read_text())['passed'] or json.loads((ROOT/'usd_inspection.json').read_text())['usd_sha256']!=sha(ROOT/'strict_symmetric.usd'):
  run('scripts/locomotion/inspect_issue20_usd.py');run('scripts/locomotion/audit_issue20_usd.py')
 assert json.loads((ROOT/'strict_usd_preflight.json').read_text())['passed']
 a,out=args('strict_zero_step_readback',usd=str(ROOT/'strict_symmetric.usd'));a[2]='scripts/locomotion/read_runtime_issue20.py';a[a.index('--num_envs')+1]='64';run(a[2],a[3:],name='runtime_readback.log');run('scripts/locomotion/audit_runtime_issue20.py')
 a,out=args('strict_no_policy_standing',usd=str(ROOT/'strict_symmetric.usd'));a[a.index('--command_filter')+1]='stand';a+=['--issue20_default'];run(a[2],a[3:],name='no_policy_standing.log');assert out.exists()
 import numpy as np
 z=np.load(out.parent/'moments/stand_yaw.npz');d=z['heading_delta_sum'][1]*180/np.pi
 save('no_policy_standing.json',{'passed':True,'num_envs':1024,'seconds':20,'mean_signed_drift_degrees':float(d.mean()),'mean_absolute_drift_degrees':float(abs(d).mean()),'maximum_absolute_drift_degrees':float(abs(d).max()),'failed_trials':int((z['first_failure_step']>=0).sum()),'policy_calls_replaced_by_zero_action':True,'raw_output':str(out)})
 save('usd_preflight_progress.json',{'state':'complete','driver_pid':os.getpid()});print('ISSUE20_USD_PREFLIGHT_COMPLETE',flush=True)
if __name__=='__main__':main()
