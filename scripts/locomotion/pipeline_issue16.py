"""Follow the live issue16 queue, then execute raw analysis, zero-step readback and reconstruction."""
import json,os,time
from pathlib import Path
from issue16_common import ROOT,REPO,run,evaluate_args
from run_issue16 import verify_frozen
def main():
 os.chdir(REPO);pid=os.getpid()
 def state(value,**more):
  (ROOT/'postprocess_driver.json').write_text(json.dumps(dict(pid=pid,state=value,updated_unix=time.time(),**more))+'\n')
 state('waiting_for_formal_driver')
 try:
  while not (ROOT/'formal_evaluation_complete').exists():
   info=json.loads((ROOT/'formal_driver_process.json').read_text());p=Path('/proc')/str(info['pid'])
   assert p.exists() and (p/'stat').read_text().split()[2]!='Z','Formal driver terminal before completion; inspect, do not restart blindly'
   assert b'run_issue16.py' in (p/'cmdline').read_bytes(),'Formal driver PID identity changed'
   assert not (ROOT/'stop').exists()
   time.sleep(20)
  verify_frozen()
  for phase,script in [('analysis','analyze_issue16.py')]:
   state(phase);run(['python','-u','scripts/locomotion/'+script],ROOT/(phase+'.log'),phase)
  state('runtime_zero_step_readback')
  cohort=json.loads((ROOT/'cohort.json').read_text());args=evaluate_args(cohort['models']['control6000_s1'],1,ROOT/'runtime_readback/unused.json',num_envs=64,commands=['stand'],plain=True,observer=False)
  args[2]='scripts/locomotion/read_runtime_issue16.py'
  run(args,ROOT/'runtime_readback.log','runtime_zero_step_readback')
  assert json.loads((ROOT/'runtime_readback/metadata.json').read_text())['raw_readback_completed']
  state('original_state_reconstruction');run(['python','-u','scripts/locomotion/reconstruct_issue16.py'],ROOT/'reconstruction.log','original_state_reconstruction')
  state('analysis_runtime_and_reconstruction_complete')
  (ROOT/'initial_postprocessing_complete').write_text('Raw analysis, zero-step runtime readback and original source-state reconstruction completed; conclusions/media/manualreview/publication remain.\n')
  print('ISSUE16_INITIAL_POSTPROCESSING_COMPLETE',flush=True)
 except Exception as error:
  state('failed',error=repr(error));raise
if __name__=='__main__':main()
