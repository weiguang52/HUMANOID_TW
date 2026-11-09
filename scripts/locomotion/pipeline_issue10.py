"""Wait for stepA, then obey gate, render diagnostics and audit inputs."""
import subprocess,time,json,os
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue10_20261109')
os.chdir('/root/gpufree-data/projects/HUMANOID_TW')
while not (ROOT/'stepA_evaluation_complete').exists() or not (ROOT/'gate_decision.json').exists():
 try:os.kill(json.loads((ROOT/'runnerA_process.json').read_text())['pid'],0)
 except ProcessLookupError:raise RuntimeError('stepA driver stopped before gate')
 time.sleep(20)
for script,log in [('continue_issue10.py','runnerB.log'),('video_issue10.py','runner_video.log'),('audit_issue10_inputs.py','input_audit.log')]:
 with (ROOT/log).open('w') as f:
  p=subprocess.run(['bash','-c','source scripts/locomotion/remote_env.sh; exec python -u "$1"','issue10','scripts/locomotion/'+script],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(script)
(ROOT/'pipeline_complete').write_text('complete');print('PIPELINE_COMPLETE',flush=True)
