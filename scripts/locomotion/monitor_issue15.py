"""Wait on verified issue15 process handles without restarting the scientific run."""
import argparse,json,time
from status_issue15 import build_status
parser=argparse.ArgumentParser();parser.add_argument("--target",type=int,required=True);parser.add_argument("--max_wait_seconds",type=int,default=2400)
args=parser.parse_args();deadline=time.monotonic()+args.max_wait_seconds
while True:
 s=build_status();driver=s.get("formal_driver",{});live=driver.get("process",{}).get("matches_live_command",False)
 event={"done":s["completed_evaluation_jobs"],"target":args.target,"driver_pid":driver.get("pid"),"driver_live":live,"phase":s.get("current_phase"),"phase_process":s.get("current_phase_process"),"remaining_eval_seconds":s.get("estimated_remaining_evaluation_seconds"),"free_GiB":s["free_GiB"]}
 print("ISSUE15_MONITOR "+json.dumps(event),flush=True)
 if s["completed_evaluation_jobs"]>=args.target:
  print("ISSUE15_MONITOR_TARGET_REACHED",flush=True);break
 if not live:
  print("ISSUE15_MONITOR_DRIVER_STOPPED_INSPECT_ORIGINAL_HANDLE",flush=True);break
 if time.monotonic()>=deadline:
  print("ISSUE15_MONITOR_OBSERVATION_TIMEOUT_JOB_NOT_RESTARTED",flush=True);break
 time.sleep(min(60,max(0,deadline-time.monotonic())))
