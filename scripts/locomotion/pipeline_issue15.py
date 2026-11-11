"""Wait for actual formal driver completion then finish all automatic postprocessing."""
import json,os,time
from pathlib import Path
from issue15_common import ROOT,REPO,run
def main():
 os.chdir(REPO);record=ROOT/"postprocess_driver.json"
 def state(s):record.write_text(json.dumps({"pid":os.getpid(),"state":s,"updated_unix":time.time()}))
 state("waiting_for_formal_driver")
 while True:
  p=ROOT/"formal_driver_process.json"
  if not p.exists():raise RuntimeError("No formal driver handle; inspect before starting/restarting work")
  d=json.loads(p.read_text());pid=d["pid"]
  try:cmd=Path(f"/proc/{pid}/cmdline").read_bytes();live=b"run_issue15.py" in cmd
  except FileNotFoundError:live=False
  if d["state"]=="complete":
   assert (ROOT/"formal_evaluation_complete").exists()
   if not live:break
  elif not live:raise RuntimeError("Formal driver stopped without completion; inspect original PID/log, do not restart blindly")
  state("waiting_for_formal_driver");time.sleep(20)
 for phase,script in [("analysis","analyze_issue15.py"),("conclusions","conclude_issue15.py"),("plots","plot_issue15.py"),("reconstruction","reconstruct_issue15.py"),("trajectory_plot","plot_issue15_trajectories.py"),("video_driver","render_issue15.py"),("video_audit","audit_issue15_videos.py")]:
  state(phase);run(["python","-u","scripts/locomotion/"+script],ROOT/(phase+".log"),phase)
 (ROOT/"automated_postprocessing_complete").write_text("All automatic stages complete; actual manual review and delivery/publication audit still required.\n")
 state("complete_manual_review_pending");print("ISSUE15_POSTPROCESS_COMPLETE_MANUAL_REVIEW_PENDING",flush=True)
if __name__=="__main__":main()
