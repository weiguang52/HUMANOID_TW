"""Wait on actual issue14 process, then derive analysis/media and stop for manual review."""
import json,os,time
from pathlib import Path
from issue14_common import ROOT,REPO,run
def live(pid):
 try:
  p=Path(f"/proc/{pid}")
  return p.joinpath("stat").read_text().split(") ",1)[1][0]!="Z" and b"run_issue14.py" in p.joinpath("cmdline").read_bytes()
 except (FileNotFoundError,ProcessLookupError):return False
def main():
 os.chdir(REPO);driver=json.loads((ROOT/"formal_driver_process.json").read_text());status=ROOT/"postprocess_driver.json"
 status.write_text(json.dumps({"pid":os.getpid(),"state":"waiting_for_formal_training_and_evaluation","formal_driver_pid":driver["pid"]},indent=2))
 while not (ROOT/"formal_evaluation_complete").exists():
  if (ROOT/"stop").exists():raise RuntimeError("Explicit stop marker")
  if not live(driver["pid"]):raise RuntimeError("Actual formal driver terminal without completion; inspect budget-preserving recovery, do not restart")
  time.sleep(20)
 for script,phase in [("analyze_issue14.py","analysis"),("plot_issue14.py","plots"),("reconstruct_issue14.py","reconstruction"),("render_issue14.py","video_driver"),("audit_issue14_videos.py","video_audit")]:
  status.write_text(json.dumps({"pid":os.getpid(),"state":phase},indent=2));run(["python","-u","scripts/locomotion/"+script],ROOT/(phase+".log"),phase)
 (ROOT/"automated_postprocessing_complete").write_text("Three-group analysis,6figures,new median reconstruction,6colored clips and36sample checks done; manual review still pending.\n")
 status.write_text(json.dumps({"pid":os.getpid(),"state":"complete_manual_review_pending"},indent=2));print("ISSUE14_POSTPROCESS_COMPLETE_MANUAL_REVIEW_PENDING",flush=True)
if __name__=="__main__":main()
