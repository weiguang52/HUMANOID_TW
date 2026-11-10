"""Follow the actual frozen evaluator then derive exact-state media; stop for actual manual review."""
import json,os,time,sys
from pathlib import Path
from issue13_common import ROOT,REPO,run
def live(pid):
 try:
  os.kill(pid,0);p=Path(f"/proc/{pid}")
  return p.joinpath("stat").read_text().split(") ",1)[1][0]!="Z" and b"run_issue13.py" in p.joinpath("cmdline").read_bytes()
 except (ProcessLookupError,FileNotFoundError):return False
def main():
 os.chdir(REPO);formal=json.loads((ROOT/"formal_driver_process.json").read_text());status=ROOT/"postprocess_driver.json"
 status.write_text(json.dumps({"pid":os.getpid(),"state":"waiting_for_formal_evaluation","formal_driver_pid":formal["pid"]},indent=2))
 while not (ROOT/"formal_evaluation_complete").exists():
  if (ROOT/"stop").exists():raise RuntimeError("Explicit stop marker")
  if not live(formal["pid"]):raise RuntimeError("Formal driver terminal without completion; inspect actual state, do not restart")
  time.sleep(20)
 for script,phase in [("analyze_issue13.py","analysis"),("plot_issue13.py","plots"),("reconstruct_issue13.py","reconstruction"),("render_issue13.py","video_driver"),("audit_issue13_videos.py","video_audit")]:
  if "--resume-media" in sys.argv and phase in {"analysis","plots","reconstruction"}:
   assert (ROOT/(phase+"_complete")).exists()
   continue
  status.write_text(json.dumps({"pid":os.getpid(),"state":phase},indent=2));run(["python","-u","scripts/locomotion/"+script],ROOT/(phase+".log"),phase)
 (ROOT/"automated_postprocessing_complete").write_text("Frozen analysis,6figures,2exact median reconstructions,4colored replays,encoding/sample checks complete; manual review still required.\n")
 status.write_text(json.dumps({"pid":os.getpid(),"state":"complete_manual_review_pending"},indent=2))
 print("ISSUE13_POSTPROCESS_COMPLETE_MANUAL_REVIEW_PENDING",flush=True)
if __name__=="__main__":main()
