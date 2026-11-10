"""Follow the actual fixed-study controller, then derive and render all deliverables."""
import json,os,time
from pathlib import Path
from issue12_common import ROOT,REPO,run

def live_study(pid):
    try:
        os.kill(pid,0)
        stat=Path(f"/proc/{pid}/stat").read_text().split(") ",1)[1]
        cmd=Path(f"/proc/{pid}/cmdline").read_bytes()
        return stat[0]!="Z" and b"run_issue12.py" in cmd
    except (ProcessLookupError,FileNotFoundError):
        return False

def main():
    os.chdir(REPO)
    driver=json.loads((ROOT/"formal_driver_process.json").read_text())
    status=ROOT/"postprocess_driver.json"
    status.write_text(json.dumps({"pid":os.getpid(),"state":"waiting_for_formal_study",
                                 "formal_driver_pid":driver["pid"]},indent=2))
    while not (ROOT/"evaluation_complete").exists():
        if (ROOT/"stop").exists():raise RuntimeError("Explicit stop marker")
        if not live_study(driver["pid"]):
            raise RuntimeError("Formal controller is terminal without complete evaluation; inspect, do not restart")
        time.sleep(20)
    assert (ROOT/"training_complete").exists()
    assert len(json.loads((ROOT/"final_models.json").read_text()))==10
    for script,phase in [("analyze_issue12.py","analysis"),("plot_issue12.py","plots"),
                         ("render_issue12.py","video_driver"),("audit_issue12_videos.py","video_audit")]:
        status.write_text(json.dumps({"pid":os.getpid(),"state":phase},indent=2))
        run(["python","-u","scripts/locomotion/"+script],ROOT/(phase+".log"),phase)
    (ROOT/"automated_postprocessing_complete").write_text("Analysis,plots,20videos,encoding+sampled frame checks; manual review still required.")
    status.write_text(json.dumps({"pid":os.getpid(),"state":"complete_manual_review_pending"},indent=2))
    print("ISSUE12_POSTPROCESS_COMPLETE_MANUAL_QA_PENDING",flush=True)
if __name__=="__main__":main()
