"""Read actual issue14 process handles and compact formal progress."""
import json,os,re,time
from pathlib import Path
from issue14_common import ROOT
def live(pid):
 try:
  p=Path(f"/proc/{pid}");return p.exists() and p.joinpath("stat").read_text().split(") ",1)[1][0]!="Z"
 except (FileNotFoundError,ProcessLookupError):return False
def main():
 d=json.loads((ROOT/"progress.json").read_text()) if (ROOT/"progress.json").exists() else {}
 driver=json.loads((ROOT/"formal_driver_process.json").read_text()) if (ROOT/"formal_driver_process.json").exists() else {}
 out={"phase":d.get("phase"),"child_pid":d.get("pid"),"child_live":live(d.get("pid",0)),"driver_pid":driver.get("pid"),"driver_live":live(driver.get("pid",0)),"driver_state":driver.get("state"),"elapsed_current_seconds":time.time()-d.get("started_unix",time.time())}
 if d.get("log") and Path(d["log"]).exists():
  with Path(d["log"]).open("rb") as f:f.seek(max(0,f.seek(0,2)-24000));s=f.read().decode(errors="replace")
  s=re.sub(r"\x1b\[[0-9;?]*[A-Za-z]","",s);iters=re.findall(r"Learning iteration\s+(\d+)/(\d+)",s)
  if iters:
   out["last_iteration"]=int(iters[-1][0]);out["target_iteration_exclusive"]=int(iters[-1][1]);out["updates_in_current_run"]=out["last_iteration"]-(out["target_iteration_exclusive"]-1500)+1
  for pattern,key in [(r"ETA:\s+([^\n]+)","logged_ETA"),(r"Iteration time:\s+([^\n]+)","last_iteration_time")]:
   values=re.findall(pattern,s)
   if values:out[key]=values[-1].strip()
 for filename,key in [("formal_training_state.json","finished_models"),("training_audits.json","audited_models"),("formal_job_state.json","finished_evaluations")]:
  out[key]=len(json.loads((ROOT/filename).read_text())) if (ROOT/filename).exists() else 0
 out["training_complete"]=(ROOT/"training_complete").exists();out["evaluation_complete"]=(ROOT/"formal_evaluation_complete").exists()
 print(json.dumps(out,indent=2))
if __name__=="__main__":main()
