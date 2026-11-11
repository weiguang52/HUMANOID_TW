"""Read-only issue15 process/elapsed/remaining-evaluation status; not a completion certificate."""
import json,os,time,shutil
from pathlib import Path
from issue15_common import ROOT
def read(name):
 for attempt in range(3):
  try:return json.loads((ROOT/name).read_text())
  except (FileNotFoundError,json.JSONDecodeError):
   if attempt==2:return None
   time.sleep(.1)
def process(pid,needle):
 try:
  path=Path("/proc")/str(pid);cmd=(path/"cmdline").read_bytes()
  rest=(path/"stat").read_text().rsplit(")",1)[1].split()
  elapsed=float(Path("/proc/uptime").read_text().split()[0])-int(rest[19])/os.sysconf("SC_CLK_TCK")
  return {"pid":pid,"matches_live_command":needle.encode() in cmd,"elapsed_seconds":round(elapsed)}
 except (FileNotFoundError,ProcessLookupError):return {"pid":pid,"matches_live_command":False}
def build_status():
 jobs=read("formal_jobs.json") or {};driver=read("formal_driver_process.json");progress=read("progress.json");post=read("postprocess_driver.json")
 result={"completed_evaluation_jobs":len(jobs),"total_evaluation_jobs":108,"formal_training_updates":0,"free_GiB":round(shutil.disk_usage(ROOT).free/1024**3,2)}
 if driver:
  result["formal_driver"]=dict(driver,process=process(driver["pid"],"run_issue15.py"))
 if progress:
  result["current_phase"]=progress["phase"];result["current_phase_process"]=process(progress["pid"],"python")
 if post:result["postprocessing"]=dict(post,process=process(post["pid"],"pipeline_issue15.py"))
 if driver and driver["state"]=="running" and result["formal_driver"]["process"]["matches_live_command"] and 0<len(jobs)<108:
  elapsed=result["formal_driver"]["process"]["elapsed_seconds"];active=max(0,time.time()-progress["started_unix"]) if progress else 0
  average=max(1,(elapsed-active)/len(jobs));remaining=max(0,(108-len(jobs))*average-active)
  result["estimated_remaining_evaluation_seconds"]=round(remaining)
  result["eta_scope"]="Evaluation only, extrapolated from completed jobs; excludes analysis, media/manual review, archive and publication."
 elif len(jobs)==108:result["estimated_remaining_evaluation_seconds"]=0
 return result
def main():print(json.dumps(build_status(),indent=2))
if __name__=="__main__":main()
