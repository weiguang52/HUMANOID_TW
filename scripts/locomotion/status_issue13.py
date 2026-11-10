"""Read-only issue13 actual process and completion status."""
import json,os,time,shutil
from pathlib import Path
ROOT=Path("/root/gpufree-data/tw_stage1/issue13_20261110")
def live(pid):
 try:
  os.kill(pid,0);return Path(f"/proc/{pid}/stat").read_text().split(") ",1)[1][0]!="Z"
 except (ProcessLookupError,FileNotFoundError):return False
def main():
 p=json.loads((ROOT/"progress.json").read_text())
 d={"phase":p["phase"],"state":p["state"],"job_pid":p["pid"],"job_live_now":live(p["pid"]),
 "elapsed_seconds":round(time.time()-p["started_unix"]),"free_disk_GiB":round(shutil.disk_usage(ROOT).free/1024**3,2),
 "runtime_smoke_complete":(ROOT/"runtime_smoke_complete").exists(),"formal_evaluation_complete":(ROOT/"formal_evaluation_complete").exists()}
 if (ROOT/"formal_job_state.json").exists():
  a=json.loads((ROOT/"formal_job_state.json").read_text());d.update(formal_jobs_complete=len(a),commands_complete=sum(x["commands"] for x in a.values()),initial_trials_complete=sum(x["initial_trials"] for x in a.values()))
 if (ROOT/"formal_driver_process.json").exists():
  driver=json.loads((ROOT/"formal_driver_process.json").read_text());d["formal_driver"]={**driver,"live_now":live(driver["pid"])}
 print(json.dumps(d,indent=2))
if __name__=="__main__":main()
