"""Run all35 unique lineage policies under the frozen clean protocol, plus original FT5."""
import json,os,shutil,subprocess
from pathlib import Path
from issue15_common import ROOT,REPO,BASE,COMMANDS,sha,run,evaluate_args
def verify_frozen():
 assert json.loads((ROOT/"smoke_audit.json").read_text())["passed"]
 frozen=json.loads((ROOT/"frozen_evaluation_sources.json").read_text())
 for p,h in frozen["sha256"].items():assert sha(p)==h,("Frozen source changed",p)
 for name,h in frozen["inputs"].items():assert sha(ROOT/name)==h,name
 assert subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()==BASE
 assert shutil.disk_usage(ROOT).free>2*1024**3,"Preserve user files; insufficient disk"
 assert not os.environ.get("TW_RESET_STD")
def audit_eval(output,m,seed,n=1024,legacy=False):
 import numpy as np
 d=json.loads(output.read_text());assert d["deterministic_policy"] and d["checkpoint"]==m["checkpoint"]
 commands=["stand","forward_01","forward_02","forward_03","yaw_positive","yaw_negative"] if legacy else COMMANDS
 assert [x["command"] for x in d["results"]]==commands
 assert not d["pushes"] and d["observation_noise"]==legacy
 assert d["observation_dims"]=={"policy":[101],"critic":[136 if m.get("critic_clock") else 134]}
 assert sha(m["checkpoint"])==m["sha256"]
 for row in d["results"]:assert row["num_envs"]==n and row["seconds"]==20 and row["seed"]==seed
 if not legacy:
  t=d["issue15_tracking"];assert t["model_optimizer_state_unchanged"] and t["training_updates"]==0
  p=t["physics_audit"];assert p["mass_equal_nominal"] and p["materials_equal_across_environments"] and p["action_class"]=="JointPositionAction"
  assert all(p[k]==1 for k in ["material_static_min","material_static_max","material_dynamic_min","material_dynamic_max","terrain_static","terrain_dynamic"])
  assert not any(p[k] for k in ["pushes","observation_noise","mass_event","material_event"])
  for cmd in COMMANDS:
   folder=Path(t["output"]);meta=json.loads((folder/(cmd+"_yaw.json")).read_text())
   assert meta["steps"]==1000 and meta["num_envs"]==1024
   with np.load(folder/(cmd+"_yaw.npz")) as z:
    assert z["heading_count"].shape==(2,n)
    if cmd.startswith("forward"):assert z["wz"].shape==z["heading_yaw"].shape==(1000,n)
 return {"passed":True,"eval_seed":seed,"checkpoint_sha256":m["sha256"],"commands":len(commands),"initial_trials":n*len(commands),"training_updates":0,"output_sha256":sha(output)}
def main():
 os.chdir(REPO);verify_frozen();process=ROOT/"formal_driver_process.json"
 process.write_text(json.dumps({"pid":os.getpid(),"state":"running"}))
 cohort=json.loads((ROOT/"cohort.json").read_text());state=ROOT/"formal_jobs.json";jobs=json.loads(state.read_text()) if state.exists() else {}
 for name,m in cohort["unique_evaluation_models"].items():
  for es in [1,2,3]:
   verify_frozen();out=ROOT/"clean"/(name+"_eval"+str(es)+".json");key="clean:"+name+":"+str(es)
   if not out.exists():
    assert not out.with_suffix(".log").exists(),"Inspect incomplete job before retry"
    run(evaluate_args(m,es,out),out.with_suffix(".log"),key)
   jobs[key]=audit_eval(out,m,es);state.write_text(json.dumps(jobs,indent=2))
 ft5=cohort["models"]["ft5_s1"]
 for es in [1,2,3]:
  verify_frozen();out=ROOT/"legacy"/("ft5_eval"+str(es)+".json");key="legacy:ft5:"+str(es)
  if not out.exists():
   assert not out.with_suffix(".log").exists()
   run(evaluate_args(ft5,es,out,num_envs=128,observer=False,legacy=True,plain=True),out.with_suffix(".log"),key)
  jobs[key]=audit_eval(out,ft5,es,n=128,legacy=True);state.write_text(json.dumps(jobs,indent=2))
 assert len(jobs)==108 and sum(x["initial_trials"] for x in jobs.values())==647424
 (ROOT/"formal_evaluation_complete").write_text("105 clean jobs=645120 trials +3 original FT5 jobs=2304 trials; zero training.\n")
 process.write_text(json.dumps({"pid":os.getpid(),"state":"complete"}));print("ISSUE15_FORMAL_EVALUATION_COMPLETE",flush=True)
if __name__=="__main__":main()
