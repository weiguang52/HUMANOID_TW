"""Execute exactly the prospectively frozen issue13 analysis-only formal cohort."""
import json,os,time,subprocess
from pathlib import Path
import numpy as np
from issue13_common import ROOT,REPO,BASE,CLEAN,FORWARD,args_for,run,sha
def array_equal(a,b):
 with np.load(a) as x,np.load(b) as y:
  assert set(x.files)==set(y.files)
  for key in x.files:assert np.array_equal(x[key],y[key]),(str(a),key)
def verify_frozen():
 assert subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()==BASE
 frozen=json.loads((ROOT/"frozen_sources.json").read_text())
 for name,h in frozen["sha256"].items():assert sha(name)==h,name
 assert sha(ROOT/"protocol_plan.md")==json.loads((ROOT/"protocol_plan_sha256.json").read_text())["sha256"]
 for m in json.loads((ROOT/"input_models.json").read_text()).values():assert sha(m["checkpoint"])==m["sha256"]
def audit(name,seed,condition,out):
 models=json.loads((ROOT/"input_models.json").read_text());m=models[name];d=json.loads(Path(out).read_text())
 assert d["checkpoint"]==m["checkpoint"] and d["deterministic_policy"] and d["paired_resets"]
 assert d["observation_dims"]=={"policy":[101],"critic":[134]}
 track=d["issue13_tracking"];assert track["checkpoint_sha256"]==m["sha256"] and track["training_updates"]==0 and track["model_optimizer_state_unchanged"]
 assert [x["command"] for x in d["results"]]==(CLEAN if condition=="clean" else FORWARD)
 for row in d["results"]:
  assert row["num_envs"]==1024 and row["seconds"]==20 and row["seed"]==seed
  assert (Path(track["output"])/(row["command"]+".npz")).exists()
 result={"passed":True,"model":name,"eval_seed":seed,"condition":condition,"training_updates":0,
  "commands":len(d["results"]),"initial_trials":len(d["results"])*1024,"model_optimizer_unchanged":True,
  "independent_reference_moments_checked":all(x["reference_moment_reconstruction_passed"] for x in track["commands"])}
 if condition=="clean":
  p=track["physics_audit"]
  assert p["mass_equal_nominal"] and p["materials_equal_across_environments"] and p["action_class"]=="JointPositionAction"
  assert not any(p[k] for k in ["pushes","observation_noise","mass_event","material_event"])
  result["actual_nominal_no_perturbations_verified"]=True
 else:
  old=REPO/"validation_artifacts/stage1_issue12_20261110"
  before=json.loads((old/f"{name}_eval{seed}.json").read_text())
  for key,value in before.items():assert d[key]==value,("Old issue12 output changed",name,seed,key)
  folder=Path(out).parent/"traces"/Path(out).stem
  for command in FORWARD:
   for suffix in [".npz","_trials.npz"]:array_equal(folder/(command+suffix),old/"traces"/f"{name}_eval{seed}"/(command+suffix))
   for suffix in [".json","_trials.json"]:
    assert json.loads((folder/(command+suffix)).read_text())==json.loads((old/"traces"/f"{name}_eval{seed}"/(command+suffix)).read_text())
  result["every_original_issue12_field_and_all_trial_failure_arrays_exactly_reproduced"]=True
 return result
def main():
 os.chdir(REPO)
 assert (ROOT/"runtime_smoke_complete").exists()
 verify_frozen()
 state_path=ROOT/"formal_job_state.json";state=json.loads(state_path.read_text()) if state_path.exists() else {}
 process={"pid":os.getpid(),"state":"running","training_updates":0,"started_unix":time.time()}
 (ROOT/"formal_driver_process.json").write_text(json.dumps(process,indent=2))
 for condition in ["clean","combined"]:
  for seed in range(1,6):
   for group in ["control","treatment"]:
    name=f"{group}_s{seed}"
    for es in [1,2,3]:
     key=f"{condition}_{name}_eval{es}";out=ROOT/condition/f"{name}_eval{es}.json";out.parent.mkdir(exist_ok=True)
     verify_frozen()
     if key not in state:
      assert not out.exists(),("Unaccounted prior result; inspect rather than blindly repeating",str(out))
      run(args_for(name,es,condition,out),out.with_suffix(".log"),key)
     result=audit(name,es,condition,out);state[key]=result;state_path.write_text(json.dumps(state,indent=2))
 assert len(state)==60 and sum(x["commands"] for x in state.values())==240 and sum(x["initial_trials"] for x in state.values())==245760
 verify_frozen()
 (ROOT/"formal_evaluation_complete").write_text("All60analysis-only jobs,240commands,245760initial trials; original issue12 combined outputs exactly reproduced; no training.\n")
 process.update(state="complete",finished_unix=time.time());(ROOT/"formal_driver_process.json").write_text(json.dumps(process,indent=2))
 print("ISSUE13_FORMAL_EVALUATION_COMPLETE",flush=True)
if __name__=="__main__":main()
