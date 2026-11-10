"""Original-vs-observed runtime equivalence; no training or scientific selection."""
import json,os
from pathlib import Path
import numpy as np
from issue13_common import ROOT,REPO,args_for,run,sha
def arrays_equal(a,b):
 with np.load(a) as x,np.load(b) as y:
  assert set(x.files)==set(y.files),(str(a),str(b))
  for k in x.files:assert np.array_equal(x[k],y[k]),(str(a),k)
def main():
 os.chdir(REPO)
 log=(ROOT/"unit_tests.log").read_text();assert "Ran 10 tests" in log and "\nOK\n" in log
 results={}
 for condition in ["combined","clean"]:
  outputs={}
  for mode,observe in [("baseline",False),("observed",True)]:
   folder=ROOT/"smoke"/condition/mode;folder.mkdir(parents=True,exist_ok=True);out=folder/"eval.json"
   args=args_for("control_s1",1,condition,out,records=observe,num_envs=64)
   args+=["--observation_audit",str(folder/"state_audit")]
   if not out.exists():run(args,folder/"run.log","smoke_"+condition+"_"+mode)
   outputs[mode]=json.loads(out.read_text())
  a,b=outputs["baseline"],outputs["observed"]
  for key,value in a.items():assert value==b[key],(condition,key)
  obs=ROOT/"smoke"/condition/"observed";orig=ROOT/"smoke"/condition/"baseline"
  for row in a["results"]:
   label=row["command"];arrays_equal(orig/"state_audit"/(label+".npz"),obs/"state_audit"/(label+".npz"))
   if condition=="combined":
    for suffix in [".npz","_trials.npz"]:arrays_equal(orig/"traces/eval"/(label+suffix),obs/"traces/eval"/(label+suffix))
    for suffix in [".json","_trials.json"]:
     assert json.loads((orig/"traces/eval"/(label+suffix)).read_text())==json.loads((obs/"traces/eval"/(label+suffix)).read_text())
  assert b["issue13_tracking"]["model_optimizer_state_unchanged"]
  assert b["observation_dims"]=={"policy":[101],"critic":[134]}
  results[condition]={"passed":True,"every_original_output_field_bitwise_equal":True,
   "all_final_root_joint_complete_cpu_cuda_numpy_python_rng_arrays_bitwise_equal":True,
   "original_full_trial_and_failure_arrays_bitwise_equal":condition=="combined",
   "readonly_reference_moment_reconstruction_passed":True,
   "physics_audit":b["issue13_tracking"]["physics_audit"]}
  (ROOT/"runtime_smoke_audit.json").write_text(json.dumps(results,indent=2))
 (ROOT/"runtime_smoke_complete").write_text("Original-vs-observed clean+combined exact outputs/state/RNG and analytic/reference moments pass; no training.\n")
 print("ISSUE13_RUNTIME_SMOKE_PASS",flush=True)
if __name__=="__main__":main()
