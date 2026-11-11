"""Compare new clean observations with the exact committed issue13/14 first-five-command records."""
import json,subprocess
from pathlib import Path
import numpy as np
from issue15_common import ROOT,REPO,BASE,sha
_CACHE={}
def committed_manifest(bundle):
 if bundle not in _CACHE:
  path="validation_artifacts/"+bundle+"/sha256.json"
  _CACHE[bundle]=json.loads(subprocess.check_output(["git","show",BASE+":"+path]))
 return _CACHE[bundle]
def compare_job(model,eval_seed,new_output):
 node=model["node"];seed=model["training_seed"]
 assert node in ["control6000","treatment","clock"]
 bundle="stage1_issue14_20261111" if node=="clock" else "stage1_issue13_20261110"
 stem=("control" if node=="control6000" else node)+f"_s{seed}_eval{eval_seed}"
 archive=REPO/"validation_artifacts"/bundle;oldpath=archive/"clean"/(stem+".json");manifest=committed_manifest(bundle)
 refs={}
 def bind(p):
  rel=str(p.relative_to(archive));h=sha(p);assert manifest[rel]==h,("Historical reference changed",str(p));refs[str(p)]=h
 bind(oldpath);old=json.loads(oldpath.read_text());new=json.loads(Path(new_output).read_text())
 assert old["checkpoint"]==new["checkpoint"]==model["checkpoint"]
 assert old["issue13_tracking"]["checkpoint_sha256"]==model["sha256"]
 assert old["observation_dims"]==new["observation_dims"]
 assert old["issue13_tracking"]["physics_audit"]==new["issue15_tracking"]["physics_audit"]
 commands=[];newrows={r["command"]:r for r in new["results"]};exact_metrics=True;exact_arrays=True
 oldfolder=archive/"clean/moments"/stem;newfolder=Path(new["issue15_tracking"]["output"])
 for a in old["results"]:
  cmd=a["command"];b=newrows[cmd]
  assert all(a[k]==b[k] for k in ["command","target","seed","num_envs","seconds"])
  differences={k:float(b["metrics"][k]-v) for k,v in a["metrics"].items()}
  equal=a==b;exact_metrics &=equal;changed={}
  oldnpz=oldfolder/(cmd+".npz");bind(oldnpz)
  with np.load(oldnpz) as x,np.load(newfolder/(cmd+".npz")) as y:
   assert set(x.files)==set(y.files)
   for k in x.files:
    left,right=x[k],y[k];assert left.shape==right.shape and left.dtype==right.dtype
    if left.tobytes()!=right.tobytes():
     changed[k]={"different_elements":int(np.count_nonzero(left!=right))}
     if left.dtype.kind in "fiu":changed[k]["max_absolute_difference"]=float(np.max(abs(left.astype(np.float64)-right.astype(np.float64))))
  exact_arrays &=not bool(changed)
  commands.append({"command":cmd,"original_row_exactly_reproduced":equal,"metric_differences":differences,
   "original_minimum_height":a["minimum_base_height"],"new_minimum_height":b["minimum_base_height"],
   "original_falls":a["failed_trials"],"new_falls":b["failed_trials"],"original_moment_arrays_bitwise_equal":not bool(changed),"changed_arrays":changed})
 assert len(commands)==5
 return {"comparison_completed":True,"node":node,"training_seed":seed,"eval_seed":eval_seed,"same_checkpoint_sha256":model["sha256"],
  "original_job":str(oldpath),"new_job":str(new_output),"all_five_result_rows_exact":exact_metrics,"all_five_moment_arrays_bitwise_equal":exact_arrays,
  "commands":commands,"committed_old_input_sha256":refs}
def summarize(jobs):
 assert len(jobs)==45 and sum(len(x["commands"]) for x in jobs)==225
 return {"comparison_completed":True,"jobs":jobs,"job_count":45,"command_count":225,
 "all_original_metrics_exactly_reproduced":all(x["all_five_result_rows_exact"] for x in jobs),
 "all_original_moment_arrays_bitwise_equal":all(x["all_five_moment_arrays_bitwise_equal"] for x in jobs),
 "scope":"All control/treatment/clock first five clean commands from issue13/14. Newly appended stand has no old counterpart. This is consistency evidence, not a changed acceptance criterion."}
