"""Resolve the complete issue15 lineage without changing or training policies."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path("/root/gpufree-data/tw_stage1/issue15_20261111")
REPO=Path("/root/gpufree-data/projects/HUMANOID_TW")
BASE="60a8cc25ccce3a8ae22fcf7e5dd6f8d53db7613b"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(bundle,name):return json.loads((REPO/"validation_artifacts"/bundle/name).read_text())
def main():
 assert subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()==BASE
 ROOT.mkdir(exist_ok=True)
 old=load("stage1_issues_20261108","checkpoint_audit.json")
 models={};nodes={}
 def add(node,seed,path,h=None,run=None,**extra):
  path=Path(path);assert path.exists(),str(path)
  h2=sha(path);assert h is None or h2==h,(node,seed)
  key=node+"_s"+str(seed)
  models[key]={"node":node,"training_seed":seed,"checkpoint":str(path),"sha256":h2,"run_dir":str(run or path.parent),**extra}
  nodes.setdefault(node,[]).append(key)
 add("ft5",1,REPO/"validation_artifacts/tw_stage1_ft5_20261107/model_5997.pt",run=REPO/"logs/rsl_rl/tw_stage1/2026-11-07_17-12-49_tw_stage1_s1_ft5")
 for name in ["robust05","robust08"]:
  m=old[name];add(name,1,m["checkpoint"],m["sha256"])
 for k,m in load("stage1_issue5_20261109","training_state.json").items():
  if k.startswith("strengthen_"):add("strengthen",m["seed"],m["checkpoint"],run=m["run_dir"],parent_sha256=m["parent_sha256"])
 for k,m in load("stage1_issue6_20261109","training_state.json").items():
  add("bounded",m["seed"],m["checkpoint"],run=m["run_dir"],parent_sha256=m["parent_sha256"])
 for k,m in load("stage1_issue10_20261109","input_models.json").items():
  if k.startswith("requested_") and m["target_updates"] in [1500,3000,4500]:
   add("requested"+str(m["target_updates"]),m["training_seed"],m["checkpoint"],m["sha256"],requested_updates=m["target_updates"],actual_logged_updates=m["actual_logged_updates"])
 for k,m in load("stage1_issue12_20261110","final_models.json").items():
  add("control6000" if k.startswith("control") else "treatment",m["training_seed"],m["checkpoint"],m["sha256"],run=m["run_dir"],parent_sha256=m["parent_sha256"])
 for k,m in load("stage1_issue14_20261111","final_models.json").items():
  add("clock",m["training_seed"],m["checkpoint"],m["sha256"],run=m["run_dir"],parent_sha256=m["parent_sha256"],critic_clock=True)
 unique={};byhash={}
 for k,m in models.items():
  if m["sha256"] not in byhash:byhash[m["sha256"]]=k;unique[k]=m
  m["evaluation_model"]=byhash[m["sha256"]]
 assert len(models)==37 and len(unique)==35,(len(models),len(unique))
 for seed in [1,2]:assert models[f"bounded_s{seed}"]["sha256"]==models[f"requested1500_s{seed}"]["sha256"]
 data={"base_commit":BASE,"nodes":nodes,"models":models,"unique_evaluation_models":unique,"lineage_entries":37,"unique_checkpoints":35,
 "alias_note":"The two issue6 bounded models are exactly requested1500 seeds1/2, not additional independent policies.",
 "display_order":["ft5","robust05","robust08","strengthen","bounded","requested1500","requested3000","requested4500","control6000","treatment","clock"],
 "comparison_edges":[["ft5","robust05"],["robust05","robust08"],["robust08","strengthen"],["strengthen","bounded"],["bounded","requested1500"],["requested1500","requested3000"],["requested3000","requested4500"],["requested4500","control6000"],["control6000","treatment"],["treatment","clock"]],
 "ancestry_note":"Display edges are diagnostic comparisons, NOT all actual parent-child transitions. Strengthen/bounded restart from robust05; control/treatment/clock are sibling continuations of requested4500. FT5->robust05 includes unshown yaw-training intermediate ancestors; configuration audit must expose this."}
 (ROOT/"cohort.json").write_text(json.dumps(data,indent=2)+"\n")
 print("ISSUE15_COHORT",len(models),len(unique),len(unique)*3)
if __name__=="__main__":main()
