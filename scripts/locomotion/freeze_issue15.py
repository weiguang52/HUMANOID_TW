"""Freeze zero-training evaluation scope, all tracked behavior sources and actual model/config hashes."""
import json,os,subprocess,time
from pathlib import Path
import yaml
from issue15_common import ROOT,REPO,BASE,sha
def diff(a,b,path=""):
 if isinstance(a,dict) and isinstance(b,dict):
  for k in sorted(set(a)|set(b)):
   if k not in a or k not in b:yield [path+"/"+str(k),a.get(k),b.get(k)]
   else:yield from diff(a[k],b[k],path+"/"+str(k))
 elif a!=b:yield [path,a,b]
def main():
 os.chdir(REPO);assert json.loads((ROOT/"smoke_audit.json").read_text())["passed"]
 log=(ROOT/"unit_tests.log").read_text();assert "Ran 9 tests" in log and log.rstrip().endswith("OK")
 assert not subprocess.check_output(["git","diff","--name-only"]).strip(),"Tracked production/configuration changes forbidden"
 assert subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()==BASE
 cohort=json.loads((ROOT/"cohort.json").read_text());configs={};audit={}
 for key,m in cohort["models"].items():
  assert sha(m["checkpoint"])==m["sha256"]
  configs[key]={};audit[key]={"checkpoint_sha256":m["sha256"],"run_dir":m["run_dir"],"config_files":{}}
  for kind in ["env","agent"]:
   p=Path(m["run_dir"])/"params"/(kind+".yaml");assert p.exists(),str(p)
   configs[key][kind]=yaml.load(p.read_text(),Loader=yaml.BaseLoader)
   audit[key]["config_files"][kind]={"path":str(p),"sha256":sha(p)}
  ag=configs[key]["agent"];audit[key]["actual_resume_settings"]={k:ag.get(k) for k in ["seed","resume","load_run","load_checkpoint","max_iterations","run_name"]}
 edges=[]
 for before,after in cohort["comparison_edges"]:
  left={cohort["models"][k]["training_seed"]:k for k in cohort["nodes"][before]};right={cohort["models"][k]["training_seed"]:k for k in cohort["nodes"][after]}
  for seed in sorted(set(left)&set(right)):
   a,b=left[seed],right[seed];changes={}
   for kind in ["env","agent"]:changes[kind]=list(diff(configs[a][kind],configs[b][kind]))
   edges.append({"before":a,"after":b,"same_checkpoint":cohort["models"][a]["sha256"]==cohort["models"][b]["sha256"],"all_saved_config_differences":changes})
 result={"models":audit,"diagnostic_edge_config_differences":edges,"actual_ancestry_warning":cohort["ancestry_note"],"not_a_causal_identification_claim":True}
 (ROOT/"configuration_lineage_audit.json").write_text(json.dumps(result,indent=2)+"\n")
 tracked=subprocess.check_output(["git","ls-files","scripts/locomotion","source/unitree_rl_lab"]).decode().splitlines()
 files=[REPO/p for p in tracked if Path(p).suffix in [".py",".yaml",".yml",".json"]]
 files+=list((REPO/"scripts/locomotion").glob("*issue15*.py"))
 files+=[REPO/"tests/locomotion/test_issue15_math.py",REPO/"docs/stage1_issue15_20261111.md"]
 files=sorted(set(files));inputs=["issue_source.json","cohort.json","protocol_plan.md","configuration_lineage_audit.json","smoke_audit.json","unit_tests.log"]
 frozen={"base_commit":BASE,"created_unix":time.time(),"zero_training_updates":True,"sha256":{str(p):sha(p) for p in files},"inputs":{p:sha(ROOT/p) for p in inputs},"formal_jobs":108,"formal_trials":647424,"model_count":35}
 assert not (ROOT/"frozen_evaluation_sources.json").exists(),"Existing prospective freeze must be preserved"
 (ROOT/"frozen_evaluation_sources.json").write_text(json.dumps(frozen,indent=2)+"\n")
 (ROOT/"formal_gates_complete").write_text("9 unit tests and observer/dynamics/RNG neutrality passed; scope/config/source/checkpoints frozen.\n")
 print("ISSUE15_FROZEN",len(files),flush=True)
if __name__=="__main__":main()
