"""Resolve saved resume ancestry, including unshown parents, without extending evaluation scope."""
import json
from pathlib import Path
import yaml
from issue15_common import ROOT,REPO,sha
def main():
 cohort=json.loads((ROOT/"cohort.json").read_text());logroot=REPO/"logs/rsl_rl/tw_stage1";records={};todo=[Path(m["checkpoint"]) for m in cohort["models"].values()]
 while todo:
  checkpoint=todo.pop();h=sha(checkpoint)
  if h in records:continue
  run=checkpoint.parent
  # Archived FT5 is byte-identical to its actual historical run checkpoint.
  if h==cohort["models"]["ft5_s1"]["sha256"]:run=Path(cohort["models"]["ft5_s1"]["run_dir"])
  agent=run/"params/agent.yaml";env=run/"params/env.yaml";entry={"checkpoint":str(checkpoint),"sha256":h,"run_dir":str(run),"evaluated_aliases":[k for k,v in cohort["models"].items() if v["sha256"]==h]}
  if not agent.exists():entry["parent_resolution"]="Saved agent YAML unavailable; no guessed ancestry";records[h]=entry;continue
  ag=yaml.load(agent.read_text(),Loader=yaml.BaseLoader);entry["agent_config_sha256"]=sha(agent)
  if env.exists():entry["env_config_sha256"]=sha(env)
  entry["saved_resume_settings"]={k:ag.get(k) for k in ["resume","load_run","load_checkpoint","seed","run_name","max_iterations"]}
  if ag.get("resume")=="true":
   parent=logroot/str(ag.get("load_run"))/str(ag.get("load_checkpoint"))
   if parent.exists():
    ph=sha(parent);entry["parent_checkpoint"]=str(parent);entry["parent_sha256"]=ph
    entry["parent_resolution"]="Exact saved resume checkpoint found"
    if ph!=h:todo.append(parent)
   else:entry["parent_resolution"]="Saved resume path not an existing literal checkpoint; not guessed"
  else:entry["parent_resolution"]="Saved run not configured to resume"
  records[h]=entry
 ft5=cohort["models"]["ft5_s1"]["sha256"];rob05=cohort["models"]["robust05_s1"]["sha256"];chain=[];cur=rob05;seen=set()
 while cur not in seen and cur in records:
  seen.add(cur);chain.append(cur);cur=records[cur].get("parent_sha256")
 assert ft5 not in chain,"If actual evidence differs, revise explanatory addendum"
 notes=["The issue's FT5->robust05 display comparison is not literal checkpoint ancestry: robust05 resumes yaw_s1, which shares the FT4 ancestor with FT5. FT5 is not the saved parent/ancestor of robust05.",
 "Prospective scope still evaluates exactly35 unique requested checkpoints. Intermediate parents are inspected for provenance only, not silently added as new evaluated policies.",
 "requested3000 and requested4500 are intermediate/final checkpoints within one continuation run; their saved resume settings point to requested1500, not a separate restart at3000.",
 "Strengthen/bounded are sibling restarts from robust05. Control/treatment/clock are sibling continuations from requested4500. Diagnostic adjacent contrasts alone do not prove causality."]
 out={"passed":True,"records_by_checkpoint_sha256":records,"notes":notes,"robust05_saved_ancestor_chain_sha256":chain,"FT5_not_literal_ancestor_of_robust05":True,"source_sha256":sha(__file__)}
 (ROOT/"actual_resume_lineage.json").write_text(json.dumps(out,indent=2)+"\n");print("ACTUAL_ANCESTRY_AUDIT_COMPLETE",len(records))
if __name__=="__main__":main()
