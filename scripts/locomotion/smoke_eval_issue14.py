"""Operational adapter checks on smoke models; no formal trials or updates."""
import json,os
from pathlib import Path
from issue14_common import ROOT,REPO,run,sha
from run_issue14 import evaluate_args
def main():
 os.chdir(REPO);assert json.loads((ROOT/"runtime_smoke_audit.json").read_text())["passed"]
 state=json.loads((ROOT/"smoke_state.json").read_text())["on_ppo"];summary=state["summary"]
 p=Path(state["run_dir"])/f"model_{summary['final_iteration']}.pt";model={"checkpoint":str(p),"sha256":sha(p)}
 folder=ROOT/"eval_smoke";folder.mkdir(exist_ok=True);checks={}
 for condition in ["clean","combined"]:
  out=folder/(condition+".json");assert not out.exists(),"Inspect existing output before retrying"
  args=evaluate_args(model,1,condition,out);args[args.index("--num_envs")+1]="64";args[args.index("--command_filter")+1]="forward_08"
  run(args,out.with_suffix(".log"),"eval_adapter_smoke_"+condition)
  d=json.loads(out.read_text());assert d["observation_dims"]=={"policy":[101],"critic":[136]} and len(d["results"])==1
  assert d["results"][0]["num_envs"]==64 and d["results"][0]["seconds"]==20
  assert d["issue13_tracking"]["model_optimizer_state_unchanged"] and d["issue13_tracking"]["training_updates"]==0
  checks[condition]={"passed":True,"checkpoint_sha256":model["sha256"],"policy":101,"critic":136,"formal_trial_denominator":0}
 (ROOT/"eval_adapter_smoke_audit.json").write_text(json.dumps({"passed":True,"checks":checks,"formal_updates":0,"formal_trials":0},indent=2))
 print("ISSUE14_EVAL_ADAPTER_SMOKE_PASS",flush=True)
if __name__=="__main__":main()
