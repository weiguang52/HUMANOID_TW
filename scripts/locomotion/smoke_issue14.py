"""Fixed preflight: exact switch-off PPO regression plus clock-enabled smoke."""
import json,os,copy,subprocess
from pathlib import Path
import numpy as np
import torch,yaml
from issue13_tracking import digest
from issue14_common import ROOT,REPO,train_args,run,sha
from run_issue12 import difference,load_yaml

def compare_npz(a,b):
 with np.load(a) as x,np.load(b) as y:
  assert set(x.files)==set(y.files)
  for k in x.files:assert x[k].dtype==y[k].dtype and x[k].shape==y[k].shape and x[k].tobytes()==y[k].tobytes(),(str(a),k)
def main():
 os.chdir(REPO)
 assert "Ran 10 tests" in (ROOT/"unit_tests.log").read_text() and "\nOK\n" in (ROOT/"unit_tests.log").read_text()
 records={};state=ROOT/"smoke_state.json"
 if state.exists():records=json.loads(state.read_text())
 for name,clock,legacy,rollout in [("legacy_ppo",False,True,False),("off_ppo",False,False,False),("on_ppo",True,False,False),("on_rollout",True,False,True)]:
  audit=ROOT/"smoke"/name
  if name not in records:
   assert not audit.exists(),"Incomplete smoke output; inspect before retrying"
   args=train_args(1,"issue14_smoke_"+name,audit,clock,64,2,legacy)
   if rollout:args+=["--issue12_smoke_steps","1000"]
   run(args,ROOT/(name+".log"),name)
   summary=json.loads((audit/"run_summary.json").read_text());folder=Path(summary["log_dir"])
   records[name]={"audit":str(audit),"run_dir":str(folder),"summary":summary}
   state.write_text(json.dumps(records,indent=2))
  assert json.loads((audit/"loaded_state_audit.json").read_text())["passed"]
  if clock:assert json.loads((audit/"critic_clock_migration.json").read_text())["passed"]
 a=records["legacy_ppo"];b=records["off_ppo"]
 for kind in ["env","agent"]:
  x=load_yaml(Path(a["run_dir"])/f"params/{kind}.yaml");y=load_yaml(Path(b["run_dir"])/f"params/{kind}.yaml")
  changes=list(difference(x,y));allowed={"/log_dir"} if kind=="env" else {"/run_name"}
  assert all(d[0] in allowed for d in changes),changes
 for point in ["initial"]:
  compare_npz(Path(a["audit"])/(point+"_state")/(point+".npz"),Path(b["audit"])/(point+"_state")/(point+".npz"))
 def checkpoint(record):
  return torch.load(Path(record["run_dir"])/f"model_{record['summary']['final_iteration']}.pt",map_location="cpu",weights_only=False)
 assert digest(checkpoint(a))==digest(checkpoint(b)),"Switch-off PPO state diverged from exact legacy training source"
 assert (Path(a["audit"])/"reward_contributions.csv").read_bytes()==(Path(b["audit"])/"reward_contributions.csv").read_bytes()
 old=Path("/root/gpufree-data/tw_stage1/issue12_20261110/smoke/treatment_attempt1");new=Path(records["on_rollout"]["audit"])
 for point in ["initial","final"]:compare_npz(old/(point+"_state")/(point+".npz"),new/(point+"_state")/(point+".npz"))
 compare_npz(old/"smoke_rewards.npz",new/"smoke_rewards.npz")
 # Clock is privileged: on/off construction starts with exactly the same physical state and complete RNG.
 compare_npz(Path(a["audit"])/"initial_state/initial.npz",Path(records["on_ppo"]["audit"])/"initial_state/initial.npz")
 final=checkpoint(records["on_ppo"]);assert final["critic_state_dict"]["mlp.0.weight"].shape[1]==136
 result={"passed":True,"unit_tests":10,"switch_off_two_PPO_updates_exact_legacy_model_critic_optimizer_std":True,
 "switch_off_training_reward_CSV_byte_equal":True,"clock_on_1000_step_rollout_exact_archived_treatment_rewards_costs_root_joints_and_all_RNG":True,
 "clock_on_off_initial_physics_and_complete_RNG_equal":True,"clock_on_two_update_PPO_smoke_passed":True,"smoke_updates":6,"smoke_envs":64,
 "formal_updates_not_started":True,"no_smoke_checkpoint_used_for_formal_training":True,"runs":records}
 (ROOT/"runtime_smoke_audit.json").write_text(json.dumps(result,indent=2));(ROOT/"runtime_smoke_complete").write_text("Clock boundaries, strict parent migration, off-switch PPO and on-clock rollouts passed.\n")
 print("ISSUE14_RUNTIME_SMOKE_PASS",flush=True)
if __name__=="__main__":main()
