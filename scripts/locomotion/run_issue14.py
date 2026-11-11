"""Five fixed1500-update critic-clock continuations then thirty formal evaluation jobs."""
import csv,json,os,re,subprocess,shutil
from pathlib import Path
import numpy as np
import torch
from issue14_common import ROOT,REPO,BASE,WEIGHT,sha,train_args,run
from run_issue12 import difference,load_yaml
CLEAN=["forward_03","yaw_positive","yaw_negative","forward_05","forward_08"]
FORWARD=["forward_03","forward_05","forward_08"]

def verify_frozen():
 assert (ROOT/"formal_gates_complete").exists()
 frozen=json.loads((ROOT/"frozen_training_sources.json").read_text())
 for p,h in frozen["sha256"].items():assert sha(p)==h,("Frozen source changed",p)
 for p,h in json.loads((ROOT/"protocol_plan_sha256.json").read_text()).items():assert sha(ROOT/p)==h
 assert subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()==BASE
 assert shutil.disk_usage(ROOT).free>2*1024**3
 assert not os.environ.get("TW_RESET_STD")

def evaluate_args(model,es,condition,output,capture=-1):
 args=["python","-u","scripts/locomotion/eval_issue14.py","--task","Unitree-TW-30dof-Velocity","--headless",
 "--num_envs","1024","--seed",str(es),"--checkpoint",model["checkpoint"],"--experiment_profile","robust08",
 "--mass_randomization","bounded","--last_action_observation","requested","--issue5_condition",condition,
 "--extended_commands","--command_filter",",".join(CLEAN if condition=="clean" else FORWARD),
 "--paired_resets","--critic_recovery_clock","--output",str(output),
 "--issue13_records",str(output.parent/"moments"/output.stem)]
 if condition=="combined":args+=["--with_pushes","--failure_traces",str(output.parent/"traces"/output.stem),"--trial_risk_records"]
 if capture>=0:args+=["--issue13_capture_env",str(capture),"--issue13_capture_command","forward_08"]
 return args

def audit_training(seed,record,parent):
 from smoke_issue14 import compare_npz
 folder=Path(record["run_dir"]);audit=Path(record["audit_dir"])
 historical=json.loads((ROOT/"historical_models.json").read_text())[f"treatment_s{seed}"];old=Path(historical["run_dir"]);changes={}
 for kind in ["env","agent"]:
  a=load_yaml(old/f"params/{kind}.yaml");b=load_yaml(folder/f"params/{kind}.yaml");diffs=list(difference(a,b))
  allowed={"/log_dir","/observations/critic/recovery_clock"} if kind=="env" else {"/run_name"}
  assert all(x[0] in allowed for x in diffs),(seed,kind,diffs)
  if kind=="env":
   assert b["scene"]["num_envs"]=="4096" and float(b["rewards"]["post_push_pitch_recovery"]["weight"])==-WEIGHT
  else:
   assert b["max_iterations"]=="1500" and b["resume"]=="true" and b["load_checkpoint"]==Path(parent["checkpoint"]).name
  changes[kind]=diffs
 loaded=json.loads((audit/"loaded_state_audit.json").read_text());migration=json.loads((audit/"critic_clock_migration.json").read_text());summary=json.loads((audit/"run_summary.json").read_text())
 assert loaded["passed"] and loaded["checkpoint_sha256"]==parent["sha256"] and migration["passed"]
 assert loaded["observation_dims"]=={"policy":[101],"critic":[136]}
 assert summary["additional_updates"]==1500 and summary["num_envs"]==4096 and not summary["std_reset"]
 agent=load_yaml(folder/"params/agent.yaml");steps=int(agent["num_steps_per_env"])
 rows=list(csv.DictReader((audit/"reward_contributions.csv").open()));assert len(rows)==1500
 assert summary["environment_steps"]==1500*steps
 assert all(int(float(x["environment_steps"]))==steps and float(x["max_term_identity_error"])<=1e-6 for x in rows)
 olddata=torch.load(parent["checkpoint"],map_location="cpu",weights_only=False);newdata=torch.load(record["checkpoint"],map_location="cpu",weights_only=False)
 assert newdata["iter"]==parent["iteration"]+1499==record["iteration"]
 a=olddata["optimizer_state_dict"]["state"];b=newdata["optimizer_state_dict"]["state"];assert set(a)==set(b)
 increments=[float(b[k]["step"])-float(a[k]["step"]) for k in a if "step" in a[k]]
 expected=1500*int(agent["algorithm"]["num_learning_epochs"])*int(agent["algorithm"]["num_mini_batches"])
 assert increments and min(increments)==max(increments)==expected
 assert newdata["critic_state_dict"]["mlp.0.weight"].shape==(512,136)
 assert newdata["actor_state_dict"]["mlp.0.weight"].shape==(512,101)
 assert all(torch.isfinite(x).all() for key in ["actor_state_dict","critic_state_dict"] for x in newdata[key].values())
 # Original paired start includes complete CPU/CUDA/Python/NumPy RNG and physical state.
 compare_npz(Path(historical["audit_dir"])/"initial_state/initial.npz",audit/"initial_state/initial.npz")
 log=re.sub(r"\x1b\[[0-9;?]*[A-Za-z]","",(ROOT/f"clock_s{seed}_train.log").read_text())
 iters=re.findall(r"Learning iteration\s+(\d+)/(\d+)",log);assert len(iters)==1500
 assert int(iters[0][0])==loaded["iteration"] and int(iters[-1][0])==record["iteration"]
 return {"passed":True,"config_changes_vs_original_treatment":changes,"parent_load_exact":True,"migration":migration,
 "updates":1500,"optimizer_step_increase":expected,"initial_physics_and_complete_RNG_match_original_treatment":True,"policy":101,"critic":136,"std_reset":False}

def audit_eval(output,condition,model):
 d=json.loads(output.read_text());commands=CLEAN if condition=="clean" else FORWARD
 assert [r["command"] for r in d["results"]]==commands
 assert d["checkpoint"]==model["checkpoint"] and d["deterministic_policy"]
 assert d["observation_dims"]=={"policy":[101],"critic":[136]}
 track=d["issue13_tracking"];assert track["model_optimizer_state_unchanged"] and track["training_updates"]==0
 for row in d["results"]:
  assert row["num_envs"]==1024 and row["seconds"]==20
  folder=Path(track["output"]);assert (folder/(row["command"]+".npz")).exists()
  if condition=="combined":assert (output.parent/"traces"/output.stem/(row["command"]+"_trials.npz")).exists()
 if condition=="clean":
  physics=track["physics_audit"]
  assert physics["mass_equal_nominal"] and physics["action_class"]=="JointPositionAction"
  assert all(physics[k]==1.0 for k in ["material_static_min","material_static_max","material_dynamic_min","material_dynamic_max","terrain_static","terrain_dynamic"])
  assert not any(physics[k] for k in ["pushes","observation_noise","mass_event","material_event"])
 assert sha(model["checkpoint"])==model["sha256"]
 return {"passed":True,"condition":condition,"model_sha256":model["sha256"],"commands":len(commands),"initial_trials":len(commands)*1024,"training_updates":0,"model_optimizer_unchanged":True}

def main():
 os.chdir(REPO);verify_frozen()
 (ROOT/"formal_driver_process.json").write_text(json.dumps({"pid":os.getpid(),"state":"running"}))
 parents=json.loads((ROOT/"input_parents.json").read_text());statepath=ROOT/"formal_training_state.json";models=json.loads(statepath.read_text()) if statepath.exists() else {};audits={}
 for seed in range(1,6):
  name=f"clock_s{seed}";parent=parents[f"requested_s{seed}_u4500"];verify_frozen()
  if name not in models:
   audit=ROOT/"training"/name;suffix=f"issue14_20261111_{name}"
   assert not audit.exists() and not list((REPO/"logs/rsl_rl/tw_stage1").glob("*_"+suffix)),"Incomplete run requires budget-preserving recovery"
   run(train_args(seed,suffix,audit),ROOT/(name+"_train.log"),"training_"+name)
   summary=json.loads((audit/"run_summary.json").read_text());folder=Path(summary["log_dir"]);iteration=summary["final_iteration"];checkpoint=folder/f"model_{iteration}.pt"
   record={"group":"clock","training_seed":seed,"weight":WEIGHT,"critic_recovery_clock":True,"run_dir":str(folder),"audit_dir":str(audit),
    "parent_checkpoint":parent["checkpoint"],"parent_sha256":parent["sha256"],"additional_updates":1500,"cumulative_requested_updates":6000,
    "iteration":iteration,"checkpoint":str(checkpoint),"sha256":sha(checkpoint)}
   models[name]=record;statepath.write_text(json.dumps(models,indent=2))
  else:assert sha(models[name]["checkpoint"])==models[name]["sha256"]
  audits[name]=audit_training(seed,models[name],parent);(ROOT/"training_audits.json").write_text(json.dumps(audits,indent=2))
 assert len(models)==5 and len(audits)==5
 (ROOT/"final_models.json").write_text(json.dumps(models,indent=2));(ROOT/"training_complete").write_text("Exactly five1500-update continuations from original4500 parents.\n")
 jobsfile=ROOT/"formal_job_state.json";jobs=json.loads(jobsfile.read_text()) if jobsfile.exists() else {}
 for condition in ["clean","combined"]:
  folder=ROOT/condition;folder.mkdir(exist_ok=True)
  for name,m in models.items():
   for es in [1,2,3]:
    verify_frozen();output=folder/f"{name}_eval{es}.json";key=f"{condition}:{name}:eval{es}"
    if not output.exists():run(evaluate_args(m,es,condition,output),output.with_suffix(".log"),key)
    jobs[key]=audit_eval(output,condition,m);jobsfile.write_text(json.dumps(jobs,indent=2))
 assert len(jobs)==30 and sum(x["initial_trials"] for x in jobs.values())==122880
 (ROOT/"formal_evaluation_complete").write_text("Thirty new-policy jobs,120command records,122880initial trials; old two groups reused.\n")
 (ROOT/"formal_driver_process.json").write_text(json.dumps({"pid":os.getpid(),"state":"complete"}));print("ISSUE14_TRAINING_AND_EVALUATION_COMPLETE",flush=True)
if __name__=="__main__":main()
