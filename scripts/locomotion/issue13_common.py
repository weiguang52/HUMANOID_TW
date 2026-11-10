"""Issue13 fixed analysis-only cohort and remote orchestration."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path("/root/gpufree-data/tw_stage1/issue13_20261110")
REPO=Path("/root/gpufree-data/projects/HUMANOID_TW")
BASE="0ec49f358d1634b62b7f8b149381809b54d23b9b"
FORWARD=["forward_03","forward_05","forward_08"]
CLEAN=["forward_03","yaw_positive","yaw_negative","forward_05","forward_08"]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def args_for(name,seed,condition,output,records=True,num_envs=1024,capture_env=-1):
 m=json.loads((ROOT/"input_models.json").read_text())[name]
 assert sha(m["checkpoint"])==m["sha256"]
 args=["python","-u","scripts/locomotion/eval_issue13.py" if records else "scripts/locomotion/eval_cmd.py",
 "--task","Unitree-TW-30dof-Velocity","--headless","--num_envs",str(num_envs),"--seed",str(seed),
 "--checkpoint",m["checkpoint"],"--experiment_profile","robust08","--mass_randomization","bounded",
 "--last_action_observation","requested","--issue5_condition",condition,"--extended_commands",
 "--command_filter",",".join(CLEAN if condition=="clean" else FORWARD),"--paired_resets","--output",str(output)]
 if condition=="combined":
  args+=["--with_pushes","--failure_traces",str(Path(output).parent/"traces"/Path(output).stem),"--trial_risk_records"]
 if records:args+=["--issue13_records",str(Path(output).parent/"moments"/Path(output).stem)]
 if capture_env>=0:
  assert records
  args+=["--issue13_capture_env",str(capture_env),"--issue13_capture_command","forward_08"]
 return args
def run(args,log,phase):
 if (ROOT/"stop").exists():raise RuntimeError("Explicit stop marker")
 log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
 with log.open("w") as f:
  p=subprocess.Popen(["bash","-c",'source scripts/locomotion/remote_env.sh; exec "$@"',"issue13",*map(str,args)],
   cwd=REPO,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  status={"phase":phase,"pid":p.pid,"state":"running","args":list(map(str,args)),"started_unix":time.time(),"log":str(log)}
  (ROOT/"progress.json").write_text(json.dumps(status,indent=2))
  print("START",phase,p.pid,flush=True)
  while p.poll() is None:
   if (ROOT/"stop").exists():os.killpg(p.pid,signal.SIGTERM);p.wait();break
   time.sleep(2)
 status.update(state="complete" if p.returncode==0 else "failed",returncode=p.returncode,finished_unix=time.time())
 (ROOT/"progress.json").write_text(json.dumps(status,indent=2));print("END",phase,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(str(log))
