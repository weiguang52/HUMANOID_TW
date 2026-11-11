"""Remote issue15 evaluation orchestration, zero training budget."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path("/root/gpufree-data/tw_stage1/issue15_20261111")
REPO=Path("/root/gpufree-data/projects/HUMANOID_TW")
BASE="60a8cc25ccce3a8ae22fcf7e5dd6f8d53db7613b"
COMMANDS=["forward_03","yaw_positive","yaw_negative","forward_05","forward_08","stand"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(args,log,phase):
 assert not (ROOT/"stop").exists()
 Path(log).parent.mkdir(parents=True,exist_ok=True)
 with Path(log).open("w") as f:
  p=subprocess.Popen(["bash","-c",'source scripts/locomotion/remote_env.sh; exec "$@"',"issue15",*map(str,args)],cwd=REPO,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  x={"phase":phase,"pid":p.pid,"args":list(map(str,args)),"state":"running","started_unix":time.time(),"log":str(log)}
  (ROOT/"progress.json").write_text(json.dumps(x,indent=2));print("START",phase,p.pid,flush=True)
  while p.poll() is None:
   if (ROOT/"stop").exists():os.killpg(p.pid,signal.SIGTERM);p.wait();break
   time.sleep(2)
 x.update(state="complete" if p.returncode==0 else "failed",returncode=p.returncode,finished_unix=time.time())
 (ROOT/"progress.json").write_text(json.dumps(x,indent=2));print("END",phase,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(str(log))
def evaluate_args(model,seed,output,num_envs=1024,commands=None,observer=True,capture=-1,legacy=False,plain=False):
 args=["python","-u","scripts/locomotion/eval_cmd.py" if plain else "scripts/locomotion/eval_issue15.py","--task","Unitree-TW-30dof-Velocity","--headless","--num_envs",str(num_envs),"--seed",str(seed),"--checkpoint",model["checkpoint"],"--output",str(output)]
 if not legacy:
  args+=["--experiment_profile","robust08","--mass_randomization","bounded","--last_action_observation","requested","--issue5_condition","clean","--extended_commands","--command_filter",",".join(commands or COMMANDS),"--paired_resets"]
  if model.get("critic_clock"):args+=["--critic_recovery_clock"]
  if observer:args+=["--issue15_records",str(output.parent/"moments"/output.stem)]
  if capture>=0:args+=["--issue15_capture_env",str(capture),"--issue15_capture_command","forward_03"]
 return args
