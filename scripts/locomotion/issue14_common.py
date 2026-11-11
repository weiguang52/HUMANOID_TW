"""Frozen issue14 cohort and remote-only orchestration."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path("/root/gpufree-data/tw_stage1/issue14_20261111")
REPO=Path("/root/gpufree-data/projects/HUMANOID_TW")
BASE="5a1f8b9a96a4b5d44cd8de2bc0900332d88e07bb"
WEIGHT=34.67877532210905
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def train_args(seed,name,audit,clock=True,num_envs=4096,updates=1500,legacy=False):
 parent=json.loads((ROOT/"input_parents.json").read_text())[f"requested_s{seed}_u4500"];p=Path(parent["checkpoint"])
 assert sha(p)==parent["sha256"]
 args=["python","-u","scripts/locomotion/train_legacy_issue14.py" if legacy else "scripts/locomotion/train.py",
 "--task","Unitree-TW-30dof-Velocity","--headless","--num_envs",str(num_envs),"--max_iterations",str(updates),
 "--run_name",name,"--seed",str(seed),"--resume","--load_run",p.parent.name,"--checkpoint",p.name,
 "--experiment_profile","robust08","--issue5_training_mode","strengthen","--issue5_factor","push",
 "--mass_randomization","bounded","--last_action_observation","requested","--issue12_audit_dir",str(audit),
 "--post_push_pitch_weight",str(WEIGHT)]
 if clock:args+=["--critic_recovery_clock"]
 return args
def run(args,log,phase):
 if (ROOT/"stop").exists():raise RuntimeError("Explicit stop marker")
 with Path(log).open("w") as f:
  p=subprocess.Popen(["bash","-c",'source scripts/locomotion/remote_env.sh; exec "$@"',"issue14",*map(str,args)],
   cwd=REPO,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  status={"phase":phase,"pid":p.pid,"args":list(map(str,args)),"state":"running","started_unix":time.time(),"log":str(log)}
  (ROOT/"progress.json").write_text(json.dumps(status,indent=2));print("START",phase,p.pid,flush=True)
  while p.poll() is None:
   if (ROOT/"stop").exists():os.killpg(p.pid,signal.SIGTERM);p.wait();break
   time.sleep(2)
 status.update(state="complete" if p.returncode==0 else "failed",returncode=p.returncode,finished_unix=time.time())
 (ROOT/"progress.json").write_text(json.dumps(status,indent=2));print("END",phase,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(str(log))
