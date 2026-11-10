"""Fixed issue12 cohort and remote-only process orchestration."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path("/root/gpufree-data/tw_stage1/issue12_20261110")
REPO = Path("/root/gpufree-data/projects/HUMANOID_TW")
BASE = "67d9879bf23ec5a79d23cb1e6c54d612a2c62172"
COMMANDS = ["forward_03", "forward_05", "forward_08"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, log, phase):
    if (ROOT / "stop").exists():
        raise RuntimeError("Explicit stop marker exists")
    print("START", phase, log.name, flush=True)
    with Path(log).open("w") as out:
        p = subprocess.Popen(["bash", "-c",
            'source scripts/locomotion/remote_env.sh; exec "$@"', "issue12", *map(str,args)],
            cwd=REPO, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True)
        state = {"phase": phase, "state": "running", "pid": p.pid,
                 "log": str(log), "args": list(map(str,args)), "started_unix": time.time()}
        (ROOT/"progress.json").write_text(json.dumps(state,indent=2))
        while p.poll() is None:
            if (ROOT/"stop").exists():
                os.killpg(p.pid, signal.SIGTERM)
                p.wait()
                break
            time.sleep(2)
    state.update(state="complete" if p.returncode==0 else "failed",
                 returncode=p.returncode,finished_unix=time.time())
    (ROOT/"progress.json").write_text(json.dumps(state,indent=2))
    print("END",phase,log.name,p.returncode,flush=True)
    if p.returncode:
        raise RuntimeError(str(log))


def train_args(seed, group, audit, weight, num_envs=4096):
    models=json.loads((ROOT/"input_models.json").read_text())
    parent=models[f"requested_s{seed}_u4500"]
    checkpoint=Path(parent["checkpoint"])
    assert sha(checkpoint)==parent["sha256"]
    args=["python","-u","scripts/locomotion/train.py","--task","Unitree-TW-30dof-Velocity",
          "--headless","--num_envs",str(num_envs),"--max_iterations","1500",
          "--run_name",f"issue12_20261110_{group}_s{seed}","--seed",str(seed),
          "--resume","--load_run",checkpoint.parent.name,"--checkpoint",checkpoint.name,
          "--experiment_profile","robust08","--issue5_training_mode","strengthen",
          "--issue5_factor","push","--mass_randomization","bounded",
          "--last_action_observation","requested","--issue12_audit_dir",str(audit)]
    if weight is not None:
        args += ["--post_push_pitch_weight",str(weight)]
    return args


def evaluate(checkpoint, seed, output, trace=True, num_envs=1024):
    args=["python","-u","scripts/locomotion/eval_cmd.py","--task","Unitree-TW-30dof-Velocity",
          "--headless","--num_envs",str(num_envs),"--seed",str(seed),"--checkpoint",str(checkpoint),
          "--experiment_profile","robust08","--with_pushes","--issue5_condition","combined",
          "--mass_randomization","bounded","--last_action_observation","requested",
          "--extended_commands","--command_filter",",".join(COMMANDS),"--paired_resets",
          "--output",str(output)]
    if trace:
        args += ["--failure_traces",str(ROOT/"traces"/Path(output).stem),"--trial_risk_records"]
    return args
