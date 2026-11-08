"""Run matched baseline challenge/symmetry tests after the main GPU queue."""
import json,subprocess,time
from pathlib import Path
root=Path('/root/gpufree-data/tw_stage1/issue_runs_20261108')
while not (root/'all_training_evaluation_complete').exists():
    if (root/'stop').exists():raise RuntimeError('stop requested')
    time.sleep(20)
for protocol in ['nominal','challenge']:
    for seed in [1,2,3]:
        output=root/f'ft5_{protocol}_seed{seed}.json'
        if output.exists():json.loads(output.read_text());continue
        args=['python','-u','scripts/locomotion/eval_cmd.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','128','--seed',str(seed),'--checkpoint','validation_artifacts/tw_stage1_ft5_20261107/model_5997.pt','--output',str(output),'--extended_commands']
        if protocol=='challenge':args+=['--experiment_profile','robust08','--with_pushes']
        print('START',output.name,flush=True)
        with output.with_suffix('.log').open('w') as out:
            p=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','supplement',*args],stdout=out,stderr=subprocess.STDOUT)
            (root/'supplement_progress.json').write_text(json.dumps({'pid':p.pid,'output':str(output),'started_unix':time.time()},indent=2))
            p.wait()
        if p.returncode:raise RuntimeError(f'{output}: exit {p.returncode}')
        json.loads(output.read_text());print('END',output.name,flush=True)
(root/'baseline_supplement_complete').write_text('complete')
