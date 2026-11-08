"""Stage 1 of issue #5: paired single-factor evaluation without retraining."""
import json,subprocess
from pathlib import Path
root=Path('/root/gpufree-data/tw_stage1/issue5_20261109')
checkpoint=Path('validation_artifacts/stage1_issues_20261108/models/robust08/model_8995.pt')
for condition in ['clean','push','material','delay','noise','combined']:
    for seed in [1,2,3]:
        output=root/f'robust08_s1_{condition}_eval{seed}.json'
        if output.exists():json.loads(output.read_text());continue
        args=['python','-u','scripts/locomotion/eval_cmd.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','128','--seed',str(seed),'--checkpoint',str(checkpoint),'--experiment_profile','robust08','--with_pushes','--issue5_condition',condition,'--extended_commands','--command_filter','forward_03,forward_05,forward_08','--paired_resets','--output',str(output)]
        print('START',output.name,flush=True)
        with output.with_suffix('.log').open('w') as log:
            p=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','factors',*args],stdout=log,stderr=subprocess.STDOUT)
            (root/'progress.json').write_text(json.dumps({'pid':p.pid,'output':str(output)},indent=2));p.wait()
        if p.returncode:raise RuntimeError(f'{output}: exit {p.returncode}')
        json.loads(output.read_text());print('END',output.name,flush=True)
(root/'factor_evaluation_complete').write_text('complete')
