"""Render the two predeclared representative modes from recorded states."""
import json,subprocess,time
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue7_20261109')
models=json.loads((ROOT/'input_models.json').read_text())
selections=json.loads((ROOT/'representatives.json').read_text())
assert len(selections)==2
for path in selections:
    selection=json.loads(Path(path).read_text());rank=selection['rank']
    for view in ['front','side']:
        output=ROOT/f'mode{rank}_{view}.json'
        if output.exists():continue
        args=['python','-u','scripts/locomotion/eval_cmd.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1','--seed',str(selection['eval_seed']),'--checkpoint',models[selection['model']]['checkpoint'],'--experiment_profile','robust08','--with_pushes','--issue5_condition','combined','--mass_randomization','bounded','--video','--camera_view',view,'--trajectory_replay',path,'--output',str(output)]
        with output.with_suffix('.log').open('w') as out:
            p=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue7replay',*args],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
            progress={'pid':p.pid,'log':str(output.with_suffix('.log')),'state':'running','started_unix':time.time()}
            (ROOT/'progress.json').write_text(json.dumps(progress,indent=2));p.wait()
        progress.update(state='complete' if p.returncode==0 else 'failed',returncode=p.returncode,finished_unix=time.time())
        (ROOT/'progress.json').write_text(json.dumps(progress,indent=2))
        print('REPLAY',rank,view,p.returncode,flush=True)
        if p.returncode:raise RuntimeError(output)
        data=json.loads(output.read_text());assert not data['physics_integrated'] and data['frames']==51
(ROOT/'replays_complete').write_text('complete')
