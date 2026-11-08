"""Sequential, resumable remote issue #1-4 experiment runner."""
import json, os, subprocess, time
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue_runs_20261108')
REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
os.chdir(REPO)

def run(args, log):
    with log.open('w') as out:
        print('START',log.name,flush=True)
        p=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','runner',*args],stdout=out,stderr=subprocess.STDOUT)
        progress={'pid':p.pid,'args':args,'log':str(log),'started_unix':time.time(),'state':'running'}
        (ROOT/'progress.json').write_text(json.dumps(progress,indent=2))
        p.wait()
        progress.update(state='complete' if p.returncode==0 else 'failed',returncode=p.returncode,finished_unix=time.time())
        (ROOT/'progress.json').write_text(json.dumps(progress,indent=2))
        print('END',log.name,p.returncode,flush=True)
    if p.returncode:
        raise RuntimeError(f'{log}: exit {p.returncode}')

def evaluate(checkpoint,name,challenge=False):
    for seed in [1,2,3]:
        output=ROOT/f'{name}_{"challenge" if challenge else "nominal"}_seed{seed}.json'
        if output.exists():
            json.loads(output.read_text()); continue
        args=['python','-u','scripts/locomotion/eval_cmd.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','128','--seed',str(seed),'--checkpoint',str(checkpoint),'--output',str(output),'--extended_commands']
        if challenge: args+=['--experiment_profile','robust08','--with_pushes']
        run(args,output.with_suffix('.log'))
        json.loads(output.read_text())

while not (ROOT/'baseline_complete').exists():
    if (ROOT/'stop').exists(): raise RuntimeError('stop requested')
    time.sleep(20)
# Validate action-delay integration before committing a long training budget.
smoke_done=ROOT/'robust_smoke_complete'
if not smoke_done.exists():
    run(['python','-u','scripts/locomotion/train.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','64','--max_iterations','2','--run_name','issues_robust_smoke','--seed','1','--resume','--load_run','2026-11-07_17-12-49_tw_stage1_s1_ft5','--checkpoint','model_5997.pt','--experiment_profile','robust05'],ROOT/'robust_smoke.log')
    smoke_done.write_text('complete')
# Check camera rendering on the exact precolored experiment asset.
video_smoke=ROOT/'video_smoke.json'
if not video_smoke.exists():
    run(['python','-u','scripts/locomotion/eval_cmd.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1','--seed','1','--seconds','0.2','--video','--camera_view','front','--checkpoint','validation_artifacts/tw_stage1_ft5_20261107/model_5997.pt','--output',str(video_smoke)],ROOT/'video_smoke.log')
state_path=ROOT/'training_state.json'
state=json.loads(state_path.read_text()) if state_path.exists() else {}
for item in json.loads((ROOT/'plan.json').read_text()):
    name=item['name']
    if (ROOT/'stop').exists(): raise RuntimeError('stop requested')
    if name not in state:
        if item['parent']=='ft4':
            parent=REPO/'logs/rsl_rl/tw_stage1/2026-11-07_16-32-16_tw_stage1_s1_ft4/model_4498.pt'
        else: parent=Path(state[item['parent']]['checkpoint'])
        run_name='issues20261108_'+name
        existing=list((REPO/'logs/rsl_rl/tw_stage1').glob('*_'+run_name))
        if existing: raise RuntimeError(f'Incomplete previous run requires explicit recovery: {existing}')
        run(['python','-u','scripts/locomotion/train.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','4096','--max_iterations',str(item['iterations']),'--run_name',run_name,'--seed',str(item['seed']),'--resume','--load_run',parent.parent.name,'--checkpoint',parent.name,'--experiment_profile',item['profile']],ROOT/(name+'_train.log'))
        paths=list((REPO/'logs/rsl_rl/tw_stage1').glob('*_'+run_name))
        assert len(paths)==1
        ckpt=max(paths[0].glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
        state[name]={**item,'checkpoint':str(ckpt),'run_dir':str(paths[0])}
        state_path.write_text(json.dumps(state,indent=2))
    evaluate(state[name]['checkpoint'],name)
    if item['profile'] in ['range05','range08','robust05','robust08']:
        evaluate(state[name]['checkpoint'],name,challenge=True)
# Representative videos cover every commanded motion for baseline and final
# robustness arm, while quantitative acceptance always uses 128-env/3-seed tests.
for name, checkpoint, profile, extended in [
    ('ft5', REPO/'validation_artifacts/tw_stage1_ft5_20261107/model_5997.pt','default',False),
    ('robust08', Path(state['robust08']['checkpoint']),'default',True),
]:
    for view in ['front','side']:
        output=ROOT/f'{name}_video_{view}.json'
        if output.exists():
            json.loads(output.read_text()); continue
        args=['python','-u','scripts/locomotion/eval_cmd.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1','--seed','1','--video','--camera_view',view,'--checkpoint',str(checkpoint),'--output',str(output),'--experiment_profile',profile]
        if extended: args+=['--extended_commands']
        run(args,output.with_suffix('.log'))
(ROOT/'all_training_evaluation_complete').write_text('complete')
