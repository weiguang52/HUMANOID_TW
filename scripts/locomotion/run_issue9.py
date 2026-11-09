"""Issue #9 fixed-seed expansion; no configuration changes."""
import json,hashlib,subprocess,time,os
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue9_20261109')
REPO=Path('/root/gpufree-data/projects/HUMANOID_TW');os.chdir(REPO)
PARENT=REPO/'logs/rsl_rl/tw_stage1/2026-11-08_20-44-43_issues20261108_robust05/model_7496.pt'
SHA='c200e355d29fda60225761550ea80642bb222778ce50c00ae3deb3d95fa7a38d'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==SHA
assert not json.loads((ROOT/'parent_availability_audit.json').read_text())['independent_other_training_seed_available']
state_path=ROOT/'training_state.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}
inputs={}
for seed in [1,2]:
    for variant,bundle,name in [('requested','stage1_issue6_20261109','bounded_strengthen'),('applied','stage1_issue8_20261109','applied')]:
        p=REPO/f'validation_artifacts/{bundle}/models/{name}_s{seed}/model_8995.pt'
        inputs[f'{variant}_s{seed}']={'checkpoint':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'variant':variant,'training_seed':seed,'origin':'prior_round'}
def run(args,log):
    if (ROOT/'stop').exists():raise RuntimeError('stop requested')
    print('START',log.name,flush=True)
    with log.open('w') as out:
        p=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue9',*args],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
        progress={'pid':p.pid,'log':str(log),'args':args,'state':'running','started_unix':time.time()}
        (ROOT/'progress.json').write_text(json.dumps(progress,indent=2));p.wait()
    progress.update(state='complete' if p.returncode==0 else 'failed',returncode=p.returncode,finished_unix=time.time())
    (ROOT/'progress.json').write_text(json.dumps(progress,indent=2));print('END',log.name,p.returncode,flush=True)
    if p.returncode:raise RuntimeError(log)
def train(name,variant,seed):
    return ['python','-u','scripts/locomotion/train.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','4096','--max_iterations','1500','--run_name','issue9_20261109_'+name,'--seed',str(seed),'--resume','--load_run',PARENT.parent.name,'--checkpoint',PARENT.name,'--experiment_profile','robust08','--issue5_training_mode','strengthen','--issue5_factor','push','--mass_randomization','bounded','--last_action_observation',variant]
def evaluate(checkpoint,variant,seed,output):
    return ['python','-u','scripts/locomotion/eval_cmd.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1024','--seed',str(seed),'--checkpoint',str(checkpoint),'--experiment_profile','robust08','--with_pushes','--issue5_condition','combined','--mass_randomization','bounded','--last_action_observation',variant,'--extended_commands','--command_filter','forward_03,forward_05,forward_08','--paired_resets','--output',str(output)]
for seed in [3,4,5]:
    for variant in ['requested','applied']:
        name=f'{variant}_s{seed}'
        if name not in state:
            run_name='issue9_20261109_'+name
            if list((REPO/'logs/rsl_rl/tw_stage1').glob('*_'+run_name)):raise RuntimeError('Incomplete run requires explicit recovery')
            run(train(name,variant,seed),ROOT/f'{name}_train.log')
            folders=list((REPO/'logs/rsl_rl/tw_stage1').glob('*_'+run_name));assert len(folders)==1
            checkpoint=max(folders[0].glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
            state[name]={'variant':variant,'seed':seed,'checkpoint':str(checkpoint),'run_dir':str(folders[0]),'parent_sha256':SHA,'iterations':1500,'num_envs':4096}
            state_path.write_text(json.dumps(state,indent=2))
        inputs[name]={'checkpoint':state[name]['checkpoint'],'sha256':hashlib.sha256(Path(state[name]['checkpoint']).read_bytes()).hexdigest(),'variant':variant,'training_seed':seed,'origin':'issue9'}
        (ROOT/'input_models.json').write_text(json.dumps(inputs,indent=2))
        for eval_seed in [1,2,3]:
            output=ROOT/f'{name}_eval{eval_seed}.json'
            if not output.exists():run(evaluate(state[name]['checkpoint'],variant,eval_seed,output)+['--failure_traces',str(ROOT/'traces'/output.stem)],output.with_suffix('.log'))
            d=json.loads(output.read_text());assert d['last_action_observation']==variant and d['critic_last_action_params']=={}
(ROOT/'high_resolution_complete').write_text('complete')
videos=[]
for name,entry in state.items():
    rows=[x for seed in [1,2,3] for x in json.loads((ROOT/f'{name}_eval{seed}.json').read_text())['results']]
    commands=['forward_03','forward_05','forward_08'];worst=max(commands,key=lambda c:sum(x['failed_trials'] for x in rows if x['command']==c));videos.append({'model':name,'worst_command':worst})
    for view in ['front','side']:
        output=ROOT/f'{name}_worst_{view}.json'
        if output.exists():continue
        args=evaluate(entry['checkpoint'],entry['variant'],1,output);args[args.index('--num_envs')+1]='1';args[args.index('--command_filter')+1]=worst;args+=['--video','--camera_view',view]
        run(args,output.with_suffix('.log'))
(ROOT/'video_plan.json').write_text(json.dumps(videos,indent=2))
(ROOT/'all_training_evaluation_video_complete').write_text('complete')
print('ISSUE9_ALL_JOBS_COMPLETE',flush=True)
