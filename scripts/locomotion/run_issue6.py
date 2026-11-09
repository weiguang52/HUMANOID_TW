"""Issue #6: audited bounded mass, fixed-budget training and paired evaluation."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue6_20261109')
REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
os.chdir(REPO)
PARENT=REPO/'logs/rsl_rl/tw_stage1/2026-11-08_20-44-43_issues20261108_robust05/model_7496.pt'
PARENT_SHA='c200e355d29fda60225761550ea80642bb222778ce50c00ae3deb3d95fa7a38d'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
assert '\nOK\n' in (ROOT/'mass_tests.log').read_text()
def run(args,log):
    if (ROOT/'stop').exists():raise RuntimeError('stop requested')
    print('START',log.name,flush=True)
    with log.open('w') as out:
        p=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue6',*args],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
        progress={'pid':p.pid,'log':str(log),'args':args,'state':'running','started_unix':time.time()}
        (ROOT/'progress.json').write_text(json.dumps(progress,indent=2))
        p.wait()
    progress.update(state='complete' if p.returncode==0 else 'failed',returncode=p.returncode,finished_unix=time.time())
    (ROOT/'progress.json').write_text(json.dumps(progress,indent=2))
    print('END',log.name,p.returncode,flush=True)
    if p.returncode:raise RuntimeError(f'{log}: exit {p.returncode}')
def eval_args(checkpoint,condition,seed,output):
    return ['python','-u','scripts/locomotion/eval_cmd.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','128','--seed',str(seed),'--checkpoint',str(checkpoint),'--experiment_profile','robust08','--with_pushes','--issue5_condition',condition,'--mass_randomization','bounded','--extended_commands','--command_filter','forward_03,forward_05,forward_08','--paired_resets','--output',str(output)]
for profile in ['legacy','bounded']:
    for seed in [1,2]:
        output=ROOT/f'mass_audit_{profile}_seed{seed}.json'
        if not output.exists():
            args=eval_args(PARENT,'combined',seed,output)
            args[args.index('--num_envs')+1]='4096'
            args[args.index('--mass_randomization')+1]=profile
            args+=['--mass_audit_output',str(output)]
            run(args,output.with_suffix('.log'))
        data=json.loads(output.read_text())
        assert data['num_envs']==4096
        assert data['friction_static']==[.4,1.4] and data['friction_dynamic']==[.3,1.2]
        if profile=='bounded':
            for body in data['bodies']:
                assert body['minimum_nominal_ratio']>=.5-1.e-6
                assert body['nominal_min_kg']==body['nominal_max_kg']
                assert body['applied_max_kg']<=1.5*body['nominal_min_kg']+1.e-7
                assert body['at_safety_clamp_count']==body['requested_at_or_below_clamp_count']==body['below_1pct_nominal_count']==0
(ROOT/'mass_audit_complete').write_text('complete')
def train_args(name,seed,count,envs):
    return ['python','-u','scripts/locomotion/train.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs',str(envs),'--max_iterations',str(count),'--run_name','issue6_20261109_'+name,'--seed',str(seed),'--resume','--load_run',PARENT.parent.name,'--checkpoint',PARENT.name,'--experiment_profile','robust08','--issue5_training_mode','strengthen','--issue5_factor','push','--mass_randomization','bounded']
if not (ROOT/'smoke_complete').exists():
    run(train_args('smoke',1,2,64),ROOT/'smoke.log')
    (ROOT/'smoke_complete').write_text('complete')
state_path=ROOT/'training_state.json'
state=json.loads(state_path.read_text()) if state_path.exists() else {}
for seed in [1,2]:
    name=f'bounded_strengthen_s{seed}'
    if name not in state:
        run_name='issue6_20261109_'+name
        existing=list((REPO/'logs/rsl_rl/tw_stage1').glob('*_'+run_name))
        if existing:raise RuntimeError(f'Incomplete prior run: {existing}; requires explicit recovery')
        run(train_args(name,seed,1500,4096),ROOT/f'{name}_train.log')
        paths=list((REPO/'logs/rsl_rl/tw_stage1').glob('*_'+run_name));assert len(paths)==1
        checkpoint=max(paths[0].glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
        state[name]={'seed':seed,'checkpoint':str(checkpoint),'run_dir':str(paths[0]),'parent_sha256':PARENT_SHA,'iterations':1500,'num_envs':4096,'mass_randomization':'bounded','training_push':.45}
        state_path.write_text(json.dumps(state,indent=2))
    for condition in ['combined','material']:
        for eval_seed in [1,2,3]:
            output=ROOT/f'{name}_{condition}_eval{eval_seed}.json'
            if not output.exists():run(eval_args(state[name]['checkpoint'],condition,eval_seed,output),output.with_suffix('.log'))
            json.loads(output.read_text())
prior=json.loads((REPO/'validation_artifacts/stage1_issue5_20261109/training_state.json').read_text())
baseline=REPO/'validation_artifacts/stage1_issues_20261108/models/robust08/model_8995.pt'
for seed in [1,2]:
    for model,condition,checkpoint in [
        (f'old_robust08_s{seed}','material',baseline if seed==1 else Path(prior['robust08_s2']['checkpoint'])),
        (f'old_strengthen_s{seed}','combined',Path(prior[f'strengthen_s{seed}']['checkpoint']))]:
        for eval_seed in [1,2,3]:
            output=ROOT/f'{model}_{condition}_eval{eval_seed}.json'
            if not output.exists():run(eval_args(checkpoint,condition,eval_seed,output),output.with_suffix('.log'))
            json.loads(output.read_text())
(ROOT/'quantitative_complete').write_text('complete')
videos=[]
for name,entry in state.items():
    rows=[x for seed in [1,2,3] for x in json.loads((ROOT/f'{name}_combined_eval{seed}.json').read_text())['results']]
    commands=['forward_03','forward_05','forward_08']
    worst=max(commands,key=lambda c:sum(x['failed_trials'] for x in rows if x['command']==c))
    videos.append({'model':name,'worst_command':worst,'checkpoint':entry['checkpoint']})
    for view in ['front','side']:
        output=ROOT/f'{name}_worst_{view}.json'
        if output.exists():continue
        args=eval_args(entry['checkpoint'],'combined',1,output)
        args[args.index('--num_envs')+1]='1';args[args.index('--command_filter')+1]=worst
        args+=['--video','--camera_view',view]
        run(args,output.with_suffix('.log'))
(ROOT/'video_plan.json').write_text(json.dumps(videos,indent=2))
(ROOT/'all_training_evaluation_video_complete').write_text('complete')
print('ISSUE6_ALL_JOBS_COMPLETE',flush=True)
