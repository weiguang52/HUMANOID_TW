"""Frozen-budget training and evaluation following issue #5 factor attribution."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue5_20261109')
REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
os.chdir(REPO)
FACTORS=['push','material','delay','noise']
CONDITIONS=['clean',*FACTORS,'combined']

def run(args,log):
    print('START',log.name,flush=True)
    with log.open('w') as out:
        p=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue5',*args],stdout=out,stderr=subprocess.STDOUT)
        progress={'pid':p.pid,'log':str(log),'args':args,'state':'running','started_unix':time.time()}
        (ROOT/'progress.json').write_text(json.dumps(progress,indent=2));p.wait()
    progress.update(state='complete' if p.returncode==0 else 'failed',returncode=p.returncode,finished_unix=time.time())
    (ROOT/'progress.json').write_text(json.dumps(progress,indent=2))
    print('END',log.name,p.returncode,flush=True)
    if p.returncode:raise RuntimeError(f'{log}: exit {p.returncode}')

while not (ROOT/'factor_evaluation_complete').exists():
    if (ROOT/'stop').exists():raise RuntimeError('stop requested')
    time.sleep(20)

plan_path=ROOT/'training_plan.json'
if not plan_path.exists():
    scores={};by_seed={}
    for condition in CONDITIONS:
        by_seed[condition]=[]
        for seed in [1,2,3]:
            d=json.loads((ROOT/f'robust08_s1_{condition}_eval{seed}.json').read_text())
            assert len(d['results'])==3 and all(x['num_envs']==128 and x['seconds']==20 for x in d['results'])
            by_seed[condition].append(sum(x['failed_trials'] for x in d['results'])/(3*128))
        scores[condition]=sum(by_seed[condition])/3
    ordered=sorted(FACTORS,key=lambda k:(-scores[k],FACTORS.index(k)))
    factor=ordered[0];excess=scores[factor]-scores['clean'];full=scores['combined']-scores['clean']
    stable=all(factor==sorted(FACTORS,key=lambda k:(-by_seed[k][i],FACTORS.index(k)))[0] for i in range(3))
    dominant=(full>0 and excess>=.05 and scores[factor]-scores[ordered[1]]>=.05 and excess>=.5*full and stable)
    attribution={'scores':scores,'scores_by_eval_seed':by_seed,'ranked_factors':ordered,'dominant_factor':factor if dominant else None,'training_hypothesis_factor':factor,'selection_rule':'frozen_plan.md: 5pp excess and margin, >=half combined excess, same top factor in all evaluation seeds','claim':'dominant isolated factor' if dominant else 'no single dominant factor; strongest isolated stressor is exploratory only'}
    (ROOT/'factor_attribution.json').write_text(json.dumps(attribution,indent=2))
    historical=json.loads(Path('/root/gpufree-data/tw_stage1/issue_runs_20261108/training_state.json').read_text())
    parent=Path(historical['robust05']['checkpoint']);assert parent.is_file()
    parent_sha=hashlib.sha256(parent.read_bytes()).hexdigest()
    runs=[{'name':'robust08_s2','mode':'legacy','seed':2}]
    runs += [{'name':f'{mode}_s{seed}','mode':mode,'seed':seed} for mode in ['remove','strengthen'] for seed in [1,2]]
    plan={'parent':str(parent),'parent_sha256':parent_sha,'factor':factor,'dominant':dominant,'iterations':1500,'num_envs':4096,'runs':runs,'prior_frozen_spec_sha256':json.loads((ROOT/'frozen_plan_sha256.json').read_text())['plan_sha256']}
    plan_path.write_text(json.dumps(plan,indent=2));print('ATTRIBUTION',json.dumps(attribution),flush=True)
plan=json.loads(plan_path.read_text());parent=Path(plan['parent'])
assert hashlib.sha256(parent.read_bytes()).hexdigest()==plan['parent_sha256']

def train_args(name,mode,seed,count,envs):
    return ['python','-u','scripts/locomotion/train.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs',str(envs),'--max_iterations',str(count),'--run_name','issue5_20261109_'+name,'--seed',str(seed),'--resume','--load_run',parent.parent.name,'--checkpoint',parent.name,'--experiment_profile','robust08','--issue5_training_mode',mode,'--issue5_factor',plan['factor']]

for mode in ['remove','strengthen']:
    marker=ROOT/f'{mode}_smoke_complete'
    if not marker.exists():
        run(train_args('smoke_'+mode,mode,1,2,64),ROOT/(mode+'_smoke.log'));marker.write_text('complete')

state_path=ROOT/'training_state.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}

def evaluate(checkpoint,name,conditions):
    for condition in conditions:
        for seed in [1,2,3]:
            output=ROOT/f'{name}_{condition}_eval{seed}.json'
            if output.exists():json.loads(output.read_text());continue
            args=['python','-u','scripts/locomotion/eval_cmd.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','128','--seed',str(seed),'--checkpoint',str(checkpoint),'--experiment_profile','robust08','--with_pushes','--issue5_condition',condition,'--extended_commands','--command_filter','forward_03,forward_05,forward_08','--paired_resets','--output',str(output)]
            run(args,output.with_suffix('.log'));json.loads(output.read_text())

for item in plan['runs']:
    if (ROOT/'stop').exists():raise RuntimeError('stop requested')
    name=item['name']
    if name not in state:
        run_name='issue5_20261109_'+name
        existing=list((REPO/'logs/rsl_rl/tw_stage1').glob('*_'+run_name))
        if existing:raise RuntimeError(f'Incomplete prior run requires explicit recovery: {existing}')
        run(train_args(name,item['mode'],item['seed'],1500,4096),ROOT/(name+'_train.log'))
        paths=list((REPO/'logs/rsl_rl/tw_stage1').glob('*_'+run_name));assert len(paths)==1
        checkpoint=max(paths[0].glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
        state[name]={**item,'checkpoint':str(checkpoint),'run_dir':str(paths[0]),'parent_sha256':plan['parent_sha256'],'iterations':1500,'factor':plan['factor']}
        state_path.write_text(json.dumps(state,indent=2))
    conditions=CONDITIONS if name=='robust08_s2' else ['clean',plan['factor'],'combined']
    evaluate(state[name]['checkpoint'],name,conditions)

baseline=REPO/'validation_artifacts/stage1_issues_20261108/models/robust08/model_8995.pt'
video_plan=[]
for name,checkpoint in [('robust08_s1',baseline),*[(n,Path(e['checkpoint'])) for n,e in state.items()]]:
    rows=[x for seed in [1,2,3] for x in json.loads((ROOT/f'{name}_combined_eval{seed}.json').read_text())['results']]
    commands=['forward_03','forward_05','forward_08']
    worst=max(commands,key=lambda c:sum(x['failed_trials'] for x in rows if x['command']==c))
    video_plan.append({'model':name,'checkpoint':str(checkpoint),'worst_command':worst})
    for view in ['front','side']:
        output=ROOT/f'{name}_worst_{view}.json'
        if output.exists():json.loads(output.read_text());continue
        args=['python','-u','scripts/locomotion/eval_cmd.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1','--seed','1','--checkpoint',str(checkpoint),'--experiment_profile','robust08','--with_pushes','--issue5_condition','combined','--extended_commands','--command_filter',worst,'--paired_resets','--video','--camera_view',view,'--output',str(output)]
        run(args,output.with_suffix('.log'));json.loads(output.read_text())
(ROOT/'video_plan.json').write_text(json.dumps(video_plan,indent=2))
(ROOT/'all_training_evaluation_video_complete').write_text('complete')
