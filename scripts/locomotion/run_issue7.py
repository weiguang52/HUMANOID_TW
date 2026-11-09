"""Issue #7: high-resolution observational evaluation only."""
import json,subprocess,time,hashlib,os
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue7_20261109')
REPO=Path('/root/gpufree-data/projects/HUMANOID_TW');os.chdir(REPO)
assert json.loads((ROOT/'instrumentation_parity.json').read_text())['recording_parity']=='passed'
assert '\nOK\n' in (ROOT/'interval_tests.log').read_text()
models={f'{prefix}_s{seed}':REPO/f'validation_artifacts/{bundle}/models/{name}_s{seed}/model_8995.pt' for prefix,bundle,name in [('new_bounded','stage1_issue6_20261109','bounded_strengthen'),('old_strengthen','stage1_issue5_20261109','strengthen')] for seed in [1,2]}
(ROOT/'input_models.json').write_text(json.dumps({n:{'checkpoint':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in models.items()},indent=2))
for name,checkpoint in models.items():
    for seed in [1,2,3]:
        if (ROOT/'stop').exists():raise RuntimeError('stop requested')
        output=ROOT/f'{name}_eval{seed}.json'
        if output.exists():json.loads(output.read_text());continue
        args=['python','-u','scripts/locomotion/eval_cmd.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1024','--seed',str(seed),'--checkpoint',str(checkpoint),'--experiment_profile','robust08','--with_pushes','--issue5_condition','combined','--mass_randomization','bounded','--extended_commands','--command_filter','forward_03,forward_05,forward_08','--paired_resets','--failure_traces',str(ROOT/'traces'/f'{name}_eval{seed}'),'--output',str(output)]
        log=output.with_suffix('.log')
        with log.open('w') as out:
            p=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue7',*args],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
            progress={'pid':p.pid,'log':str(log),'state':'running','args':args,'started_unix':time.time()}
            (ROOT/'progress.json').write_text(json.dumps(progress,indent=2));print('START',log.name,flush=True);p.wait()
        progress.update(state='complete' if p.returncode==0 else 'failed',returncode=p.returncode,finished_unix=time.time())
        (ROOT/'progress.json').write_text(json.dumps(progress,indent=2));print('END',log.name,p.returncode,flush=True)
        if p.returncode:raise RuntimeError(log)
        d=json.loads(output.read_text())
        assert len(d['results'])==3
        for row in d['results']:
            m=json.loads((ROOT/'traces'/f'{name}_eval{seed}'/(row['command']+'.json')).read_text())
            assert m['failed_trials']==row['failed_trials']==len(m['records'])
(ROOT/'high_resolution_complete').write_text('complete')
print('ISSUE7_HIGH_RESOLUTION_COMPLETE',flush=True)
