"""Verify issue #9 repeats unchanged variant configurations across fixed seeds."""
from pathlib import Path
import json,yaml
ROOT=Path('/root/gpufree-data/tw_stage1/issue9_20261109')
def diff(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        for key in sorted(set(a)|set(b)):
            if key not in a or key not in b:yield path+'/'+key,a.get(key),b.get(key)
            else:yield from diff(a[key],b[key],path+'/'+key)
    elif a!=b:yield path,a,b
records={};paired={}
for seed in [3,4,5]:
    paths={}
    for variant,bundle,model in [('requested','stage1_issue6_20261109','bounded_strengthen_s1'),('applied','stage1_issue8_20261109','applied_s1')]:
        found=list(Path('logs/rsl_rl/tw_stage1').glob(f'*_issue9_20261109_{variant}_s{seed}'))
        if not found:continue
        assert len(found)==1;paths[variant]=found[0]
        record={}
        for kind in ['env','agent']:
            old=Path(f'validation_artifacts/{bundle}/models/{model}/params/{kind}.yaml');new=found[0]/f'params/{kind}.yaml'
            a=yaml.load(old.read_text(),Loader=yaml.BaseLoader);b=yaml.load(new.read_text(),Loader=yaml.BaseLoader)
            assert b['seed']==str(seed),(variant,seed,kind,b.get('seed'))
            changes=list(diff(a,b));allowed={'/seed','/log_dir'} if kind=='env' else {'/seed','/run_name'}
            assert all(x[0] in allowed for x in changes),(variant,seed,kind,changes)
            record[kind]={'baseline':str(old),'current':str(new),'differences':changes}
        records[f'{variant}_s{seed}']=record
    if len(paths)==2:
        record={}
        for kind in ['env','agent']:
            a=yaml.load((paths['requested']/f'params/{kind}.yaml').read_text(),Loader=yaml.BaseLoader)
            b=yaml.load((paths['applied']/f'params/{kind}.yaml').read_text(),Loader=yaml.BaseLoader)
            changes=list(diff(a,b));allowed={'/log_dir','/observations/policy/last_action/params/action_name'} if kind=='env' else {'/run_name'}
            assert all(x[0] in allowed for x in changes),(seed,kind,changes)
            record[kind]=changes
        paired[str(seed)]=record
(ROOT/'config_difference_audit.json').write_text(json.dumps({'runs':records,'matched_pairs':paired},indent=2))
print('PASS unchanged configurations;',len(records),'new runs;',len(paired),'new matched pairs',flush=True)
