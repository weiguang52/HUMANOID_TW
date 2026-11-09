"""Confirm only policy last_action differs from matching issue #6 training."""
from pathlib import Path
import json,yaml
ROOT=Path('/root/gpufree-data/tw_stage1/issue8_20261109')
def diff(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        for key in sorted(set(a)|set(b)):
            if key not in a or key not in b:yield path+'/'+key,a.get(key),b.get(key)
            else:yield from diff(a[key],b[key],path+'/'+key)
    elif a!=b:yield path,a,b
records={}
for seed in [1,2]:
    matches=list(Path('logs/rsl_rl/tw_stage1').glob(f'*_issue8_20261109_applied_s{seed}'))
    if not matches:continue
    assert len(matches)==1
    record={}
    for kind in ['env','agent']:
        old=Path(f'validation_artifacts/stage1_issue6_20261109/models/bounded_strengthen_s{seed}/params/{kind}.yaml')
        new=matches[0]/f'params/{kind}.yaml'
        a=yaml.load(old.read_text(),Loader=yaml.BaseLoader);b=yaml.load(new.read_text(),Loader=yaml.BaseLoader)
        changes=list(diff(a,b));print(seed,kind,json.dumps(changes))
        allowed={'/observations/policy/last_action/params/action_name','/log_dir'} if kind=='env' else {'/run_name'}
        assert all(x[0] in allowed for x in changes)
        record[kind]={'old':str(old),'new':str(new),'differences':changes}
    records[str(seed)]=record
(ROOT/'config_difference_audit.json').write_text(json.dumps(records,indent=2))
print('PASS only policy last_action parameter and run/log names differ;',len(records),'training seeds')
