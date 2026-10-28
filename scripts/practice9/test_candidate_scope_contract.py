from pathlib import Path
import json,copy,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
from candidate_action_scope import make_scope_contract
from candidate_scope_contract import add_training_scope,split_training_scope,validate_scope_contract,validate_scope_jobs,require_scope_action_names,validate_scope_resume
root=Path('/root/gpufree-data/datasets/practice9/training_speed_blend_20261026');cp=Path((root/'actual_formal/final_checkpoint').read_text().strip());base=json.loads((cp.parent/'params/training_speed_blend.json').read_text());names=json.loads((cp.parent/'params/control_runtime.json').read_text())['joint_names'];before=json.dumps(base,sort_keys=True)
all_scope=make_scope_contract(names,'all');lower=make_scope_contract(names,'lower_body_waist');scoped=add_training_scope(base,names,'lower_body_waist')
assert add_training_scope(base,names,'all') is base and json.dumps(base,sort_keys=True)==before
split,scope=split_training_scope(json.loads(json.dumps(scoped)));assert split==base and scope==lower
assert split_training_scope(base)==(base,None)
assert validate_scope_jobs(base,[{}])==(base,None)
assert validate_scope_jobs(base,[{'candidate_action_scope':'all'}])==(base,None)
assert validate_scope_jobs(scoped,[{'candidate_action_scope':'lower_body_waist'}])==(base,lower)
assert require_scope_action_names(scoped,names)==lower and require_scope_action_names(base,None) is None
assert validate_scope_resume(None,all_scope)['legacy_initialization']
assert validate_scope_resume(None,lower,True)['legacy_initialization']
assert not validate_scope_resume(lower,lower)['legacy_initialization']
def reject(fn):
 try:fn()
 except ValueError:return
 raise AssertionError('invalid scope compatibility accepted')
reject(lambda:validate_scope_jobs(base,[{'candidate_action_scope':'lower_body_waist'}]))
reject(lambda:validate_scope_jobs(scoped,[{}]))
reject(lambda:validate_scope_jobs(scoped,[{'candidate_action_scope':'all'}]))
reject(lambda:validate_scope_jobs(scoped,[]))
reject(lambda:require_scope_action_names(scoped,list(reversed(names))))
reject(lambda:validate_scope_resume(None,lower))
reject(lambda:validate_scope_resume(lower,all_scope))
reject(lambda:validate_scope_resume(lower,make_scope_contract(list(reversed(names)),'lower_body_waist')))
for mutation in ['candidate_joint_names','teacher_joint_names','version','extra']:
 bad=copy.deepcopy(lower)
 if mutation in ['candidate_joint_names','teacher_joint_names']:bad[mutation]=list(reversed(bad[mutation]))
 elif mutation=='version':bad[mutation]=2
 else:bad['extra']=True
 reject(lambda:validate_scope_contract(bad))
reject(lambda:split_training_scope(dict(base,candidate_action_scope=lower)))
reject(lambda:split_training_scope(dict(scoped,version=3)))
reject(lambda:add_training_scope(scoped,names,'lower_body_waist'))
reject(lambda:make_scope_contract([[]]+names[1:],'lower_body_waist'))
reject(lambda:make_scope_contract(None,'lower_body_waist'))
print('Legacy contract identity/JSON unchanged; version2 partition roundtrip and delegated base exact')
print('Replay default/scope mismatch, runtime joint order, silent resume, malformed membership/version rejection passed')
print('Pure schema checks only; no live integration or simulator quality claim')
