from pathlib import Path
import json,hashlib,ast
R=Path('/root/gpufree-data/tw_stage1/issue16_20261111');A=Path('validation_artifacts/stage1_issue15_20261111')
s=Path('scripts/locomotion/issue15_common.py').read_text().replace('issue15','issue16').replace('60a8cc25ccce3a8ae22fcf7e5dd6f8d53db7613b','770014ee9a4393da6dff34eced89a04d83e788dd')
s=s.replace('legacy=False,plain=False):','legacy=False,plain=False,mode="mirror",capture_command="forward_05"):')
s=s.replace('str(capture),"--issue16_capture_command","forward_03"','str(capture),"--issue16_capture_command",capture_command')
s=s.replace(' return args',' if not plain:\n  assert observer\n  args+=["--issue16_mode",mode]\n return args')
p=Path('scripts/locomotion/issue16_common.py');compile(s,str(p),'exec');p.write_text(s)
old=json.loads((A/'cohort.json').read_text());wanted=['robust05','requested4500','control6000'];nodes={n:old['nodes'][n] for n in wanted};models={n:old['models'][n] for names in nodes.values() for n in names}
c=dict(base_commit='770014ee9a4393da6dff34eced89a04d83e788dd',nodes=nodes,models=models,formal_mirror_jobs=33,formal_no_policy_jobs=1,formal_training_updates=0,source_issue15_cohort_sha256=hashlib.sha256((A/'cohort.json').read_bytes()).hexdigest(),baseline_archive=str(A.resolve()),models_evaluated_without_checkpoint_selection=True)
(R/'cohort.json').write_text(json.dumps(c,indent=2)+'\n')
print('ISSUE16_COHORT',len(models),'COMMON_READY')
