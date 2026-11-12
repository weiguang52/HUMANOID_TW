"""Run the exact clean evaluator only through environment creation, then read parameters and return."""
import difflib,hashlib,subprocess,sys
from pathlib import Path
BASE='770014ee9a4393da6dff34eced89a04d83e788dd'
original=Path(__file__).with_name('eval_cmd.py');source=original.read_text()
assert original.read_bytes()==subprocess.check_output(['git','show',BASE+':scripts/locomotion/eval_cmd.py'])
original_sha=hashlib.sha256(original.read_bytes()).hexdigest()
before='    runner.load(resume_path)'
after='    from issue16_runtime_readback import collect\n    collect(env.unwrapped,_ORIGINAL_SHA,_PATCHED_SHA)\n    env.close()\n    return'
assert source.count(before)==1;source=source.replace(before,after)
patched_sha=hashlib.sha256(source.encode()).hexdigest()
root=Path('/root/gpufree-data/tw_stage1/issue16_20261111')
(root/'runtime_zero_step_adapter.patch').write_text(''.join(difflib.unified_diff(original.read_text().splitlines(True),source.splitlines(True),fromfile='original_eval_cmd.py',tofile='zero_step_readback.py')))
sys.argv[0]=str(original)
exec(compile(source,str(original),'exec'),{'__name__':'__main__','__file__':str(original),'_ORIGINAL_SHA':original_sha,'_PATCHED_SHA':patched_sha})
