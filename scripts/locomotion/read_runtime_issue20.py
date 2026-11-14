"""Strict-model zero-step readback before policy loading or evaluation."""
from pathlib import Path
p=Path(__file__).with_name('eval_issue20.py');s=p.read_text();marker="marker='for old,new in replacements:'";assert marker in s
s=s.replace(marker,"extra.append(('    runner.load(resume_path)', '    from issue20_readback import collect\\n    collect(env.unwrapped,_ORIGINAL_SHA,_PATCHED_SHA)\\n    env.close()\\n    return'))\n"+marker);exec(compile(s,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
