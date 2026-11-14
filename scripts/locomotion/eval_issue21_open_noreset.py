"""Use frozen evaluation setup and runtime with open-only lifecycle wrapper."""
from pathlib import Path
p=Path(__file__).with_name('eval_issue21.py');s=p.read_text();old='from issue21_runtime import run';assert s.count(old)==1;s=s.replace(old,'from issue21_open_noreset import run');exec(compile(s,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
