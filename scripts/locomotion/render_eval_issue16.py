"""Reuse the frozen evaluator's precolored asset loader for exact-state playback only."""
import hashlib,subprocess,sys
from pathlib import Path
BASE="770014ee9a4393da6dff34eced89a04d83e788dd"
p=Path(__file__).with_name("eval_cmd.py");source=p.read_text()
assert p.read_bytes()==subprocess.check_output(["git","show",BASE+":scripts/locomotion/eval_cmd.py"])
old="from replay_failure import replay";assert source.count(old)==1
source=source.replace(old,"from replay_issue16 import replay")
assert "--trajectory_replay" in sys.argv and "--video" in sys.argv
assert source.count('choices=["front","side"]')==1
source=source.replace('choices=["front","side"]','choices=["front","side","overhead"]')
sys.argv[0]=str(p)
exec(compile(source,str(p),"exec"),{"__name__":"__main__","__file__":str(p)})
