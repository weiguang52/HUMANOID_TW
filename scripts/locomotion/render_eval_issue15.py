"""Reuse the frozen evaluator's precolored asset loader for exact-state playback only."""
import hashlib,subprocess,sys
from pathlib import Path
BASE="0ec49f358d1634b62b7f8b149381809b54d23b9b"
p=Path(__file__).with_name("eval_cmd.py");source=p.read_text()
assert p.read_bytes()==subprocess.check_output(["git","show",BASE+":scripts/locomotion/eval_cmd.py"])
old="from replay_failure import replay";assert source.count(old)==1
source=source.replace(old,"from replay_issue15 import replay")
assert "--trajectory_replay" in sys.argv and "--video" in sys.argv
sys.argv[0]=str(p)
exec(compile(source,str(p),"exec"),{"__name__":"__main__","__file__":str(p)})
