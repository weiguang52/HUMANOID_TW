"""Fixed issue20 paths and paired-evaluation provenance."""
from pathlib import Path
import hashlib,json
ROOT=Path('/root/tw_stage1/issue20_20261114')
REPO=Path(__file__).resolve().parents[2]
OLD=Path('/root/gpufree-data/tw_stage1/issue19_20261113')
USD=Path('/root/gpufree-data/datasets/practice9/custom_robot/usd/urdf0711_training_30dof_tw123_yaw_waist_v3.usd')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
