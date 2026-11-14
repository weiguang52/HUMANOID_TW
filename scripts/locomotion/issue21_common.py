"""Remote-only diagnostic provenance for issue21."""
from pathlib import Path
import json,hashlib
ROOT=Path('/root/tw_stage1/issue21_20261114');OLD=Path('/root/tw_stage1/issue20_20261114');REPO=Path(__file__).resolve().parents[2]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(n,d):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
