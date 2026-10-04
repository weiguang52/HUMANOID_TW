"""Keep completed replays on validation restart; reject incomplete artifacts."""
import sys,json
from pathlib import Path
p=Path(sys.argv[1]);jobs=json.loads(p.read_text());remaining=[]
for job in jobs:
    out=Path(job['output'])
    if out.exists():
        json.loads(out.read_text())
        assert out.with_suffix('.npz').is_file(),out
    else:
        remaining.append(job)
q=Path(str(p)+'.remaining')
q.write_text(json.dumps(remaining,indent=2))
Path(str(q)+'.done').unlink(missing_ok=True)
# Empty remainder is complete; shell skips the batch evaluator.
if not remaining:
    Path(str(q)+'.done').write_text('all prior replays exist\n')
