"""Build one seeded validation batch without modifying the manifest."""
import sys,json
from pathlib import Path
manifest,out,seed=sys.argv[1:];out=Path(out)
m=json.load(open(manifest))
jobs=[dict(motion_id=i,steps=max(1000,x['frames']+1),output=str(out/(x['id']+'.seed'+seed+'.json'))) for i,x in enumerate(m['motions'])]
(out/('jobs'+seed+'.json')).write_text(json.dumps(jobs,indent=2)+'\n')
