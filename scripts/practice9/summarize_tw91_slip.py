import json
from pathlib import Path
from summarize_tw75_contact import summarize
root=Path('/root/gpufree-data/datasets/practice9/tw91_slip_v1')
report={'note':'Matched single-startup-per-eval-seed batch protocol. Six validation clips, not independent test acceptance.'}
for variant in ['control','slip']:
 for seed in [42,123]:
  p=root/variant/f'seed{seed}'
  if (p/'status').exists() and (p/'status').read_text().strip()=='completed':
   report[f'{variant}_seed{seed}']=summarize(p)
(root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
