"""Verify fresh smoke checkpoint and coordinate contract before formal training."""
import json
from pathlib import Path
import torch
S=Path('/root/gpufree-data/datasets/practice9/tw123_v1')
result={}
for variant in ('bounded','balanced'):
    root=S/f'smoke_{variant}'
    assert (root/'status').read_text().strip()=='trained'
    p=Path((root/'final_checkpoint').read_text().strip())
    checkpoint=torch.load(p,map_location='cpu',weights_only=False)
    assert all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values())
    runtime=json.loads((p.parent/'params/control_runtime.json').read_text())
    assert runtime['limit_target_position'] is True
    result[variant]={'checkpoint':str(p),'finite':True,'target_position_limited':True}
print(json.dumps(result,indent=2))
