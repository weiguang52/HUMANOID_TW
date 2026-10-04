"""Reproduce target saturation diagnosis from existing validation telemetry."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
u=ET.parse('/root/gpufree-data/datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf')
limits={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper')))
    for j in u.findall('joint') if j.find('limit') is not None and j.find('limit').get('lower') is not None}
root=Path('/root/gpufree-data/datasets/practice9/tw100_expression_v1/validation')
rows=[]
for variant in ('control','expression'):
    for p in sorted((root/variant).glob('*.seed*.npz')):
        with np.load(p) as d:
            for i,name in enumerate(d['joint_names']):
                q=d['q'][:,i];t=d['target_q'][:,i];lo,hi=limits[name]
                rows.append(dict(variant=variant,replay=p.stem,joint=str(name),
                    limits=[lo,hi],actual_range=[float(q.min()),float(q.max())],
                    target_range=[float(t.min()),float(t.max())],
                    target_outside_fraction=float(np.mean((t<lo)|(t>hi))),
                    target_error_rms_rad=float(np.sqrt(np.mean((t-q)**2)))))
print(json.dumps(dict(note='All recorded samples including reset boundaries; deterministic target-limit audit. Implicit-actuator torque is estimated, not measured.',rows=rows),indent=2))
