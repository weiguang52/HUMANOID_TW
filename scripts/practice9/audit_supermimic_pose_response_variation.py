"""Read-only fit/validation response variation; no policy or model fitting."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--audit',type=Path,required=True)
p.add_argument('--collection-plan',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
d=json.loads(a.audit.read_text());plan=json.loads(a.collection_plan.read_text())
assert hashlib.sha256(a.collection_plan.read_bytes()).hexdigest()==d['collection_plan_sha256']
scale=np.array(plan['base_delta'])
reference=d['models'][0]['conditions']
def matrix(c,key):
 return np.array(c[key]['root_velocity_by_root_pose_output_by_input'])*scale[None,:]
fit=[c for c in reference if c['split']=='fit']
constant=np.mean([matrix(c,'actual') for c in fit],axis=0)
rows=[]
for model in d['models']:
 for split in ['fit','validation']:
  cs=[c for c in model['conditions'] if c['split']==split]
  actual=np.stack([matrix(c,'actual') for c in cs])
  pred=np.stack([matrix(c,'predicted') for c in cs])
  for axis in range(6):
   for label,sl in [('linear',slice(0,3)),('angular',slice(3,6))]:
    x=actual[:,sl,axis];y=pred[:,sl,axis];b=constant[sl,axis]
    rows.append(dict(model_sha256=model['sha256'],split=split,axis=axis,output_group=label,
     native_condition_std=float(np.sqrt(np.mean((x-x.mean(0))**2))),
     model_condition_std=float(np.sqrt(np.mean((y-y.mean(0))**2))),
     model_rmse=float(np.sqrt(np.mean((y-x)**2))),
     fit_constant_response_rmse=float(np.sqrt(np.mean((b-x)**2)))))
out=dict(audit_sha256=hashlib.sha256(a.audit.read_bytes()).hexdigest(),rows=rows,
 scope='Fixed fit-mean response diagnostic; no parameter update, no new acceptance threshold, no inference deployment. Input responses scaled by declared base_delta; output units m/s or rad/s.')
a.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps([v for v in rows if v['split']=='validation' and v['axis']==2 and v['output_group']=='angular']))
