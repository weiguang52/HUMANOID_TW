"""Freeze source-family development splits and audit nominal-pose PD offsets."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from supermimic.config.tw.robot import ACTION_SCALE

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--train',type=Path,required=True)
p.add_argument('--holdout',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert not a.output.exists(), 'Preserve frozen split'
train=json.loads(a.train.read_text());test=json.loads(a.holdout.read_text())
families=sorted({r['source_family'] for r in train['motions']},key=lambda s:hashlib.sha256(('tw-wm-calibration-v1:'+s).encode()).hexdigest())
assert not set(families)&{r['source_family'] for r in test['motions']}
validation=set(families[:max(1,len(families)//5)])
a.output.mkdir(parents=True)
for split in ('fit','validation'):
 rows=[r for r in train['motions'] if (r['source_family'] in validation)==(split=='validation')]
 manifest=dict(train,motions=rows,calibration_split=split,prior_policy_exposure=True)
 (a.output/f'{split}.json').write_text(json.dumps(manifest,indent=2)+'\n')
all_offsets=[];names=None
for row in train['motions']:
 with np.load(row['file']) as d:
  current=d['joint_names'].tolist()
  if names is None:names=current
  assert current==names
  nominal=np.array([-math.pi/2 if n=='left_shoulder_roll_joint' else math.pi/2 if n=='right_shoulder_roll_joint' else 0 for n in names])
  scale=np.array([ACTION_SCALE[n] for n in names])
  all_offsets.append((d['joint_pos']-nominal)/scale)
x=np.concatenate(all_offsets)
report=dict(train_manifest_sha256=hashlib.sha256(a.train.read_bytes()).hexdigest(),families=len(families),validation_families=len(validation),frames=len(x),joint_stats=[dict(name=n,offset_min=float(x[:,i].min()),offset_max=float(x[:,i].max()),absolute_p95=float(np.quantile(np.abs(x[:,i]),.95))) for i,n in enumerate(names)],scope='Inverse nominal-pose PD offsets, not inverse dynamics or recommended action limits. Fit/validation source families disjoint and disjoint from existing final holdout. Both development partitions were seen by the old1000-iteration policy; not a fresh generalization test. New model fitting must exclude validation trajectories.')
(a.output/'action_offset_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='joint_stats'},indent=2))
print(sorted(report['joint_stats'],key=lambda r:r['absolute_p95'],reverse=True)[:6])
