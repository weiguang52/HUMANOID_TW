"""Bake a separate colored USD offline, preserving original composed attributes."""
from pathlib import Path
import hashlib,json,sys
from pxr import Usd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
from joint_colors import apply
source=Path('/root/gpufree-data/datasets/practice9/custom_robot/usd/urdf0711_training_30dof_tw123_yaw_waist_v3.usd')
target=source.with_name(source.stem+'_colored.usd')
assert not target.exists(), f'Refusing to overwrite {target}'
original=Usd.Stage.Open(str(source));original.Flatten().Export(str(target))
colored=Usd.Stage.Open(str(target));apply(colored,str(target.with_suffix('.colors.json')),[colored.GetDefaultPrim()]);colored.GetRootLayer().Save()
checked=0
for prim in original.Traverse():
 other=colored.GetPrimAtPath(prim.GetPath());assert other
 for attr in prim.GetAttributes():
  assert str(attr.Get())==str(other.GetAttribute(attr.GetName()).Get()), str(attr.GetPath())
  checked+=1
 for rel in prim.GetRelationships():
  if rel.GetName().startswith('material:binding'):continue
  assert rel.GetTargets()==other.GetRelationship(rel.GetName()).GetTargets(),str(rel.GetPath())
audit=dict(source=str(source),target=str(target),original_attributes_unchanged=checked,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
target.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2));print(audit)
