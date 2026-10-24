"""Build a separate zero-range head articulation without changing mass/inertia."""
from pathlib import Path
import json
from pxr import Usd, UsdPhysics
SOURCE=Path('/root/gpufree-data/datasets/practice9/custom_robot/usd/urdf0711_training_30dof_tw123_yaw_waist_v3.usd')
TARGET=Path('/root/gpufree-data/datasets/practice9/head_locked_20261024/robot.usd')
NAMES=('neck','neck_linkage_roll','head_pitch')
if __name__=='__main__':
    assert not TARGET.exists(), 'Refusing to overwrite an experiment asset'
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    source=Usd.Stage.Open(str(SOURCE));source.Flatten().Export(str(TARGET))
    stage=Usd.Stage.Open(str(TARGET));changed=[]
    for prim in stage.Traverse():
        if prim.GetName() in NAMES and prim.IsA(UsdPhysics.RevoluteJoint):
            joint=UsdPhysics.RevoluteJoint(prim)
            joint.GetLowerLimitAttr().Set(0.);joint.GetUpperLimitAttr().Set(0.)
            changed.append(prim.GetName())
    assert sorted(changed)==sorted(NAMES)
    stage.GetRootLayer().Save()
    differences=[]
    for prim in source.Traverse():
        other=stage.GetPrimAtPath(prim.GetPath())
        for attr in prim.GetAttributes():
            if str(attr.Get())!=str(other.GetAttribute(attr.GetName()).Get()):
                differences.append(str(attr.GetPath()))
        for rel in prim.GetRelationships():
            assert rel.GetTargets()==other.GetRelationship(rel.GetName()).GetTargets()
    assert len(differences)==6 and all(x.endswith(('physics:lowerLimit','physics:upperLimit')) for x in differences)
    TARGET.with_suffix('.changes.json').write_text(json.dumps(differences,indent=2))
