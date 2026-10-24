"""Optional, visual-only colors for URDF child links during replay."""
import colorsys
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def apply(stage, output, robot_roots=None):
    from pxr import Gf, Usd, UsdShade
    urdf = Path('/root/gpufree-data/datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf')
    joints = [j for j in ET.parse(urdf).getroot().findall('joint') if j.get('type') != 'fixed']
    palette = {}
    for i, joint in enumerate(joints):
        rgb = colorsys.hsv_to_rgb((i * .61803398875) % 1, .72, .95)
        link = joint.find('child').get('link')
        palette[link] = dict(joint=joint.get('name'), link=link, rgb=rgb,
                             hex='#' + ''.join(f'{round(c*255):02x}' for c in rgb), bindings=0)
    roots = robot_roots if robot_roots is not None else [p for p in stage.Traverse() if p.GetName() == 'Robot' and '/envs/' in str(p.GetPath())]
    assert roots, 'Robot render root missing'
    materials = {}
    for link, entry in palette.items():
        material = UsdShade.Material.Define(stage, str(roots[0].GetPath()) + '/JointColorMaterials/mat_' + link)
        shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + '/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        from pxr import Sdf
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*entry['rgb']))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.65)
        shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        materials[link] = material
    for root in roots:
        for prim in Usd.PrimRange(root):
            link = prim.GetName()
            if link not in palette or prim.IsInstanceProxy():
                continue
            # Bind at the link root, without expanding USD/PhysX instances.
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[link], bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            palette[link]['bindings'] += 1
    assert all(v['bindings'] > 0 for v in palette.values()), 'Some joint links were not colored'
    Path(output).write_text(json.dumps(list(palette.values()), indent=2))
    print('JOINT_COLORS', {k: v['bindings'] for k, v in palette.items()}, flush=True)
