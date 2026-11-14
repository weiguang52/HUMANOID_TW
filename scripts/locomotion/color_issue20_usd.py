"""Precolor the exact selected physical asset using a separate USD layer."""
import json
from pathlib import Path
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True);app=launcher.app
from pxr import Usd,UsdShade,Gf,Sdf,UsdGeom
from issue20_common import ROOT,REPO,sha,save
try:
 selected=json.loads((ROOT/'representative_selection.json').read_text());physics=Path(selected['physics_usd']);out=ROOT/('display_'+selected['usd']+'_colored.usd');assert not out.exists()
 stage=Usd.Stage.CreateNew(str(out));stage.GetRootLayer().subLayerPaths=[str(physics)];root=stage.GetPrimAtPath('/urdf0711_training_30dof');assert root;stage.SetDefaultPrim(root)
 legend=json.loads((REPO/'validation_artifacts/stage1_issue14_20261111/joint_colors.json').read_text());bindings=[]
 for row in legend:
  body=stage.GetPrimAtPath(str(root.GetPath())+'/'+row['link']);assert body
  material=UsdShade.Material.Define(stage,str(root.GetPath())+'/Issue20Colors/'+row['link']);shader=UsdShade.Shader.Define(stage,str(material.GetPath())+'/Shader');shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float,row['rgb'])));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.55);material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
  # Change bindings before any simulation initialization. Inherited physical data is untouched.
  for p in list(Usd.PrimRange(body)):
   if p.IsInstanceable() and '/visuals' in str(p.GetPath()):p.SetInstanceable(False)
  count=0
  for p in Usd.PrimRange(body):
   if p.IsA(UsdGeom.Gprim) and '/visuals' in str(p.GetPath()):UsdShade.MaterialBindingAPI.Apply(p).Bind(material);count+=1
  UsdShade.MaterialBindingAPI.Apply(body).Bind(material,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
  assert count>0,row['link'];bindings.append({**row,'bindings':count,'link_root_stronger_than_descendants':True})
 stage.GetRootLayer().Save();original=Usd.Stage.Open(str(physics));colored=Usd.Stage.Open(str(out));checked=0
 for p in original.Traverse(Usd.TraverseInstanceProxies()):
  q=colored.GetPrimAtPath(p.GetPath());assert q,p.GetPath()
  for a in p.GetAttributes():assert str(a.Get())==str(q.GetAttribute(a.GetName()).Get()),a.GetPath();checked+=1
  for rel in p.GetRelationships():
   if not rel.GetName().startswith('material:binding'):assert rel.GetTargets()==q.GetRelationship(rel.GetName()).GetTargets(),rel.GetPath()
 save('joint_colors.json',bindings);save('colored_asset_integrity.json',{'passed':True,'physics_usd':str(physics),'physics_sha256':sha(physics),'colored_usd':str(out),'colored_sha256':sha(out),'original_attributes_checked':checked,'materials_bound_before_simulation':True,'all_original_nonmaterial_relationships_equal':True});print('ISSUE20_PRECOLORED_ASSET_PASS',checked,flush=True)
except BaseException:
 import traceback;traceback.print_exc();raise
finally:app.close()
