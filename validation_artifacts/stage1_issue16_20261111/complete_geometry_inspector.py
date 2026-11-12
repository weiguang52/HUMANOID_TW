from pathlib import Path
p=Path('scripts/locomotion/inspect_issue16_usd.py');s=p.read_text();needle='   collisions.append(record)'
insert="""   record['descendant_geometry']=[]
   for child in Usd.PrimRange(p,Usd.TraverseInstanceProxies()):
    if child==p or not child.IsA(UsdGeom.Boundable):continue
    child_record={'path':str(child.GetPath()),'type':child.GetTypeName(),'attributes':attrs(child),'world_matrix_row':convert(cache.GetLocalToWorldTransform(child))}
    if child.IsA(UsdGeom.Mesh):
     mesh=UsdGeom.Mesh(child);points=np.array(mesh.GetPointsAttr().Get(),dtype='float64');ckey=key+'_desc'+str(len(record['descendant_geometry']))
     mat=cache.GetLocalToWorldTransform(child)
     geometry[ckey+'_points_local']=points;geometry[ckey+'_points_world']=np.array([list(mat.Transform(Gf.Vec3d(*v))) for v in points]);geometry[ckey+'_face_counts']=np.array(mesh.GetFaceVertexCountsAttr().Get());geometry[ckey+'_face_indices']=np.array(mesh.GetFaceVertexIndicesAttr().Get())
     child_record.update(array_key=ckey,point_count=len(points))
    else:child_record['geometry_attributes']={a.GetName():convert(a.Get()) for a in child.GetAttributes() if a.GetName() in ['size','radius','height','axis','extent']}
    record['descendant_geometry'].append(child_record)
"""
assert needle in s;s=s.replace(needle,insert+needle);compile(s,str(p),'exec');p.write_text(s)
print('USD_DESCENDANT_GEOMETRY_INSPECTION_ENABLED')
