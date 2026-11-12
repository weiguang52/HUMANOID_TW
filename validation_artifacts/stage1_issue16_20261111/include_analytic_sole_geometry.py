from pathlib import Path
p=Path('scripts/locomotion/audit_issue16_symmetry.py');s=p.read_text().replace('import csv,json,math,re','import csv,json,math,re,itertools')
s=s.replace("geometry_meta[name].append(dict(path=g['path'],points=","geometry_meta[name].append(dict(type='Mesh',path=g['path'],points=")
needle=" clouds={};hulls={}"
insert="""   elif g['type']=='Cube':
    size=float(g['geometry_attributes']['size'])
    local=np.array(list(itertools.product([-size/2,size/2],repeat=3)))
    matrix=np.array(g['world_matrix_row'])
    world=(np.c_[local,np.ones(8)]@matrix)[:,:3]
    points[name].append(world)
    geometry_meta[name].append(dict(type='Cube',path=g['path'],points=8,faces=6,indices=24,
     vertex_definition='Schema size-based eight corners transformed by the complete authored Xform. Authored extent attribute retained separately; not used as a substitute for Cube size.',
     size=size,authored_extent=g['geometry_attributes'].get('extent'),world_vertices=world.tolist()))
"""
assert needle in s;s=s.replace(needle,insert+needle)
s=s.replace("inertia_frobenius_relative_difference=float(np.linalg.norm(Ia-S@Ib@S)/np.linalg.norm(Ia)),","inertia_frobenius_relative_difference=float(np.linalg.norm(Ia-S@Ib@S)/np.linalg.norm(Ia)),\n   inertia_frobenius_absolute_difference_kg_m2=float(np.linalg.norm(Ia-S@Ib@S)),")
s=s.replace("authored_descendant_meshes=sum(len(v) for v in geometry_meta.values()),raw_authored_mesh_points=sum(g['points'] for v in geometry_meta.values() for g in v),","authored_descendant_meshes=sum(g['type']=='Mesh' for v in geometry_meta.values() for g in v),raw_authored_mesh_points=sum(g['points'] for v in geometry_meta.values() for g in v if g['type']=='Mesh'),\n  analytic_primitive_count=sum(g['type']=='Cube' for v in geometry_meta.values() for g in v),")
p.write_text(s);compile(s,str(p),'exec');print('STATIC_AUDIT_INCLUDES_ANALYTIC_FOOT_CUBES')
