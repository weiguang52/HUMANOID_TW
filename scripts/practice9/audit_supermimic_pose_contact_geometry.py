"""Read-only sole geometry alongside native pose-response conditions; no force inference."""
import argparse,hashlib,itertools,json
from pathlib import Path
import mujoco,numpy as np
p=argparse.ArgumentParser(description=__doc__)
for k in ['collection','asset','response-audit','output']:p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--ground-height',type=float,required=True,help='Explicit environment plane height; robot asset may exclude terrain')
a=p.parse_args();assert not a.output.exists()
integrity=json.loads((a.collection/'integrity.json').read_text());assert integrity['passed']
hashes={x['file']:x['sha256'] for x in integrity['files']}
mapping=json.loads((a.collection/'control_mapping.json').read_text())
m=mujoco.MjModel.from_xml_path(str(a.asset));d=mujoco.MjData(m)
ads=[int(m.joint(n).qposadr[0]) for n in mapping['state_joint_names']]
assert len(ads)==30 and len(set(ads))==30
feet=[m.geom(s+'_foot_collision').id for s in ['left','right']]
assert all(m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX for g in feet)
planes=[i for i in range(m.ngeom) if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE]
assert len(planes)<=1
ground=a.ground_height
if planes:
 pg=planes[0];assert np.allclose(m.geom_quat[pg],[1,0,0,0])
 assert abs(float(m.geom_pos[pg,2])-ground)<1e-9
signs=np.array(list(itertools.product([-1,1],repeat=3)))
rows=[]
for row in json.loads((a.collection/'results.json').read_text())['rows']:
 path=(a.collection/row['file']).resolve();assert path.is_relative_to(a.collection.resolve())
 assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[row['file']]
 with np.load(path,allow_pickle=False) as z:states=z['state'].copy()
 geometries=[]
 for s in states:
  d.qpos[:7]=s[:7];d.qpos[ads]=s[13:43];mujoco.mj_kinematics(m,d)
  heights=[]
  for g in feet:
   corners=(signs*m.geom_size[g])@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
   heights.append((corners[:,2]-ground).tolist())
  geometries.append(heights)
 rows.append(dict(split=row['split'],warmup=row['warmup'],kind=row['kind'],level=row.get('level'),assignment=row.get('assignment'),file=row['file'],sha256=hashes[row['file']],corner_height_m=geometries))
response=json.loads(a.response_audit.read_text())
assert response['collection_plan_sha256']==hashlib.sha256((a.collection/'plan.json').read_bytes()).hexdigest()
zero={(r['split'],r['warmup']):r for r in rows if r['kind']=='zero'}
summary=[]
for c in response['models'][0]['conditions']:
 h=np.array(zero[(c['split'],c['warmup'])]['corner_height_m'])
 summary.append(dict(split=c['split'],warmup=c['warmup'],level=c['level'],
  min_corner_height_mm=(h.min(axis=(0,2))*1000).tolist(),
  max_corner_height_mm=(h.max(axis=(0,2))*1000).tolist(),
  native_z_to_angular_response=c['per_input_axis'][2]['angular_response']['actual_response_norm']))
out=dict(explicit_environment_ground_height_m=ground,asset_sha256=hashlib.sha256(a.asset.read_bytes()).hexdigest(),response_audit_sha256=hashlib.sha256(a.response_audit.read_bytes()).hexdigest(),rows=rows,summary=summary,
 scope='All archived input poses, native CPU kinematics only. Corner distances are geometric; no normal force, loaded support, contact pressure or causal/stability claim. No dynamics integration or model fitting.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary))
