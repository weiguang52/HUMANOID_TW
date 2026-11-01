"""Versioned root-Z-only diagnostic; never overwrite references or select samples."""
import argparse,hashlib,itertools,json
from pathlib import Path
import mujoco,numpy as np
from contact_labels import from_features,resample
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--asset',type=Path,required=True);p.add_argument('--human-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists();a.output.mkdir(parents=True);plan=json.loads(a.plan.read_text());assert plan['variant']=='bounded_root_z_projection_v2'
m=mujoco.MjModel.from_xml_path(str(a.asset));d=mujoco.MjData(m);corners=np.array(list(itertools.product([-1,1],repeat=3)));geoms=[i for i in range(m.ngeom) if '_collision' in m.geom(i).name];feet=[m.geom(s+'_foot_collision').id for s in ['left','right']];reports=[];motions=[]
def minima():
 out=[]
 for g in geoms:
  rot=d.geom_xmat[g].reshape(3,3);pos=d.geom_xpos[g]
  if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:height=pos[2]-abs(rot[2])@m.geom_size[g]
  elif m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
   k=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]];height=(v@rot.T+pos)[:,2].min()
  else:raise AssertionError('Unhandled collision shape')
  out.append(height)
 return np.array(out)
for item in plan['motions']:
 f=Path(item['file']);assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256']
 with np.load(f,allow_pickle=False) as z:src={k:z[k].copy() for k in z.files}
 n=len(src['joint_pos']);fps=float(src['fps'].item());assert fps==50
 feature=a.human_root/'new_joint_vecs'/(item['id']+'.npy');human=a.human_root/'new_joints'/(item['id']+'.npy');features=np.load(feature);joints=np.load(human);cc,kk,_,raw=from_features(features,joints);cc,kk=resample(cc,kk,n);known=cc.astype(bool)&kk.astype(bool)
 rawfoot=np.stack([raw[:,:2].any(1),raw[:,2:].any(1)],1);t=np.linspace(0,len(raw)-1,n);mask=rawfoot[np.floor(t).astype(int)]&rawfoot[np.ceil(t).astype(int)]
 adr=[int(m.joint(str(name)).qposadr[0]) for name in src['joint_names']];bids=[m.body(str(name)).id for name in src['body_names']];heights=[];low=[];fk=0.
 for i in range(n):
  d.qpos[:3]=src['root_pos'][i];d.qpos[3:7]=src['root_quat_wxyz'][i];d.qpos[adr]=src['joint_pos'][i];mujoco.mj_kinematics(m,d)
  heights.append([float(d.geom_xpos[g,2]-abs(d.geom_xmat[g].reshape(3,3)[2])@m.geom_size[g]) for g in feet]);low.append(minima().min());fk=max(fk,float(np.linalg.norm(d.xpos[bids]-src['body_pos_w'][i],axis=1).max()))
 heights=np.array(heights);active=mask.any(1);delta=np.zeros(n);marked=np.where(mask,heights,np.inf).min(1);delta[active]=np.maximum(.001-marked[active],.0002-heights[active].min(1));newheight=heights+delta[:,None]
 stance=float(np.quantile(abs((newheight-.001)[known]),.95));all_min=float((np.array(low)+delta).min());assert fk<1e-5 and stance<=.003 and all_min>=-.0005 and abs(delta).max()<=.031
 out={k:v.copy() for k,v in src.items()};out['root_pos'][:,2]+=delta;out['body_pos_w'][:,:,2]+=delta[:,None];dv=np.gradient(delta,1/fps);out['body_lin_vel_w'][:,:,2]+=dv[:,None]
 for k in src:
  if k not in ['root_pos','body_pos_w','body_lin_vel_w']:assert np.array_equal(src[k],out[k])
 for k in ['root_pos','body_pos_w','body_lin_vel_w']:assert np.array_equal(src[k][...,:2],out[k][...,:2])
 assert all(np.isfinite(v).all() for v in out.values() if np.issubdtype(v.dtype,np.number))
 target=a.output/(item['id']+'.npz');np.savez_compressed(target,**out);sha=hashlib.sha256(target.read_bytes()).hexdigest();motions.append(dict(item,file=str(target.resolve()),sha256=sha))
 reports.append(dict(id=item['id'],source_sha256=item['sha256'],output_sha256=sha,features_sha256=hashlib.sha256(feature.read_bytes()).hexdigest(),human_sha256=hashlib.sha256(human.read_bytes()).hexdigest(),frames=n,source_fk_max_m=fk,stance_error_p95_m=stance,all_collision_min_m=all_min,root_shift_max_m=float(abs(delta).max()),root_shift_m=delta.tolist(),joint_and_orientation_arrays_unchanged=True))
(a.output/'audit.json').write_text(json.dumps(dict(passed=True,scope='Artifact geometry/invariants only; no physical or policy acceptance.',asset_sha256=hashlib.sha256(a.asset.read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),rows=reports),indent=2)+'\n')
original=json.loads((a.plan.parent/'plan.json').read_text());original.update(motions=motions,command_cache_contract='current_state_current_frame_v1');(a.output/'collection_plan.json').write_text(json.dumps(original,indent=2)+'\n');print('generated',len(reports),'versioned diagnostic motions')
