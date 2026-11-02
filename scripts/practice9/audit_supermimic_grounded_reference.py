"""Independently audit exported diagnostic references; no dynamics acceptance."""
import argparse, hashlib, json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('plan','asset','output'): p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args(); assert not a.output.exists()
    plan=json.loads(a.plan.read_text()); parent=json.loads(Path(plan['parent_plan']).read_text())
    assert plan['diagnostic_only'] and not plan['training_export_approved'] and not plan['physical_acceptance']
    assert sha(plan['parent_plan'])==plan['parent_plan_sha256'] and sha(a.asset)==plan['asset_sha256']
    for path,h in plan['audit_sha256'].items(): assert sha(path)==h
    for key in ('phase_fractions','max_steps','seed','physics','command_cache_contract','train_manifest_sha256','fixed7_manifest_sha256'):
        assert plan[key]==parent[key]
    assert len(plan['motions'])==len(parent['motions'])==8
    model=mujoco.MjModel.from_xml_path(str(a.asset)); data=mujoco.MjData(model); rows=[]
    changed={'joint_pos','joint_vel','root_pos','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w'}
    for source,item in zip(parent['motions'],plan['motions']):
        for k in ('id','split','source_family','frames'): assert source[k]==item[k]
        assert sha(source['file'])==source['sha256']==item['original_sha256'] and sha(item['file'])==item['sha256']
        assert Path(source['file']).resolve()!=Path(item['file']).resolve()
        with np.load(source['file']) as x,np.load(item['file']) as y:
            assert set(x.files)==set(y.files)
            for k in x.files:
                assert x[k].shape==y[k].shape and x[k].dtype==y[k].dtype
                if k not in changed: assert np.array_equal(x[k],y[k])
            names=y['joint_names'].tolist(); lower=[i for i,n in enumerate(names) if n.startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot'))]
            upper=[i for i in range(30) if i not in lower]; assert len(lower)==14
            assert np.array_equal(x['joint_pos'][:,upper],y['joint_pos'][:,upper])
            q=y['joint_pos'].astype(float); roots=y['root_pos'].astype(float); dt=1/float(y['fps'].item()); assert dt==.02
            assert abs(q-x['joint_pos']).max()<=.8000001
            assert np.all(abs(roots-x['root_pos']).max(axis=0)<=np.array([.06,.06,.031])+1e-7)
            qa=[int(model.joint(n).qposadr[0]) for n in names]; ids=[model.joint(n).id for n in names]; bodies=[model.body(str(n)).id for n in y['body_names']]
            assert (q>=model.jnt_range[ids,0]-1e-7).all() and (q<=model.jnt_range[ids,1]+1e-7).all()
            centers=[]; quats=[]; fk=0.; hmin=float('inf')
            for i in range(item['frames']):
                data.qpos[:]=model.qpos0; data.qpos[:3]=roots[i];data.qpos[3:7]=y['root_quat_wxyz'][i];data.qpos[qa]=q[i];mujoco.mj_kinematics(model,data)
                centers.append(data.xipos[bodies].copy());quats.append(data.xquat[bodies].copy());fk=max(fk,float(abs(data.xpos[bodies]-y['body_pos_w'][i]).max()))
                dot=abs(np.sum(data.xquat[bodies]*y['body_quat_w'][i],axis=1));assert abs(dot-1).max()<2e-6
                for g in range(model.ngeom):
                    if '_collision' not in model.geom(g).name:continue
                    rot=data.geom_xmat[g].reshape(3,3);pos=data.geom_xpos[g]
                    if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX: height=pos[2]-abs(rot[2])@model.geom_size[g]
                    else:
                        assert model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
                        mesh=model.geom_dataid[g];verts=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]];height=(verts@rot.T+pos)[:,2].min()
                    hmin=min(hmin,float(height))
            assert fk<2e-7 and hmin>=-.0005
            # Stored positions are float32; recomputed FK velocities can differ slightly from pre-cast FK.
            linear=np.gradient(np.asarray(centers),dt,axis=0);linerr=float(abs(linear-y['body_lin_vel_w']).max());assert linerr<2e-5
            quat=np.asarray(quats); angular=np.empty_like(linear)
            for b in range(len(bodies)):
                rot=Rotation.from_quat(quat[:,b][:,[1,2,3,0]]);v=(rot[2:]*rot[:-2].inv()).as_rotvec()/(2*dt);angular[:,b]=np.vstack((v[:1],v,v[-1:]))
            angerr=float(abs(angular-y['body_ang_vel_w']).max());assert angerr<2e-5
            jointvel=np.gradient(q,dt,axis=0);jerr=float(abs(jointvel-y['joint_vel']).max());assert jerr<1e-5
            caps=np.array([3. if i in lower else 2. for i in range(30)]);speed=max(float((abs(np.diff(q,axis=0))/dt/caps).max()),float((abs(y['joint_vel'])/caps).max()));assert speed<=1+1e-5/3
            rows.append(dict(id=item['id'],frames=item['frames'],collision_min_m=hmin,fk_error_m=fk,com_velocity_error_mps=linerr,world_angular_error_radps=angerr,joint_velocity_error_radps=jerr,speed_ratio=speed))
    a.output.write_text(json.dumps(dict(passed=True,scope='Exported schema, provenance, FK, velocities and numerical constraints only; no contact dynamics or policy acceptance',plan_sha256=sha(a.plan),asset_sha256=sha(a.asset),rows=rows),indent=2))
    print('EXPORTED_DIAGNOSTIC_REFERENCE_AUDIT_PASSED',len(rows))


if __name__=='__main__':main()
