"""Independent URDF FK/support audit for the derived S2 references."""
import argparse,json,itertools,hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from retarget_humanml3d import CUSTOM_JOINT_NAMES,DEFAULT_TRAINING_URDF,load_limits

def audit(root):
    manifest=json.load(open(root/'retarget_manifest.json'))
    urdf=ET.parse(DEFAULT_TRAINING_URDF).getroot()
    model=pin.buildModelFromUrdf(str(DEFAULT_TRAINING_URDF));cache=model.createData()
    ix=[model.joints[model.getJointId(n)].idx_q for n in CUSTOM_JOINT_NAMES]
    corners=[]
    for name in ('left_foot','right_foot'):
        c=urdf.find(f"link[@name='{name}']/collision")
        size=np.fromstring(c.find('geometry/box').get('size'),sep=' ')
        xyz=np.fromstring(c.find('origin').get('xyz'),sep=' ')
        corners.append(np.array(list(itertools.product((-1,1),repeat=3)))*size/2+xyz)
    lower,upper,vel=load_limits(DEFAULT_TRAINING_URDF)
    rows=[]
    for record in manifest['motions']:
        data=np.load(record['file']);q=data['joint_pos'];root_pos=data['root_pos']
        assert np.isfinite(q).all() and (q>=lower-1e-6).all() and (q<=upper+1e-6).all()
        sole=[];com_margin=[];foot_centers=[]
        for i,frame in enumerate(q):
            ordered=np.zeros(model.nq);ordered[ix]=frame
            pin.framesForwardKinematics(model,cache,ordered)
            com=pin.centerOfMass(model,cache,ordered).copy()
            rotation=Rotation.from_quat(data['root_quat_xyzw'][i]).as_matrix()
            world=[]
            for name,box in zip(('left_foot','right_foot'),corners):
                pose=cache.oMf[model.getFrameId(name)]
                points=(box@pose.rotation.T+pose.translation)@rotation.T+root_pos[i]
                world.append(points)
            sole.append([points[:,2].min() for points in world])
            foot_centers.append([points.mean(0) for points in world])
            com=rotation@com+root_pos[i]
            hull=ConvexHull(np.concatenate(world)[:,:2])
            com_margin.append(float(np.min(-(hull.equations[:,:2]@com[:2]+hull.equations[:,2]))))
        sole=np.array(sole);foot_centers=np.array(foot_centers)
        static=record['id']!='s2_wave_000113'
        if static:
            assert min(com_margin)>0, 'COM projection outside double-support hull'
            assert np.ptp(foot_centers,axis=0).max()<.001, 'Planted feet move >1mm'
            assert np.ptp(sole)<.002, 'Soles cannot share a flat support plane'
        rows.append(dict(id=record['id'],frames=len(q),min_com_projection_margin_m=min(com_margin),
            max_foot_center_axis_range_m=float(np.ptp(foot_centers,axis=0).max()),
            sole_height_range_m=[float(sole.min()),float(sole.max())],
            support_interpretation='Double support geometry only; not dynamic balance proof',
            sha256=hashlib.sha256(Path(record['file']).read_bytes()).hexdigest()))
    report=dict(references=rows,urdf_mass_kg=sum(float(l.find('inertial/mass').get('value')) for l in urdf.findall('link') if l.find('inertial/mass') is not None),
                self_collision_enabled=False, limitations=['Support hull assumes both feet in contact; dynamic contact must be tested in training','Implicit-actuator torque is an estimate; motor selection needs later torque-speed and thermal checks'])
    (root/'geometry_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);audit(p.parse_args().root)
