"""Build explicitly derived simple references; never relabel rejected sources."""
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from retarget_humanml3d import (CUSTOM_JOINT_NAMES, DEFAULT_TRAINING_URDF,
    load_limits, audit_and_time_scale)
from joint_coordinates import CONTRACT_VERSION


def build(source, output):
    output.mkdir(parents=True, exist_ok=True)
    data = np.load(source, allow_pickle=False)
    if data['joint_names'].tolist() != CUSTOM_JOINT_NAMES:
        raise ValueError('Unexpected source joint order')
    if data['joint_coordinate_contract'][0] != CONTRACT_VERSION:
        raise ValueError('Stale source coordinates')
    lo, hi, velocity = load_limits(DEFAULT_TRAINING_URDF)
    model = pin.buildModelFromUrdf(str(DEFAULT_TRAINING_URDF))
    cache = model.createData()
    indexes = [model.joints[model.getJointId(n)].idx_q for n in CUSTOM_JOINT_NAMES]
    feet = [model.getFrameId(n) for n in ('left_foot', 'right_foot')]
    seed = data['joint_pos'][-1].astype(float).copy()
    seed[27:] = 0  # Procedural stance has a neutral head, not human head tracking.
    def fk(q):
        ordered = np.zeros(model.nq); ordered[indexes] = q
        pin.framesForwardKinematics(model, cache, ordered)
        poses = [cache.oMf[i].copy() for i in feet]
        com = pin.centerOfMass(model, cache, ordered).copy()
        return poses, com
    poses, _ = fk(seed)
    anchors = np.array([p.translation for p in poses])
    anchors[:, 2] = anchors[:, 2].mean()
    def solve(previous, root_xy):
        def residual(x):
            q = seed.copy(); q[:14] = x
            poses, com = fk(q)
            errors = []
            for pose, anchor in zip(poses, anchors):
                errors.extend(30 * (pose.translation + np.r_[root_xy, 0.] - anchor))
                errors.extend(pin.log3(pose.rotation))
            errors.extend(.02 * (x - previous[:14]))
            return errors
        result = least_squares(residual, previous[:14], bounds=(lo[:14]+.04,hi[:14]-.04),
                               max_nfev=150, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        q = seed.copy(); q[:14] = result.x
        poses, com = fk(q)
        error = max(np.linalg.norm(p.translation+np.r_[root_xy,0.]-a) for p,a in zip(poses,anchors))
        rotation = max(np.linalg.norm(pin.log3(p.rotation)) for p in poses)
        if error > .001 or rotation > .01:
            raise ValueError(f'Stance IK not feasible: {error}, {rotation}')
        return q, error, rotation, com + np.r_[root_xy,0.]
    seed[:14] = np.clip(seed[:14],lo[:14]+.041,hi[:14]-.041)
    neutral, _, _, _ = solve(seed, np.zeros(2))
    records = []
    provenance = dict(backend='derived_simple_reference_v1', source_file=str(source),
                      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      urdf_sha256=hashlib.sha256(DEFAULT_TRAINING_URDF.read_bytes()).hexdigest())
    def save(name, q, root, quat, derivation, extra=None):
        q,root,quat,quality = audit_and_time_scale(q,root,quat,lo,hi,velocity,50.,True,4.)
        if not quality['quality_pass']:
            raise ValueError(f'{name}: {quality["reasons"]}')
        path = output / (name+'.retarget.npz')
        np.savez_compressed(path, joint_coordinate_contract=np.array([CONTRACT_VERSION]),
            joint_names=np.array(CUSTOM_JOINT_NAMES),joint_pos=q.astype('float32'),
            root_pos=root.astype('float32'),root_quat_xyzw=quat.astype('float32'),
            fps=np.array([50.],dtype='float32'), quality_pass=np.array([True]),
            quality_json=np.array([json.dumps(quality)]),source_id=np.array([name]),
            mapping_method=np.array([derivation]))
        records.append(dict(id=name,file=str(path.resolve()),frames=len(q),weight=1.,
            quality_pass=True,quality=quality,derivation=derivation,geometry=extra))
    t=np.arange(501)/50.
    urdf = ET.parse(DEFAULT_TRAINING_URDF).getroot()
    sole_offsets = []
    for name in ('left_foot', 'right_foot'):
        collision = urdf.find(f"link[@name='{name}']/collision")
        origin = collision.find('origin')
        if origin.get('rpy', '0 0 0') != '0 0 0':
            raise ValueError('Expected axis-aligned foot box')
        size = np.fromstring(collision.find('geometry/box').get('size'), sep=' ')
        xyz = np.fromstring(origin.get('xyz'), sep=' ')
        sole_offsets.append(xyz[2]-size[2]/2)
    root0=np.array([0.,0.,-anchors[0,2]-min(sole_offsets)])
    quat=np.tile([0.,0.,0.,1.],(len(t),1))
    save('s2_stand',np.tile(neutral,(len(t),1)),np.tile(root0,(len(t),1)),quat,
         'Procedural double-support stance; arms initialized from last source frame; head neutral; legs solved on training URDF')
    qs=[]; roots=[]; errors=[]; rotations=[]; coms=[]; previous=neutral
    for time in t:
        xy=np.array([0.,.008*np.sin(2*np.pi*time/5.)])
        q,e,r,com=solve(previous,xy); previous=q
        qs.append(q);roots.append(root0+np.r_[xy,0.]);errors.append(e);rotations.append(r);coms.append(com)
    save('s2_weight_shift',np.array(qs),np.array(roots),quat,
         'Procedural 8mm lateral pelvis shift, 5s period; double-support foot-pose constrained IK',
         dict(max_foot_position_error_m=max(errors),max_foot_rotation_error_rad=max(rotations),
              com_xy_min=np.min(coms,axis=0)[:2].tolist(),com_xy_max=np.max(coms,axis=0)[:2].tolist()))
    q=data['joint_pos']; margin=.011*(hi-lo)
    valid=((q>lo+margin)&(q<hi-margin)).all(1)
    edges=np.flatnonzero(np.diff(np.r_[False,valid,False]))
    start,end=max(zip(edges[::2],edges[1::2]),key=lambda x:x[1]-x[0])
    if end-start<250: raise ValueError('No sufficiently long interior wave segment')
    root=data['root_pos'][start:end].copy();root[:,:2]-=root[0,:2]
    save('s2_wave_000113',q[start:end].copy(),root,data['root_quat_xyzw'][start:end].copy(),
         f'Unmodified contiguous wave crop [{start}:{end}] from source retarget, excluding neck-limit segment; not a full source replay',
         dict(source_frame_start=int(start),source_frame_end_exclusive=int(end)))
    manifest=dict(schema_version=1,stage='retarget',backend=provenance,
        joint_coordinate_contract=CONTRACT_VERSION,target_fps=50.,joint_names=CUSTOM_JOINT_NAMES,
        quality_gate_version=1,motions=records,failures=[])
    path=output.parent/'retarget_manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(path, [(r['id'],r['frames']) for r in records])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();build(a.source,a.output)
