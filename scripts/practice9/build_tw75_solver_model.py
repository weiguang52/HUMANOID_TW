#!/usr/bin/env python3
"""Derive an isolated 28-coordinate native solver from the training geometry.

The two added foot-roll DoFs are held at zero, as in existing retarget output.
No training URDF or original native source is modified.
"""
import argparse, hashlib, json, shutil, subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from audit_tw75_kinematics import fk
from retarget_humanml3d import DEFAULT_ROOT, DEFAULT_TRAINING_URDF, MAPPING, CUSTOM_JOINT_NAMES, map_28_to_30
from native_retarget import NATIVE_JOINT_NAMES


def build(output, foot_orientation_weight=1.0, knee_reference=0.0):
    if not np.isfinite(knee_reference) or not -.5 <= knee_reference <= 0:
        raise ValueError("Diagnostic knee reference must be in [-.5,0]")
    if not np.isfinite(foot_orientation_weight) or not 0 <= foot_orientation_weight <= 1:
        raise ValueError("Foot orientation weight must be in [0,1]")
    output.mkdir(parents=True, exist_ok=True)
    for directory in ['src', 'include', 'tests', 'configs']:
        shutil.copytree(DEFAULT_ROOT/directory, output/directory, dirs_exist_ok=True)
    shutil.copy2(DEFAULT_ROOT/'CMakeLists.txt', output/'CMakeLists.txt')
    tree = ET.parse(DEFAULT_TRAINING_URDF)
    root = tree.getroot()
    links = {}
    for side in ['left', 'right']:
        links.update({f'{side}_mid_leg': f'{side}_calf', f'{side}_calf': f'{side}_ankle',
                      f'{side}_ankle': f'{side}_foot', f'{side}_foot': f'{side}_sole',
                      f'{side}_wrist': f'{side}_hand', f'{side}_hand': f'{side}_hand_tip'})
    for link in root.findall('link'):
        link.set('name', links.get(link.get('name'), link.get('name')))
        # Solver consumes kinematics only; do not carry unresolved mesh paths.
        for tag in ['visual', 'collision']:
            for node in list(link.findall(tag)):
                link.remove(node)
    for joint in root.findall('joint'):
        for tag in ['parent','child']:
            node=joint.find(tag); node.set('link', links.get(node.get('link'), node.get('link')))
        name=joint.get('name')
        if name in MAPPING and MAPPING[name][0] is not None:
            native, sign, offset=MAPPING[name]
            if offset != 0:
                raise ValueError('Nonzero coordinate offsets require explicit origin composition')
            joint.set('name',native)
            axis=joint.find('axis'); axis.set('xyz',' '.join(str(sign*float(x)) for x in axis.get('xyz').split()))
            limit=joint.find('limit')
            lo,hi=sorted(float(limit.get(k))/sign for k in ['lower','upper'])
            limit.set('lower',str(lo)); limit.set('upper',str(hi))
        elif joint.get('type') != 'fixed':
            if name not in ['left_foot_roll','right_foot_roll']:
                raise ValueError('Unexpected unfixed DoF '+name)
            joint.set('type','fixed')
    destination=output/'assets/urdf/Assembly.urdf'
    destination.parent.mkdir(parents=True,exist_ok=True)
    tree.write(destination,encoding='utf-8',xml_declaration=True)
    q=np.zeros(28); q[16]=-np.pi/2; q[21]=np.pi/2
    neutral=fk(destination,dict(zip(NATIVE_JOINT_NAMES,q)))
    lengths={}
    for side in ['left','right']:
        pairs=[('thigh','calf'),('calf','foot'),('upper_arm','force_arm'),('force_arm','hand')]
        lengths[side]=[float(np.linalg.norm(neutral[f'{side}_{b}'][:3,3]-neutral[f'{side}_{a}'][:3,3])) for a,b in pairs]
    head=float(np.linalg.norm(neutral['head'][:3,3]-neutral['neck_linkage'][:3,3]))
    source=(output/'src/robot.cpp').read_text()
    anchor='        if(i==16) p->ref[index]=static_cast<float>(-1.57079632679);'
    if source.count(anchor)!=1:
        raise ValueError('Native reference posture changed')
    source=source.replace(anchor,f'        if(i==3 || i==9) p->ref[index]={knee_reference:.17g};\n'+anchor)
    replacements={
      'p->model.lowerPositionLimit[index]=limits[i][0];':'// Limits are loaded from the derived training URDF.',
      'p->model.upperPositionLimit[index]=limits[i][1];':'',
      'double(limits[i][0])':'p->model.lowerPositionLimit[p->indices[i]]',
      'double(limits[i][1])':'p->model.upperPositionLimit[p->indices[i]]',
      'double(limits[i][0]*factor)':'double(p->model.lowerPositionLimit[p->indices[i]]*factor)',
      'double(limits[i][1]*factor)':'double(p->model.upperPositionLimit[p->indices[i]]*factor)',
      '*.022487329':f'*{head:.17g}',
    }
    for index,old in enumerate(['*.0832','*.1105','*.07579','*.04739']):
        replacements[old]=f'*(side ? {lengths["right"][index]:.17g} : {lengths["left"][index]:.17g})'
    for old,new in replacements.items():
        if source.count(old)!=1:
            raise ValueError('Upstream source changed: '+old)
        source=source.replace(old,new)
    anchor='wr=(k==7||k==8)?1.:(k==9?5.:0.);'
    if source.count(anchor)!=1:
        raise ValueError('Foot orientation task changed')
    source=source.replace(anchor,f'wr=(k==7||k==8)?{foot_orientation_weight:.17g}:(k==9?5.:0.);')
    old_neutral=fk(DEFAULT_ROOT/'assets/urdf/Assembly.urdf',dict(zip(NATIVE_JOINT_NAMES,np.array([(-np.pi/2 if i==16 else np.pi/2 if i==21 else 0.) for i in range(28)]))))
    corrections=[]
    for task,link in [(0,'chest'),(7,'left_foot'),(8,'right_foot'),(9,'head')]:
        rotation=old_neutral[link][:3,:3].T @ neutral[link][:3,:3]
        entries=','.join(f'{v:.17g}' for v in rotation.ravel())
        corrections.append(f'if(k=={task}){{ M3 frame_offset;frame_offset<<{entries};rot=rot*frame_offset; }}')
    anchor='            pinocchio::SE3 goal(rot,pos);'
    if source.count(anchor)!=1:
        raise ValueError('Upstream orientation task changed')
    source=source.replace(anchor,'\n'.join(corrections)+'\n'+anchor)
    (output/'src/robot.cpp').write_text(source)
    rng=np.random.default_rng(42); error=0.
    for _ in range(100):
        q=rng.uniform(-.5,.5,28)
        native=fk(destination,dict(zip(NATIVE_JOINT_NAMES,q)))
        mapped=map_28_to_30(q[None],NATIVE_JOINT_NAMES)[0]
        training=fk(DEFAULT_TRAINING_URDF,dict(zip(CUSTOM_JOINT_NAMES,mapped)))
        for name,T in training.items():
            error=max(error,float(np.max(np.abs(native[links.get(name,name)]-T))))
    if error>1e-10:
        raise ValueError(f'Geometry equivalence failed: {error}')
    metadata={'upstream_commit':subprocess.check_output(['git','-C',str(DEFAULT_ROOT),'rev-parse','HEAD'],text=True).strip(),
              'training_urdf_sha256':hashlib.sha256(DEFAULT_TRAINING_URDF.read_bytes()).hexdigest(),
              'derived_urdf_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
              'foot_orientation_weight':foot_orientation_weight,'knee_reference_rad':knee_reference,
              'fixed_joints':{'left_foot_roll':0,'right_foot_roll':0},'link_renames':links,
              'segment_lengths_m':lengths,'head_length_m':head,'fk_100_pose_max_matrix_error':error}
    (output/'derived_model.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--foot-orientation-weight',type=float,default=1.,help='Diagnostic ablation; 1 preserves selected method')
    parser.add_argument('--knee-reference',type=float,default=0.,help='Diagnostic knee initial/prior pose in radians')
    args=parser.parse_args(); build(args.output,args.foot_orientation_weight,args.knee_reference)
