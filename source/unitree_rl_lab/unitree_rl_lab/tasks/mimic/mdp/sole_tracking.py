"""URDF collision-box minimum Z, independent of contact-label confidence."""
import itertools,math,xml.etree.ElementTree as ET
import torch


def read_foot_geometry(path):
    root=ET.parse(path).getroot();corners=[];centers=[]
    for name in ['left_foot','right_foot']:
        link=root.find(f"link[@name='{name}']")
        collisions=link.findall('collision')
        if len(collisions)!=1:raise ValueError('Expected one audited foot collision box')
        c=collisions[0];box=c.find('geometry/box');o=c.find('origin')
        if box is None:raise ValueError('Foot collision must be a box')
        xyz=[float(x) for x in o.get('xyz','0 0 0').split()]
        if any(abs(float(x))>1e-9 for x in o.get('rpy','0 0 0').split()):raise ValueError('Rotated collision origin unsupported')
        size=[float(x) for x in box.get('size').split()]
        corners.append([[xyz[j]+sign[j]*size[j]/2 for j in range(3)] for sign in itertools.product([-1,1],repeat=3)])
        centers.append([float(x) for x in link.find('inertial/origin').get('xyz').split()])
    return corners,centers


def rotate(q,v):
    # Quaternion wxyz; broadcasting to [...,feet,corners,3].
    uv=torch.cross(q[...,1:].expand_as(v),v,dim=-1)
    return v+2*(q[...,:1]*uv+torch.cross(q[...,1:].expand_as(v),uv,dim=-1))


def world_corners(pos,quat,corners):
    return pos.unsqueeze(-2)+rotate(quat.unsqueeze(-2),corners.expand(*pos.shape[:-1],8,3))


def sole_height(pos,quat,corners):
    return world_corners(pos,quat,corners)[...,2].min(-1).values


def sole_cost(reference,actual,scale):
    if not math.isfinite(scale) or scale<=0:raise ValueError('Positive scale required')
    return (torch.sqrt(1+((reference-actual)/scale).square())-1).mean(-1)


def sole_tracking_cost(env,command_name,urdf_path,scale,exclude_known_stance=False):
    c=env.command_manager.get_term(command_name)
    if not hasattr(c,'_sole_corners'):
        vertices,_=read_foot_geometry(urdf_path)
        c._sole_corners=torch.tensor(vertices,device=c.device,dtype=c.joint_pos.dtype)
    ix=[c.cfg.body_names.index(n) for n in ['left_foot','right_foot']]
    ref=sole_height(c.body_pos_relative_w[:,ix],c.body_quat_relative_w[:,ix],c._sole_corners)
    actual=sole_height(c.robot_body_pos_w[:,ix],c.robot_body_quat_w[:,ix],c._sole_corners)
    if exclude_known_stance:
        from .expression import reference_contacts
        labels,known=reference_contacts(c)
        return stance_excluded_cost(ref,actual,labels,known,scale)
    return sole_cost(ref,actual,scale)


def stance_excluded_cost(reference,actual,contact,known,scale):
    # Keep denominator fixed at two feet: no amplification of remaining samples.
    if not math.isfinite(scale) or scale<=0:raise ValueError("Positive scale required")
    active=~(contact.bool() & known.bool())
    cost=torch.sqrt(1+((reference-actual)/scale).square())-1
    return (cost*active.to(cost.dtype)).mean(-1)
