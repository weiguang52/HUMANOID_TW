import importlib.util
from types import SimpleNamespace as NS
from pathlib import Path
import json
import torch
import pytest
spec=importlib.util.spec_from_file_location('stability',Path(__file__).parents[2]/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/stability_tracking.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def env(c):return NS(command_manager=NS(get_term=lambda name:c))
def quat(angle):return torch.tensor([[torch.cos(torch.tensor(angle/2)),0.,0.,torch.sin(torch.tensor(angle/2))]])

def test_heading_wrap_and_sign():
    a=quat(179*torch.pi/180);b=quat(-179*torch.pi/180)
    assert abs(m.yaw_error(a,b).item()+2*torch.pi/180)<1e-6
    assert torch.allclose(m.yaw_error(a,b),m.yaw_error(-a,b))

def test_pelvis_not_chest():
    c=NS(cfg=NS(body_names=['chest','base_link']),body_quat_w=torch.stack([quat(1.)[0],quat(0.)[0]])[None],robot_body_quat_w=torch.stack([quat(0.)[0],quat(0.)[0]])[None])
    assert m.pelvis_heading_cost(env(c),'motion',.2).item()==0

def test_standing_gating(tmp_path):
    p=tmp_path/'m.json';p.write_text(json.dumps({'motions':[{'id':'s','category':'standing_upper'},{'id':'w','category':'walking'}]}))
    c=NS(cfg=NS(motion_file=str(p)),motion=NS(clip_ids=['s','w']),device='cpu',motion_ids=torch.tensor([0,1,0]),anchor_lin_vel_w=torch.tensor([[0.,0.,0.],[0.,0.,0.],[.1,0.,0.]]),anchor_pos_w=torch.tensor([[.01,0.,0.]]*3),robot_anchor_pos_w=torch.zeros(3,3))
    x=m.precise_path_cost(env(c),'motion',.1,.02)
    assert x[0]>x[1] and x[1]==x[2]
    assert torch.allclose(x[1:],m.huber(c.anchor_pos_w[1:,:2],.1))

def test_category_order_rejected(tmp_path):
    p=tmp_path/'m.json';p.write_text(json.dumps({'motions':[{'id':'a'}]}))
    with pytest.raises(ValueError):m.standing_mask(NS(cfg=NS(motion_file=str(p)),motion=NS(clip_ids=['b'])))

def test_neck_reference_not_default():
    c=NS(cfg=NS(joint_names=['head_pitch','other','neck','neck_linkage_roll']),joint_pos=torch.tensor([[.2,4.,.1,.3]]),robot_joint_pos=torch.tensor([[.2,0.,.1,.3]]),joint_vel=torch.tensor([[.4,8.,.2,.6]]),robot_joint_vel=torch.tensor([[.4,0.,.2,.6]]))
    assert m.neck_tracking_cost(env(c),'motion',.2).item()==0
    assert m.neck_tracking_cost(env(c),'motion',1.,True).item()==0
    c.robot_joint_vel[:,0]=0
    assert m.neck_tracking_cost(env(c),'motion',1.,True).item()>0

def test_invalid_scale():
    with pytest.raises(ValueError):m.huber(torch.zeros(1,2),0)
