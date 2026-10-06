import importlib.util,pathlib,sys
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'scripts/practice9'))
from contact_ik import Model,ramps

def test_ramp_never_extends_contact():
 m=np.array([[0,1],[1,1],[1,0],[0,0]],bool);r=ramps(m)
 assert np.all(r[~m]==0) and np.all((r>=0)&(r<=1))

def test_world_corner_jacobian_matches_finite_difference():
 import xml.etree.ElementTree as ET,pinocchio as pin
 from contact_ik import URDF
 tree=ET.parse(URDF);names=[j.get('name') for j in tree.findall('joint') if j.get('type')=='revolute'];bodies=[l.get('name') for l in tree.findall('link')]
 m=Model(names,bodies);q=pin.neutral(m.m);q[2]=.4;q[m.qidx]=.02
 m.fk(q,True);points,J=m.foot(0,True)
 for i in range(11):
  b=q.copy();index=2 if i==0 else m.qidx[m.sel[i-1]];b[index]+=1e-6;m.fk(b)
  np.testing.assert_allclose((m.foot(0)-points)/1e-6,J[:,:,i],atol=5e-7)
