import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/practice9'))
from leg_yaw_interpolation import interpolate


def test_orthant_crossing_preserves_endpoints_and_joint_sign():
    q=np.array([[.9,.1,2.],[-.1,-.9,4.]])
    t=np.linspace(0,1,101)
    out=interpolate(q,np.array([0.,1.]),t,[(0,1)])
    assert (out[:,0]*out[:,1]>=0).all()
    np.testing.assert_allclose(out[[0,-1]],q)
    np.testing.assert_allclose(out[50,:2],0)
    np.testing.assert_allclose(out[:,2],2+2*t)


def test_invalid_endpoints_are_not_silently_repaired():
    with pytest.raises(ValueError):
        interpolate(np.array([[.1,-.1],[.2,.2]]),np.array([0.,1.]),np.array([.5]),[(0,1)])
