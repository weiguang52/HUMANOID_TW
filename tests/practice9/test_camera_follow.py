import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/rsl_rl'))
from camera_follow import SmoothCameraFollow

def test_camera_rejects_vertical_and_fast_horizontal_shake():
    camera=SmoothCameraFollow([0,0,.3],.02,.5)
    positions=[]
    for t in np.arange(500)*.02:
        positions.append(camera.update([t*.1+.01*np.sin(2*np.pi*4*t),0,.3+.03*np.sin(2*np.pi*4*t)]))
    positions=np.array(positions)
    assert np.all(positions[:,2]==.3)
    # A following shot retains forward motion without rapid reverse motion.
    assert np.all(np.diff(positions[100:,0])>0)
    np.testing.assert_allclose(camera.update([5,0,.4],reset=True),[5,0,.4])
