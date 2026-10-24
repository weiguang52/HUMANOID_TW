import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[2]
sys.path.insert(0,str(ROOT/'scripts/practice9'))
from refresh_cpp_dataset import rejection
from native_retarget import NativeRetargeter
class RefreshTests(unittest.TestCase):
 def test_projection_failure_is_never_eligible(self):
  self.assertEqual(rejection({'error':'solver failed'}),['solver_error'])
 def test_projection_quality_gates(self):
  a=dict(stance_error_p95=.001,flat_spread_p95=.001,max_q_change=.1,velocity_ratio_max=.8,upper_joint_change=0,joint_limit_violation=0,global_lift_m=0,max_root_z_change=.01,new_leg_acc_max=10,old_leg_acc_max=10,min_sole=.001)
  self.assertEqual(rejection(a),[])
  self.assertIn('penetration',rejection(dict(a,min_sole=-.01)))
  self.assertIn('velocity_ratio_max',rejection(dict(a,velocity_ratio_max=1.2)))
  self.assertIn('acceleration',rejection(dict(a,new_leg_acc_max=60)))
 @unittest.skipUnless(Path('/root/gpufree-data/datasets/practice9/cpp_refresh_20261024/native_training_geometry/build/native/tw_retarget').exists(),'Remote backend required')
 def test_updated_backend_real_motion(self):
  r=NativeRetargeter(Path('/root/gpufree-data/datasets/practice9/cpp_refresh_20261024/native_training_geometry'))
  self.assertEqual(r.provenance['upstream_commit'],'00d4317f6dbd2a64140346c68c916759d0e4fd75')
  x=np.load('/root/gpufree-data/datasets/practice9/humanml3d_rebuild/staging-v1/HumanML3D/new_joints/000041.npy')
  q,info=r.process(x[:20],50)
  self.assertTrue(np.isfinite(q).all());self.assertEqual(q.shape[1],28)
  self.assertLessEqual(info['max_kkt'],1e-6)
if __name__=='__main__':unittest.main()
