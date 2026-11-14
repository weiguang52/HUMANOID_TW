import unittest
import numpy as np
from issue19_analysis import decompose
class DecompositionTests(unittest.TestCase):
 def fixture(self,degrees):
  y=np.asarray(degrees,dtype=float).reshape(1000,1)*np.pi/180
  return np.arctan2(np.sin(y),np.cos(y)),np.zeros(1),np.ones((1000,2,1,1),dtype=bool)
 def test_fixed_startup_offset(self):
  r=decompose(*self.fixture(np.full(1000,25.)))
  self.assertAlmostEqual(r['startup_signed_degrees'][0],25.)
  self.assertAlmostEqual(r['late_full20s_slope_degrees_per_second'][0],0.,places=9)
 def test_sustained_slope_across_angle_wrap(self):
  t=(np.arange(1000)+1)*.02
  r=decompose(*self.fixture(30*t))
  self.assertAlmostEqual(r['startup_signed_degrees'][0],60.)
  self.assertAlmostEqual(r['late_full20s_slope_degrees_per_second'][0],30.,places=8)
 def test_failure_censors_full_window_and_retains_prefix(self):
  y,s,m=self.fixture((np.arange(1000)+1)*.02*10)
  m[400:,1]=False;m[400,0]=False;y[400:]=0.
  r=decompose(y,s,m)
  self.assertTrue(np.isnan(r['late_full20s_slope_degrees_per_second'][0]))
  self.assertAlmostEqual(r['late_initial_prefix_slope_degrees_per_second'][0],10.,places=8)
  self.assertAlmostEqual(r['initial_prefix_signed_degrees'][0],80.)
 def test_early_failure_retains_exposure_without_fabricating_endpoints(self):
  y,s,m=self.fixture(np.full(1000,7.));m[50:,1]=False;m[50,0]=False
  r=decompose(y,s,m)
  self.assertTrue(np.isnan(r['startup_signed_degrees'][0]))
  self.assertTrue(np.isnan(r['late_initial_prefix_slope_degrees_per_second'][0]))
  self.assertAlmostEqual(r['startup_prefix_seconds'][0],1.)
 def test_reset_prefix_cannot_restart(self):
  y,s,m=self.fixture(np.zeros(1000));m[50:100,1]=False
  with self.assertRaises(AssertionError):decompose(y,s,m)
if __name__=='__main__':unittest.main()
