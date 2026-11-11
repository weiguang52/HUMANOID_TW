import unittest
import numpy as np
import torch
from scripts.locomotion.issue15_math import bias_variance,paired_summary,spectrum,contact_frequencies
from scripts.locomotion.issue15_tracking import yaw_from_quaternion,wrapped_delta
class TestIssue15(unittest.TestCase):
 def test_constant_curve(self):
  x=np.full((1000,3),.4);d=bias_variance(np.full(3,1000),x.sum(0),(x*x).sum(0))
  self.assertAlmostEqual(d["within_energy_fraction"],0);self.assertAlmostEqual(d["mean_signed_bias"],.4)
 def test_oscillation(self):
  x=np.sin(2*np.pi*3*np.arange(1000)*.02);d=bias_variance([1000],[x.sum()],[(x*x).sum()])
  self.assertAlmostEqual(d["within_energy_fraction"],1);self.assertAlmostEqual(spectrum(x)["peak_hz"],3)
 def test_opposite_curves_do_not_cancel(self):
  d=bias_variance([1000,1000],[400,-400],[160,160]);self.assertEqual(d["mean_signed_bias"],0);self.assertAlmostEqual(d["rms_trial_bias"],.4);self.assertAlmostEqual(d["within_energy_fraction"],0)
 def test_target_subtraction(self):
  d=bias_variance([10],[3.],[.9],.3);self.assertAlmostEqual(d["mean_signed_bias"],0)
 def test_unwrap(self):
  a=torch.tensor([-3.1],dtype=torch.float64);b=torch.tensor([3.1],dtype=torch.float64)
  self.assertAlmostEqual(wrapped_delta(a,b).item(),2*np.pi-6.2)
 def test_quaternion(self):
  v=torch.tensor([-.8,0.,2.9],dtype=torch.float64);q=torch.zeros((3,4),dtype=torch.float64);q[:,0]=torch.cos(v/2);q[:,3]=torch.sin(v/2)
  torch.testing.assert_close(yaw_from_quaternion(q),v)
 def test_seed_ci(self):
  d=paired_summary([-1,-1,-1,-1,-1]);self.assertEqual(d["bootstrap95"],[-1,-1]);self.assertIsNone(paired_summary([2])["bootstrap95"])
 def test_contact_step_stride(self):
  t=np.arange(1000)*.02;a=np.stack([np.sin(2*np.pi*2*t)>0,np.sin(2*np.pi*2*t+np.pi)>0],1);d=contact_frequencies(a)
  self.assertAlmostEqual(d["step_hz"],4,delta=.15);self.assertAlmostEqual(d["left_stride_hz"],2,delta=.1)
 def test_short_or_flat_spectrum(self):
  with self.assertRaises(AssertionError):spectrum(np.zeros(249))
  self.assertIsNone(spectrum(np.ones(1000)))
if __name__=="__main__":unittest.main()
