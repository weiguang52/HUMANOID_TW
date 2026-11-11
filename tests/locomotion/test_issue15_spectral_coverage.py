"""Post-freeze supplementary coverage checks, not additional prospective gates."""
import unittest
import numpy as np
from scripts.locomotion.issue15_math import spectrum
class TestFullObservableBand(unittest.TestCase):
 def test_high_frequency_main_peak_not_truncated(self):
  t=np.arange(1000)*.02;x=np.sin(2*np.pi*15*t)
  self.assertAlmostEqual(spectrum(x,lo=1e-12,hi=25.)["peak_hz"],15.)
 def test_low_frequency_main_peak_not_truncated(self):
  t=np.arange(1000)*.02;x=np.sin(2*np.pi*.1*t)
  self.assertAlmostEqual(spectrum(x,lo=1e-12,hi=25.)["peak_hz"],.1)
if __name__=="__main__":unittest.main()
