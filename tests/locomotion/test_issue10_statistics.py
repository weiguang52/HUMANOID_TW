import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/locomotion'))
from issue10_statistics import gate,describe,paired_decline
class BudgetGate(unittest.TestCase):
 def test_decline_but_less_than_half_sd_stops(self):
  a=np.array([.01,.02,.03,.04,.05]);g=gate(a+.001,a)
  self.assertFalse(g['gate_passed']);self.assertAlmostEqual(g['half_sd1500'],a.std(ddof=1)/2)
 def test_large_decline_passes_and_increase_stops(self):
  a=np.array([.01,.02,.03,.04,.05]);self.assertTrue(gate(a+.03,a)['gate_passed']);self.assertFalse(gate(a-.001,a)['gate_passed'])
 def test_equality_does_not_pass(self):
  a=np.array([.01,.02,.03,.04,.05]);self.assertFalse(gate(a+a.std(ddof=1)/2,a)['gate_passed'])
  self.assertFalse(gate([.1]*5,[.1]*5)['gate_passed'])
 def test_joint_pair_bootstrap_and_sd(self):
  a=np.array([.01,.02,.03,.04,.05]);d=paired_decline(a+.03,a)
  np.testing.assert_allclose(d['bootstrap95'],[.03,.03]);self.assertTrue(d['excludes_zero']);s=describe(a)
  self.assertEqual(s['exact_resamples'],3125);self.assertAlmostEqual(s['sample_sd'],np.sqrt(.00025))
 def test_validation(self):
  for x in [[.1]*4,[.1]*4+[float('nan')],[.1]*4+[1.1]]:
   with self.assertRaises(ValueError):gate(x,[.1]*5)
if __name__=='__main__':unittest.main()
