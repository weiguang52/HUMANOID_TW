import sys,unittest,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/locomotion'))
from issue11_concentration_cv import greedy_subset
from risk_analysis_utils import push_windows,risk,quartile,directions,phase_bin,seed_effect,AMP_CUTS
class RiskUnits(unittest.TestCase):
 def test_event_window_includes_endpoint_not_past(self):
  ok,y,c=push_windows([0,1,2],[100,100,100],[1,1,1],[150,151,-1]);np.testing.assert_array_equal(y,[1,0,0]);self.assertTrue(ok.all());self.assertFalse(c.any())
 def test_post_failure_and_short_horizon_excluded(self):
  ok,y,c=push_windows([0,0,1],[100,180,951],[1,0,1],[130,-1]);np.testing.assert_array_equal(ok,[1,0,0]);np.testing.assert_array_equal(y,[1,0,0])
 def test_intervening_push_censors_but_earlier_event_counts(self):
  ok,y,c=push_windows([0,0,1,1],[100,120,100,120],[1,1,1,0],[-1,110]);self.assertTrue(c[0]);self.assertFalse(ok[0]);self.assertTrue(ok[2] and y[2])
 def test_all_successes_in_denominator_and_empty_not_zero(self):
  r=risk(2,100);self.assertEqual(r['rate'],.02);self.assertLess(r['wilson95_low'],.02);self.assertGreater(r['wilson95_high'],.02);self.assertIsNone(risk(0,0)['rate'])
  with self.assertRaises(ValueError):risk(2,1)
 def test_wilson_endpoint_roundoff_keeps_zero_and_one(self):
  self.assertEqual(risk(0,100)['wilson95_low'],0.);self.assertEqual(risk(100,100)['wilson95_high'],1.)
 def test_bins_are_outcome_blind_and_directions_signed(self):
  self.assertAlmostEqual(math.pi*AMP_CUTS[-1]**2/(4*.3**2),.75)
  np.testing.assert_array_equal(quartile([.6,.8,1.1,1.4],[.75,1,1.25]),[1,2,3,4]);np.testing.assert_array_equal(directions([[.2,0,0],[-.2,.1,0],[.1,-.2,0]]),['forward','back','side'])
 def test_contact_phase_unknown_is_explicit(self):
  np.testing.assert_array_equal(phase_bin([[.1,.9],[.4,.6],[-1,.2],[.3,.5]],[1,2,0,3]),[1,3,0,2])
 def test_secondary_budget_counts_unique_trials_and_rejects_tiny_cells(self):
  cells={('A',):{'n':20,'k':5,'trial_ids':set(range(4))},('B',):{'n':20,'k':4,'trial_ids':set(range(4))},('C',):{'n':20,'k':3,'trial_ids':set(range(4,8))},('tiny',):{'n':19,'k':19,'trial_ids':{9}}}
  selected,union=greedy_subset(cells,20)
  self.assertEqual(selected,{('A',),('B',)});self.assertEqual(union,set(range(4)))
 def test_training_seed_not_trial_bootstrap_and_zero_ref(self):
  a=np.array([[2,100],[4,100],[6,100],[8,100],[10,100]]);b=np.array([[1,100],[2,100],[3,100],[4,100],[5,100]])
  e=seed_effect(a,b);self.assertEqual(e['exact_resamples'],3125);self.assertAlmostEqual(e['relative_risk'],2);np.testing.assert_allclose(e['relative_risk_bootstrap95'],[2,2]);self.assertIsNone(seed_effect(a,np.column_stack([np.zeros(5),np.ones(5)*100]))['relative_risk'])
if __name__=='__main__':unittest.main()
