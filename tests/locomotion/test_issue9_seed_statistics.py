import sys,unittest,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/locomotion"))
from seed_variance import paired_seed_summary,paired_t_power,required_n
class SeedStatisticsTests(unittest.TestCase):
    def test_pairing_preserves_constant_treatment_offset(self):
        r=np.array([.01,.02,.08,.15,.03]);a=r+.03
        s=paired_seed_summary(r,a)
        self.assertEqual(s['exact_resamples'],3125)
        self.assertTrue(np.allclose(s['paired_applied_minus_requested']['bootstrap95'],[.03,.03],atol=1e-12))
    def test_sample_sd_and_identical_groups(self):
        r=np.array([.01,.02,.03,.04,.05]);s=paired_seed_summary(r,r)
        self.assertAlmostEqual(s['requested']['sample_sd'],math.sqrt(.00025),places=12)
        self.assertEqual(s['paired_applied_minus_requested']['bootstrap95'],[0.,0.])
        self.assertEqual(s['effect_label'],'cannot_distinguish_at_current_seed_count')
    def test_two_sided_null_power_and_planning_monotonicity(self):
        self.assertAlmostEqual(paired_t_power(10,.03,effect=0),.05,places=10)
        self.assertGreater(required_n(.06),required_n(.03))
        self.assertIsNone(required_n(0.))
    def test_roundoff_does_not_create_false_zero_exclusion(self):
        r=np.array([.01,.02,.08,.15,.03]);s=paired_seed_summary(r,r+1e-16)
        self.assertEqual(s["paired_applied_minus_requested"]["bootstrap95"],[0.,0.])
        self.assertFalse(s["difference_ci_excludes_zero"])
    def test_invalid_rates_and_unmatched_pairs(self):
        for r,a in [([.1,.2],[.1]),([-.1,.2],[.1,.2]),([.1,float('nan')],[.1,.2])]:
            with self.assertRaises(ValueError):paired_seed_summary(r,a)
if __name__=="__main__":unittest.main()
