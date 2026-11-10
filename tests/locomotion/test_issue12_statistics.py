"""Statistical gates use five paired training seeds and strict frozen inequalities."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/locomotion"))
from issue12_statistics import summary,paired,tracking,paired_delay_rr_change,decision,fall_summary,paired_falls,decision_counts

class StatisticsTests(unittest.TestCase):
    def test_five_seed_enumeration_known_binomial_quantiles(self):
        s=summary([0,0,0,0,1])
        self.assertEqual(s["exact_resamples"],3125)
        np.testing.assert_allclose(s["bootstrap95"],[0,.6])
        with self.assertRaises(ValueError):summary([0,0,1])

    def test_common_indices_preserve_constant_pair_effect(self):
        c=np.array([.1,.2,.3,.4,.5]);t=c-.01
        np.testing.assert_allclose(paired(t,c)["bootstrap95"],[-.01,-.01],atol=1e-15)
        self.assertTrue(paired(t,c)["effective_strictly_negative95"])
        self.assertFalse(paired(c,c)["effective_strictly_negative95"])

    def test_tracking_exact_boundary_and_zero_reference(self):
        self.assertTrue(tracking([1.1]*5,[1.]*5)["not_worse_than_10percent"])
        self.assertFalse(tracking([1.1001]*5,[1.]*5)["not_worse_than_10percent"])
        self.assertTrue(tracking([0]*5,[0]*5)["not_worse_than_10percent"])
        self.assertFalse(tracking([.1]*5,[0]*5)["not_worse_than_10percent"])
        self.assertIsNone(tracking([.1]*5,[0]*5)["ratio_bootstrap95"])

    def test_strict_low_bound_and_mid_tie(self):
        xy={"forward_03":tracking([1]*5,[1]*5)}
        result=decision([.02]*5,[.01]*5,[.03]*5,[.03]*5,xy)
        self.assertTrue(result["frozen_gates"]["low_speed_effect_strictly_negative95"])
        self.assertFalse(result["frozen_gates"]["treatment_low_speed_mean_ci_upper_below_1percent"])
        self.assertTrue(result["frozen_gates"]["mid_speed_treatment_mean_not_above_control"])
        self.assertFalse(result["default_promoted"])

    def test_exact_zero_fall_ci_is_not_made_negative_by_float_subtraction(self):
        c=[10,11,12,13,14];t=[7,8,9,10,16]
        result=paired_falls(t,c,3072)
        self.assertEqual(result["bootstrap95_integer_numerators"][1],0)
        self.assertEqual(result["bootstrap95"][1],0.)
        self.assertFalse(result["effective_strictly_negative95"])

    def test_exact_one_percent_bound_fails_and_integer_mid_count_tie_passes(self):
        xy={"forward_03":tracking([1]*5,[1]*5)}
        result=decision_counts([2]*5,[1]*5,[1,0,0,0,4],[2,1,1,1,0],xy,n=100)
        self.assertFalse(result["frozen_gates"]["treatment_low_speed_mean_ci_upper_below_1percent"])
        self.assertTrue(result["frozen_gates"]["mid_speed_treatment_mean_not_above_control"])
        self.assertEqual(fall_summary([1]*5,n=100)["bootstrap95"],[.01,.01])

    def test_no_rr_pseudocount_when_reference_zero(self):
        a=[[1,100]]*5;z=[[0,100]]*5
        result=paired_delay_rr_change(a,z,a,a)
        self.assertFalse(result["estimable"])
        self.assertIsNone(result["paired_seed_bootstrap95"])
        self.assertEqual(result["zero_reference_bootstrap_replicates"],3125)

    def test_positive_efficacy_does_not_override_bad_tracking(self):
        xy={"forward_03":tracking([1.2]*5,[1]*5)}
        result=decision([.02]*5,[.005]*5,[.03]*5,[.02]*5,xy)
        self.assertEqual(result["decision"],"stop_this_scheme_no_extra_budget_or_weight_scan")
        self.assertFalse(result["all_primary_tracking_gates_passed"])

if __name__=="__main__":unittest.main(verbosity=2)
