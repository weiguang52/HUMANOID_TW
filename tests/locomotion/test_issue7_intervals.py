import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/locomotion"))
from confidence_stats import wilson
from scipy.stats import binomtest
class IntervalTests(unittest.TestCase):
    def test_against_scipy_at_boundaries_and_interior(self):
        for n in [128,1024,3072]:
            for k in [0,1,2,n//2,n-1,n]:
                expected=binomtest(k,n).proportion_ci(confidence_level=.95,method="wilson")
                actual=wilson(k,n)
                self.assertAlmostEqual(actual[0],expected.low,places=12)
                self.assertAlmostEqual(actual[1],expected.high,places=12)
    def test_invalid_counts(self):
        for k,n in [(-1,128),(129,128),(0,0)]:
            with self.assertRaises(ValueError):wilson(k,n)
if __name__=="__main__":unittest.main()
