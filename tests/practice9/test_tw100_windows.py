import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/practice9'))
from summarize_tw100_validation import windows

class WindowTests(unittest.TestCase):
    def test_tail_kept(self):
        self.assertEqual(list(windows(np.zeros(601,dtype=bool),250)),[(0,250),(250,500),(500,601)])
    def test_no_cross_reset(self):
        a=np.zeros(600,dtype=bool);a[[99,399,599]]=True
        self.assertEqual(list(windows(a,250)),[(0,100),(100,350),(350,400),(400,600)])
    def test_repeated_early_failures_retained(self):
        a=np.ones(3,dtype=bool)
        self.assertEqual(list(windows(a,250)),[(0,1),(1,2),(2,3)])
    def test_windows_serialize_with_numpy_reset_indices(self):
        import json
        a=np.zeros(270,dtype=bool);a[100]=True
        json.dumps(list(windows(a,250)))

    def test_empty(self):
        self.assertEqual(list(windows(np.zeros(0,dtype=bool),250)),[])
if __name__=='__main__':unittest.main()
