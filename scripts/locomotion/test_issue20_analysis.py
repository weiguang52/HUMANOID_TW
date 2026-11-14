import unittest,numpy as np
from issue20_analysis import paired,half_cycles
class AnalysisTests(unittest.TestCase):
 def test_exact_mirror_and_D(self):
  y=np.arange(1,1001)[:,None]*np.ones((1,3))*.001;m=np.ones((1000,2,1,3),bool);s=np.zeros(3);v=paired(y,s,m,-y,s,m);np.testing.assert_allclose(v['S20_degrees'],0,atol=1e-12);np.testing.assert_allclose(v['D20_degrees'],180/np.pi,atol=1e-10)
 def test_censoring_excludes_reset_jump(self):
  y=np.zeros((1000,2));m=np.ones((1000,2,1,2),bool);m[100:,1,0,0]=False;m[100,0,0,0]=False;y[100:,0]=2.;v=paired(y,np.zeros(2),m,-y,np.zeros(2),m);self.assertTrue(np.isnan(v['S20_degrees'][0]));self.assertEqual(v['common_prefix_S_degrees'][0],0);self.assertEqual(v['common_prefix_seconds'][0],2.)
 def test_phase10_side_swap(self):
  t=np.arange(1000)[:,None];idx=np.broadcast_to(t%20,(1000,2));inc=np.where(idx<10,.002,-.001);y=np.cumsum(inc,axis=0);m=np.ones((1000,2,1,2),bool);a,_=half_cycles(y,np.zeros(2),m,idx,0);b,_=half_cycles(-y,np.zeros(2),m,(idx+10)%20,10);self.assertAlmostEqual(a['complete_cycle_left_mean_degrees'],-b['complete_cycle_right_mean_degrees']);self.assertAlmostEqual(a['complete_cycle_right_mean_degrees'],-b['complete_cycle_left_mean_degrees'])
if __name__=='__main__':unittest.main()
