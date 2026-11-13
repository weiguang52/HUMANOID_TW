"""Synthetic reset/censoring checks for the phase-resolved yaw analysis."""
import unittest
import numpy as np
from issue18_analysis import half_cycles,summary
class HalfCycleTests(unittest.TestCase):
 def data(self):
  delta=np.tile(np.r_[np.full(10,.01),np.full(10,-.01)],50)[:,None]
  yaw=np.cumsum(delta,axis=0);mask=np.ones((1000,2,5,1),bool);phase=np.arange(1000,dtype=np.int16)[:,None]
  return yaw,np.zeros(1),mask,phase
 def test_cancelling_halves(self):
  h,_=half_cycles(*self.data());self.assertEqual(h['complete_cycle_count'],50);self.assertAlmostEqual(h['complete_cycle_left_mean_degrees'],.1*180/np.pi);self.assertAlmostEqual(h['complete_cycle_sum_mean_degrees'],0)
 def test_terminal_transition_and_reset_do_not_form_fake_cycles(self):
  y,start,m,p=self.data();m[15,0]=False;m[15:,1]=False;p[16:,0]=np.arange(984);y[15:,0]+=2.8
  h,r=half_cycles(y,start,m,p);self.assertEqual(h['complete_cycle_count'],0);self.assertEqual(h['partial_cycle_valid_steps'],15);self.assertEqual(h['full20s_trials'],0);self.assertEqual(r['full_horizon_left_steps'][0]+r['full_horizon_right_steps'][0],999)
 def test_phase_misalignment_rejected(self):
  y,s,m,p=self.data();p[10]+=1
  with self.assertRaises(AssertionError):half_cycles(y,s,m,p)
 def test_wrapped_yaw_keeps_signed_increment(self):
  y,s,m,p=self.data();s[:]=3.13;y=np.arctan2(np.sin(y+s),np.cos(y+s));h,_=half_cycles(y,s,m,p);self.assertAlmostEqual(h['complete_cycle_sum_mean_degrees'],0,places=10)
 def test_bootstrap_uses_five_seed_units_and_missing_is_not_dropped(self):
  self.assertEqual(summary([2]*5)['ci95'],[2,2]);self.assertFalse(summary([1,2,None,4,5])['complete'])
if __name__=='__main__':unittest.main()
