import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/locomotion"))
import numpy as np
import torch
from issue13_tracking import window_masks,SENTINEL,MomentAccumulator
from issue13_statistics import compare,classify
class RecorderTests(unittest.TestCase):
 def test_no_push_zero(self):
  x=torch.full((3,),SENTINEL)
  for t in [0,1,50,1000]:self.assertFalse(window_masks(t,x)[0].any())
 def test_exact_postevent_and_reward_boundaries(self):
  x=torch.tensor([100])
  for t,expected,other in [(99,False,False),(100,True,False),(101,True,True),(149,True,True),(150,False,True),(151,False,False)]:
   a,b=window_masks(t,x);self.assertEqual(bool(a),expected);self.assertEqual(bool(b),other)
 def test_reset_and_new_episode_push(self):
  x=torch.tensor([100,100]);x[0]=SENTINEL
  self.assertEqual(window_masks(101,x)[0].tolist(),[False,True])
  x[0]=102
  self.assertEqual(window_masks(102,x)[0].tolist(),[True,True])
 def test_moments_reconstruct_known_temporal_variance(self):
  a=MomentAccumulator(1)
  for x,inside in [(0.,True),(2.,True),(4.,False)]:
   a.add(torch.full((1,5),x),torch.tensor([True]),torch.tensor([True]),torch.tensor([inside]),torch.tensor([inside]))
  a.check();self.assertEqual(a.count[0,0,0],3);self.assertEqual(a.sum[0,0,0,2],6);self.assertEqual(a.sumsq[0,0,0,2],20)
  variance=float(a.sumsq[0,0,0,2]/3-(a.sum[0,0,0,2]/3)**2)
  self.assertAlmostEqual(variance,8/3)
 def test_invalid_nan_and_initial_scope_excluded(self):
  a=MomentAccumulator(2);v=torch.tensor([[2.]*5,[float("nan")]*5])
  a.add(v,torch.tensor([True,False]),torch.tensor([False,True]),torch.tensor([True,True]),torch.tensor([True,True]));a.check()
  self.assertEqual(int(a.count[0,0].sum()),1);self.assertEqual(int(a.count[1,0].sum()),0)
class InferenceTests(unittest.TestCase):
 def fixture(self,c):return {str(i):c.copy() for i in range(13)}
 def test_constant_small_degradation_A(self):
  x=compare([1]*5,[1.04]*5);self.assertEqual(classify(self.fixture(x))["decision"],"A_cost_localized_to_recovery_window")
  self.assertAlmostEqual(x["paired_difference"]["bootstrap95"][0],.04)
 def test_any_confirmed_large_degradation_B(self):
  good=compare([1]*5,[1]*5);values=self.fixture(good);values["7"]=compare([1]*5,[1.2]*5)
  self.assertEqual(classify(values)["decision"],"B_global_side_effect")
 def test_uncertain_large_point_is_not_B(self):
  x=compare([1]*5,[.5,.5,.5,.5,4.]);self.assertFalse(x["point_worse_than5percent_and_positive95"])
  self.assertEqual(classify(self.fixture(x))["decision"],"unable_to_distinguish")
 def test_zero_baseline_no_infinity(self):
  x=compare([0]*5,[0]*5);self.assertIsNone(x["ratio_of_group_means"]);self.assertEqual(classify(self.fixture(x))["decision"],"A_cost_localized_to_recovery_window")
 def test_all13_required(self):
  with self.assertRaises(ValueError):classify({})
if __name__=="__main__":unittest.main()
