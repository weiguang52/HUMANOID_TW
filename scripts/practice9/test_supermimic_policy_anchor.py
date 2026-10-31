"""Check measured history alignment and sampler time provenance, without simulation."""
import unittest,tempfile,json
from pathlib import Path
import numpy as np
import torch
from supermimic.algo.anchor_windows import AnchorWindows,policy_history_parts

class PolicyAnchorTests(unittest.TestCase):
 def test_history_is_past_only_with_actual_previous_actions(self):
  s=torch.arange(12,dtype=torch.float32)[None,:,None].expand(1,12,73)
  a=(100+torch.arange(12,dtype=torch.float32))[None,:,None].expand(1,12,30)
  current,prev,past,past_actions=policy_history_parts(s,a,10)
  self.assertEqual(float(current[0,0]),11.)
  self.assertEqual(float(prev[0,0]),110.)
  self.assertEqual(past[0,:,0].tolist(),list(range(1,11)))
  self.assertEqual(past_actions[0,:,0].tolist(),list(range(100,110)))
  self.assertFalse(bool((past[:,:,0]==11).any()))
 def test_incomplete_history_rejected(self):
  with self.assertRaises(ValueError):policy_history_parts(torch.zeros(1,11,73),torch.zeros(1,11,30),10)
 def test_window_time_labels_and_sampler_rng_contract(self):
  root=Path('/root/gpufree-data/tmp/policy_anchor_tests');root.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(dir=root) as temp:
   p=Path(temp);names=[f'j{x}' for x in range(30)]
   (p/'control_mapping.json').write_text(json.dumps(dict(state_joint_names=names,action_joint_names=names)))
   s=np.repeat(np.arange(20,dtype=np.float32)[:,None],73,axis=1)
   np.savez(p/'fit.npz',state=s,next_state=s+1,action=np.repeat((100+np.arange(20,dtype=np.float32))[:,None],30,axis=1),context=np.zeros((20,34),np.float32))
   (p/'results.json').write_text(json.dumps(dict(rows=[dict(split='fit',trace='fit.npz'),dict(split='validation',trace='never_read.npz')])))
   pool=AnchorWindows(p,11,names,names,'cpu',43)
   before=pool.generator.get_state();x=pool.sample_with_times(64);pool.generator.set_state(before);y=pool.sample_with_times(64)
   self.assertTrue(all(torch.equal(a,b) for a,b in zip(x,y)))
   self.assertTrue(torch.equal(x[0][:,-1,0].long(),x[3]))
   self.assertTrue(torch.equal(x[1][:,-2,0].long(),99+x[3]))
   self.assertEqual(len(pool.metadata['fit_files']),1)
   self.assertEqual(pool.states.shape[1],12)

if __name__=='__main__':unittest.main()
