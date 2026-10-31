"""Data-integrity and sampler-resume checks, all temporary data on data disk."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from supermimic.algo.anchor_windows import AnchorWindows

class AnchorDataTests(unittest.TestCase):
 def setUp(self):
  base=Path('/root/gpufree-data/tmp/anchor_window_tests');base.mkdir(parents=True,exist_ok=True)
  self.tmp=tempfile.TemporaryDirectory(dir=base);self.root=Path(self.tmp.name)
  self.names=[f'j{x}' for x in range(30)]
  (self.root/'control_mapping.json').write_text(json.dumps(dict(state_joint_names=self.names,action_joint_names=self.names)))
  self.s=np.repeat(np.arange(10,dtype=np.float32)[:,None],73,axis=1)
  self.write_trace()
  (self.root/'results.json').write_text(json.dumps(dict(rows=[dict(split='fit',trace='fit.npz'),dict(split='validation',trace='validation_not_read.npz')])))
 def tearDown(self):self.tmp.cleanup()
 def write_trace(self,broken=False):
  n=self.s+1
  if broken:n[4,0]+=2
  np.savez(self.root/'fit.npz',state=self.s,next_state=n,action=np.zeros((10,30),np.float32),context=np.zeros((10,34),np.float32))
 def load(self):return AnchorWindows(self.root,8,self.names,self.names,'cpu',42)
 def test_fit_only_contiguous_no_padding_and_sampler_resume(self):
  a=self.load();self.assertEqual(a.metadata['windows'],2)
  self.assertEqual([x['file'] for x in a.metadata['fit_files']],['fit.npz'])
  self.assertEqual(tuple(a.states.shape),(2,9,73))
  torch.testing.assert_close(a.states[0],torch.tensor(self.s[:9]))
  rng=a.generator.get_state();first=a.sample(32);a.generator.set_state(rng);second=a.sample(32)
  for x,y in zip(first,second):self.assertTrue(torch.equal(x,y))
 def test_reset_crossing_rejected(self):
  self.write_trace(broken=True)
  with self.assertRaisesRegex(ValueError,'reset|noncontiguous'):self.load()
 def test_split_overlap_rejected(self):
  (self.root/'results.json').write_text(json.dumps(dict(rows=[dict(split='fit',trace='fit.npz'),dict(split='validation',trace='fit.npz')])))
  with self.assertRaisesRegex(ValueError,'split-overlapping'):self.load()
 def test_order_mismatch_rejected(self):
  with self.assertRaisesRegex(ValueError,'ordering'):AnchorWindows(self.root,8,list(reversed(self.names)),self.names,'cpu',42)
 def test_nonfinite_rejected(self):
  self.s[0,0]=np.nan;self.write_trace()
  with self.assertRaisesRegex(ValueError,'nonfinite'):self.load()

if __name__=='__main__':unittest.main()
