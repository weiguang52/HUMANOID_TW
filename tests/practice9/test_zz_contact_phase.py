import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[2]
sys.path.insert(0,str(ROOT/'scripts/practice9'))
from contact_labels import infer,resample,from_features

class ContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch
        spec=importlib.util.spec_from_file_location('contact_rewards',ROOT/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/contact_rewards.py')
        cls.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.m)

    def test_treadmill_horizontal_speed_does_not_erase_stance(self):
        a=np.zeros((40,22,3));a[:,:,0]=np.arange(40)[:,None]*.1
        c,k,h=infer(a)
        self.assertTrue(c.all());self.assertTrue(k[2:-2].all())
        self.assertFalse(k[:2].any())

    def test_airborne_uncertainty_not_forced_into_contact(self):
        a=np.zeros((40,22,3));a[10:30,:,1]=.3
        c,k,h=infer(a)
        self.assertFalse(k[12:28].any())

    def test_label_resampling_masks_disagreement(self):
        c=np.array([[1,0],[0,1]],dtype=np.uint8);k=np.ones_like(c)
        out,mask=resample(c,k,3)
        self.assertFalse(mask[1].any());self.assertTrue(mask[[0,2]].all())
        np.testing.assert_array_equal(out[[0,2]],c)

    def test_contact_reward_detects_short_dropouts(self):
        t=self.torch
        f=t.tensor([[[4.,0.],[0.,0.],[4.,0.]]]);c=t.tensor([[1,0]]);k=t.ones((1,2))
        self.assertAlmostEqual(self.m.phase_cost(f,c,k).item(),1/6,places=6)

    def test_perfect_stance_swing_and_unknown(self):
        t=self.torch;f=t.tensor([[[4.,0.]]]);c=t.tensor([[1,0]])
        self.assertEqual(self.m.phase_cost(f,c,t.ones((1,2))).item(),0)
        self.assertEqual(self.m.phase_cost(f,1-c,t.zeros((1,2))).item(),0)
        self.assertEqual(self.m.phase_cost(f,1-c,t.ones((1,2))).item(),1)

    def test_source_bits_are_not_replaced_by_geometry(self):
        joints=np.zeros((40,22,3));features=np.zeros((40,263))
        contact,known,height,raw=from_features(features,joints)
        self.assertFalse(contact.any());self.assertFalse(known.any())
        features[:,-4:]=[1,0,0,1]
        contact,known,height,raw=from_features(features,joints)
        self.assertTrue(contact.all());self.assertTrue(known[2:-2].all())
        np.testing.assert_array_equal(raw,features[:,-4:])

    def test_normalized_or_misaligned_features_rejected(self):
        joints=np.zeros((40,22,3));features=np.zeros((40,263));features[3,-1]=.7
        with self.assertRaises(ValueError):from_features(features,joints)
        with self.assertRaises(ValueError):from_features(features[:-1],joints)

    def test_force_threshold_validation(self):
        with self.assertRaises(ValueError):self.m.phase_cost(None,None,None,0)

    def test_nonfinite_source_rejected(self):
        a=np.zeros((40,22,3));a[0,0,0]=np.nan
        with self.assertRaises(ValueError):infer(a)

if __name__=='__main__':unittest.main()
