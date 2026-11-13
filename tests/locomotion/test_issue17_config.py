import copy,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/locomotion'))
from issue17_config import configure_issue17
class Tests(unittest.TestCase):
 def test_legacy_noop_and_new_groups_single_difference(self):
  e=NS(unchanged={'reward':3});a=NS(algorithm=NS(symmetry_cfg=None));before=copy.deepcopy((vars(e),vars(a.algorithm)))
  configure_issue17(e,a,'legacy');self.assertEqual((vars(e),vars(a.algorithm)),before)
  c,d=copy.deepcopy(e),copy.deepcopy(a);s,t=copy.deepcopy(e),copy.deepcopy(a)
  configure_issue17(c,d,'control_int');configure_issue17(s,t,'sym_int')
  self.assertEqual(vars(c),vars(s));self.assertTrue(c.issue17_integer_clock);self.assertIsNone(d.algorithm.symmetry_cfg)
  self.assertTrue(t.algorithm.symmetry_cfg['use_data_augmentation']);self.assertFalse(t.algorithm.symmetry_cfg['use_mirror_loss'])
if __name__=='__main__':unittest.main()
