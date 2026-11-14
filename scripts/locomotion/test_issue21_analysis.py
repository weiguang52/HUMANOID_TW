"""Meaningful sign, timing and censoring checks independent of simulator."""
import unittest,numpy as np,json
from issue21_analysis import Mirror,crossings,slope
from issue21_common import REPO
class Checks(unittest.TestCase):
 def test_spatial_and_phase_involution(self):
  fixture=json.loads((REPO/'tests/locomotion/fixtures/issue16_joint_axes.json').read_text());j=[x['joint'] for x in fixture['joints']];m=Mirror(j,['base_link','left_foot','right_foot']);rng=np.random.default_rng(123);x=rng.normal(size=(3,4,101));np.testing.assert_array_equal(m.obs(m.obs(x)),x);c=np.zeros((1,1,3,3));c[0,0,1]=[1,2,3];np.testing.assert_array_equal(m.contact(c)[0,0,2],[1,-2,3])
 def test_crossing_physics_offset_and_component(self):
  x=np.zeros((5,2,3));x[2,1,2]=-.02;r=crossings(x,['x','y','z'],offset=1)
  for row in r:self.assertEqual(row['first']['step'],3);self.assertEqual(row['first']['env'],1);self.assertEqual(row['first']['component'],'z');self.assertIsNone(row['per_env'][0])
 def test_exact_slope_and_censoring(self):
  t=np.arange(1,1001)*.02;y=np.stack([.3*t,-.1*t],axis=1);d=np.zeros((1000,2),bool);v,ok=slope(y,d);np.testing.assert_allclose(v,[.3*180/np.pi,-.1*180/np.pi],rtol=1e-12);d[999,1]=True;v,ok=slope(y,d);self.assertTrue(np.isnan(v[1]));self.assertFalse(ok[1])
if __name__=='__main__':unittest.main()
