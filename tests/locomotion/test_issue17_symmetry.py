import json,sys,unittest
from pathlib import Path
import torch
from tensordict import TensorDict
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/locomotion'))
from issue17_symmetry import TrainingMirror,data_augmentation_func

class Tests(unittest.TestCase):
 def setUp(self):
  fixture=json.loads(Path('tests/locomotion/fixtures/issue16_joint_axes.json').read_text())
  self.names=[r['joint'] for r in fixture['joints']];self.m=TrainingMirror(self.names)
 def test_involution_and_no_mutation(self):
  for size,fn in [(101,self.m.actor),(134,self.m.critic),(30,self.m.action)]:
   x=torch.randn(57,size);before=x.clone();self.assertTrue(torch.equal(fn(fn(x)),x));self.assertTrue(torch.equal(before,x))
 def test_clock_is_half_cycle_only_for_training(self):
  x=torch.randn(19,101);y=self.m.actor(x)
  self.assertTrue(torch.equal(y[:,-2:],-x[:,-2:]));self.assertTrue(torch.equal(y[:,:99],self.m.spatial.observation(x)[:,:99]))
  self.assertTrue(torch.equal(self.m.spatial.observation(x)[:,-2:],x[:,-2:]))
 def test_critic_vectors_and_joint_blocks(self):
  x=torch.arange(134.).repeat(2,1);y=self.m.critic(x)
  for start,signs in [(0,[1,-1,1]),(3,[-1,1,-1]),(6,[1,-1,1]),(9,[1,-1,-1])]:self.assertTrue(torch.equal(y[:,start:start+3],x[:,start:start+3]*torch.tensor(signs)))
  for start in [12,42,72,102]:self.assertTrue(torch.equal(y[:,start:start+30],x[:,start:start+30][:,self.m.spatial.perm]*self.m.spatial.sign))
 def test_tensor_dict_api_optional_inputs_and_rng(self):
  from types import SimpleNamespace
  env=SimpleNamespace(unwrapped=SimpleNamespace(_issue17_training_mirror=self.m))
  obs=TensorDict({'policy':torch.randn(7,101),'critic':torch.randn(7,134)},batch_size=[7]);a=torch.randn(7,30);state=torch.get_rng_state().clone()
  o,b=data_augmentation_func(env,obs,a)
  self.assertEqual(o.batch_size,torch.Size([14]));self.assertTrue(torch.equal(o['policy'][:7],obs['policy']));self.assertTrue(torch.equal(o['critic'][7:],self.m.critic(obs['critic'])));self.assertTrue(torch.equal(b[7:],self.m.action(a)))
  self.assertIsNone(data_augmentation_func(env,None,a)[0]);self.assertIsNone(data_augmentation_func(env,obs,None)[1]);self.assertTrue(torch.equal(state,torch.get_rng_state()))
if __name__=='__main__':unittest.main()
