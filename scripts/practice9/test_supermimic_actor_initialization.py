import unittest
from types import SimpleNamespace
import torch
from supermimic_actor_initialization import zero_readout

class InitializationTests(unittest.TestCase):
 def test_preserves_other_weights_and_optimizer_moments(self):
  class Actor(torch.nn.Module):
   def __init__(self):
    super().__init__();self.encoder=torch.nn.Linear(3,4);self.decoder=torch.nn.Sequential(torch.nn.Linear(4,5),torch.nn.ELU(),torch.nn.Linear(5,2));self.critic=torch.nn.Linear(4,1)
  torch.manual_seed(42);a=Actor();o=torch.optim.Adam(a.parameters(),lr=.01)
  sum(p.square().sum() for p in a.parameters()).backward();o.step();o.zero_grad()
  before={k:v.clone() for k,v in a.state_dict().items()}
  keep=a.decoder[0].weight;moment=o.state[keep]['exp_avg'].clone()
  result=zero_readout(a,o)
  self.assertEqual(result['removed_optimizer_entries'],2)
  self.assertTrue(torch.equal(o.state[keep]['exp_avg'],moment))
  self.assertNotIn(a.decoder[-1].weight,o.state)
  for k,v in a.state_dict().items():
   if not k.startswith('decoder.2.'):self.assertTrue(torch.equal(v,before[k]))
  output=a.decoder(torch.ones(8,4));self.assertTrue(torch.equal(output,torch.zeros_like(output)))
  output.sum().backward();self.assertGreater(float(a.decoder[-1].weight.grad.abs().sum()),0)
 def test_rejects_non_linear_readout(self):
  a=SimpleNamespace(decoder=torch.nn.Sequential(torch.nn.Tanh()))
  with self.assertRaises(ValueError):zero_readout(a,None)

if __name__=='__main__':unittest.main()
