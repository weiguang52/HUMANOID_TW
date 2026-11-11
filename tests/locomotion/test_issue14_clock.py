"""Exact reward-clock boundaries, opt-in isolation and full Adam-state migration."""
import copy,json,sys,unittest
from pathlib import Path
from types import SimpleNamespace as S
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/locomotion"))

class ClockTests(unittest.TestCase):
 def env(self):
  return S(num_envs=3,device="cpu",step_dt=.02,common_step_counter=100)
 def test_never_pushed_zero(self):
  e=self.env()
  for t in [0,100,1000000]:
   e.common_step_counter=t;self.assertTrue(torch.equal(clock.recovery_clock(e),torch.zeros(3,2)))
 def test_every_boundary_matches_reward_clock(self):
  e=self.env();reward.mark_push(e,None)
  for age in range(-2,54):
   e.common_step_counter=100+age;x=clock.recovery_clock(e)
   self.assertTrue(torch.equal(x[:,0].bool(),reward.push_window(e)))
   expected=max(0,1-age/50) if 0<=age<=50 else 0
   torch.testing.assert_close(x[:,1],torch.full((3,),float(expected)))
  e.common_step_counter=150;self.assertEqual(clock.recovery_clock(e)[0].tolist(),[1.,0.])
 def test_reset_partial_and_all(self):
  e=self.env();reward.mark_push(e,None);reward.reset_push_clock(e,torch.tensor([0,2]))
  x=clock.recovery_clock(e);self.assertEqual(x[0].tolist(),[0.,0.]);self.assertEqual(x[1].tolist(),[1.,1.])
  reward.reset_push_clock(e);self.assertEqual(clock.recovery_clock(e).sum(),0)
 def test_repeated_push_selected(self):
  e=self.env();reward.mark_push(e,None);e.common_step_counter=140;reward.mark_push(e,torch.tensor([0]));e.common_step_counter=160
  torch.testing.assert_close(clock.recovery_clock(e),torch.tensor([[1.,.6],[0.,0.],[0.,0.]]))
 def test_no_rng_or_clock_mutation(self):
  e=self.env();reward.mark_push(e,None);old=reward.push_clock(e).clone();rng=torch.get_rng_state().clone()
  clock.recovery_clock(e);self.assertTrue(torch.equal(old,reward.push_clock(e)));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
 def test_default_off_is_exact_noop(self):
  cfg=S(sentinel="same");before=copy.deepcopy(vars(cfg));clock.configure_critic_recovery_clock(cfg,False);self.assertEqual(vars(cfg),before)
 def test_only_critic_term_changes(self):
  from unitree_rl_lab.tasks.locomotion.robots.tw.velocity_env_cfg import RobotEnvCfg
  cfg=RobotEnvCfg();reward.configure_post_push_pitch_reward(cfg,34.67877532210905)
  old=copy.deepcopy(cfg.to_dict());clock.configure_critic_recovery_clock(cfg,True);new=cfg.to_dict()
  self.assertEqual(new["observations"]["policy"],old["observations"]["policy"])
  for key in ["events","rewards","actions","commands","sim"]:self.assertEqual(new[key],old[key])
  term=new["observations"]["critic"].pop("recovery_clock");self.assertEqual(new["observations"],old["observations"]);self.assertEqual(term["params"],{"window_s":1.})
 def test_train_requires_original_reward_clock(self):
  from unitree_rl_lab.tasks.locomotion.robots.tw.velocity_env_cfg import RobotEnvCfg
  with self.assertRaises(ValueError):clock.configure_critic_recovery_clock(RobotEnvCfg(),True)
 def test_evaluation_push_delegates_once_no_rng(self):
  e=self.env();calls=[]
  def original(env,ids,velocity_range):calls.append((ids,velocity_range))
  rng=torch.get_rng_state().clone();clock.clocked_push(e,torch.tensor([1]),original_push=original,velocity_range=7)
  self.assertEqual(len(calls),1);self.assertEqual(calls[0][1],7);self.assertEqual(clock.recovery_clock(e)[1].tolist(),[1.,1.]);self.assertTrue(torch.equal(rng,torch.get_rng_state()))

class MigrationTests(unittest.TestCase):
 def test_full_state_preserved_and_new_columns_learn(self):
  import tempfile
  from tensordict import TensorDict
  from rsl_rl.models import MLPModel
  from issue14_resume import expand_loaded_critic
  torch.manual_seed(14);obs=TensorDict({"policy":torch.randn(3,101),"critic":torch.randn(3,134)},batch_size=[3]);groups={"actor":["policy"],"critic":["critic"]}
  a=MLPModel(obs,groups,"actor",30,hidden_dims=[8,8],obs_normalization=False);c=MLPModel(obs,groups,"critic",1,hidden_dims=[8,8],obs_normalization=False)
  opt=torch.optim.Adam(list(a.parameters())+list(c.parameters()),lr=.001)
  (a(obs).square().mean()+c(obs).square().mean()).backward();opt.step();opt.zero_grad(set_to_none=True)
  storage=S(observations=TensorDict({"policy":torch.zeros(2,3,101),"critic":torch.zeros(2,3,134)},batch_size=[2,3]),step=0)
  alg=S(actor=a,critic=c,optimizer=opt,storage=storage)
  alg.save=lambda:{"actor_state_dict":a.state_dict(),"critic_state_dict":c.state_dict(),"optimizer_state_dict":opt.state_dict()}
  full=obs.clone();full["critic"]=torch.cat((obs["critic"],torch.ones(3,2)),dim=-1)
  base=S(obs_buf=full,observation_manager=S(group_obs_dim={"policy":(101,),"critic":(136,)}))
  with tempfile.TemporaryDirectory() as d:
   result=expand_loaded_critic(S(alg=alg),base,Path(d)/"audit.json");self.assertTrue(result["passed"])
  self.assertEqual(c.mlp[0].weight.shape,(8,136));self.assertEqual(storage.observations["critic"].shape,(2,3,136))
  (a(full).square().mean()+c(full).square().mean()).backward();opt.step()
  self.assertGreater(torch.count_nonzero(c.mlp[0].weight[:,134:]).item(),0);self.assertTrue(torch.isfinite(c.mlp[0].weight).all())

if __name__=="__main__":
 from isaaclab.app import AppLauncher
 app=AppLauncher(headless=True).app
 from unitree_rl_lab.tasks.locomotion.robots.tw import critic_recovery_clock as clock,post_push_recovery as reward
 result=unittest.main(verbosity=2,exit=False);app.close();sys.exit(0 if result.result.wasSuccessful() else 1)
